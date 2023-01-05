/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/Cost.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

using namespace facebook::velox;

Optimization::Optimization(const core::PlanNode& plan, const Schema& schema)
    : schema_(schema), inputPlan_(plan) {
  root_ = makeQueryGraph();
  root_->expandJoins();
}

RelationOpPtr Optimization::bestPlan() {
  PlanState state;
  state.dt = root_;
  makeJoins(nullptr, state);
  Distribution empty;
  bool ignore;
  return state.plans.best(empty, ignore)->op;
}

std::unordered_map<std::string, float>& baseSelectivities() {
  static std::unordered_map<std::string, float> map;
  return map;
}

FunctionSet functionBits(Name name) {
  return FunctionSet(0);
}

Plan::Plan(RelationOpPtr _op, const PlanState& state)
    : op(_op),
      unitCost(state.cost),
      setupCost(state.setupCost),
      fanout(state.fanout),
      tables(state.placed),
      columns(state.targetColumns) {}

bool Plan::isStateBetter(const PlanState& state) const {
  return unitCost * plannedInputCardinality + setupCost >
      state.cost * state.inputCardinality + state.setupCost;
}

void PlanState::addCost(const RelationOp& op) {
  cost += inputCardinality * fanout * op.unitCost;
  setupCost += op.setupCost;
  fanout *= op.fanout;
}

PlanObjectSet PlanState::downstreamColumns() const {
  auto it = downstreamPrecomputed.find(placed);
  if (it != downstreamPrecomputed.end()) {
    return it->second;
  }
  PlanObjectSet result;
  for (auto join : dt->joins) {
    int32_t numNeeded = 0;
    if (!placed.contains(join->rightTable)) {
      ++numNeeded;
      result.unionColumns(join->leftKeys);
    }
    if (join->leftTable && !placed.contains(join->leftTable)) {
      ++numNeeded;
      result.unionColumns(join->rightKeys);
    }
    if (numNeeded && join->filter) {
      result.unionSet(join->filter->columns);
    }
  }
  result.unionSet(targetColumns);
  downstreamPrecomputed[placed] = result;
  return result;
}

void PlanSet::addPlan(RelationOpPtr plan, PlanState& state) {
  bool insert = plans.empty();
  int32_t replaceIndex = -1;
  if (!insert) {
    // Compare with existing. If there is one with same distribution
    // and new is better, replace. If there is one with a different
    // distribution and the new one can produce the same distribution
    // by repartition, for cheaper, add the new one and delete the old
    // one.
    for (auto i = 0; i < plans.size(); ++i) {
      auto old = plans[i];
      if (!(state.input == old->input)) {
        continue;
      }
      if (!old->isStateBetter(state)) {
        continue;
      }
      if (old->op->distribution.isSamePartition(plan->distribution)) {
        replaceIndex = i;
        continue;
      }
    }
  }
  if (insert || replaceIndex != -1) {
    Define(Plan, newPlan, plan, state);
    if (replaceIndex < 0) {
      plans[replaceIndex] = newPlan;
    } else {
      plans.push_back(newPlan);
    }
  }
}

PlanPtr PlanSet::best(const Distribution& distribution, bool& needsShuffle) {
  PlanPtr best = nullptr;
  PlanPtr match = nullptr;
  float bestCost = -1;
  float matchCost = -1;
  for (auto i = 0; i < plans.size(); ++i) {
    float cost = plans[i]->fanout * plans[i]->unitCost + plans[i]->setupCost;
    if (!best || bestCost > cost) {
      best = plans[i];
      bestCost = cost;
    }
    if (plans[i]->op->distribution.isSamePartition(distribution)) {
      match = plans[i];
      matchCost = cost;
    }
  }
  if (best != match && match) {
    float shuffle = shuffleCost(best->op->columns) * best->fanout;
    if (bestCost + shuffle < matchCost) {
      needsShuffle = true;
      return best;
    }
  }
  needsShuffle = best != match;
  return best;
}

float startingScore(PlanObjectPtr table, DerivedTablePtr dt) {
  if (table->type == PlanType::kTable) {
    return table->as<BaseTablePtr>()
        ->schemaTable->indices[0]
        ->distribution.cardinality;
  }
  return 10;
}

std::pair<PlanObjectPtr, float> otherTable(JoinPtr join, PlanObjectPtr table) {
  return join->leftTable == table && !join->leftOptional && !join->leftExists
      ? std::pair<PlanObjectPtr, float>{join->rightTable, join->lrFanout}
      : join->rightTable == table && !join->rightOptional && !join->rightExists
      ? std::pair<PlanObjectPtr, float>{join->leftTable, join->rlFanout}
      : std::pair<PlanObjectPtr, float>{nullptr, 0};
}

std::vector<PlanObjectPtr>
joinedTables(DerivedTablePtr dt, PlanObjectPtr from, PlanObjectSet except) {
  std::vector<PlanObjectPtr> result;
  PlanObjectSet set;
  for (auto join : dt->joins) {
    set.unionSet(allTables(toRange(join->leftKeys)));
    set.unionSet(allTables(toRange(join->rightKeys)));
  }
  return set.objects();
}

const JoinVector& joinedBy(PlanObjectPtr table) {
  if (table->type == PlanType::kTable) {
    return table->as<BaseTablePtr>()->joinedBy;
  }
  VELOX_DCHECK_EQ(table->type, PlanType::kDerivedTable);
  return table->as<DerivedTablePtr>()->joinedBy;
}

// Calls 'func' with join, joined table and fanout for the joinable tables.
template <typename Func>
void forJoinedTables(DerivedTablePtr dt, const PlanState& state, Func func) {
  PlanObjectSet joinable;
  state.placed.forEach([&](PlanObjectPtr placedTable) {
    for (auto join : joinedBy(placedTable)) {
      auto [table, fanout] = otherTable(join, placedTable);
      if (table) {
        func(join, table, fanout);
      } else {
        if (!join->leftTable) {
          bool usable = true;
          for (auto key : join->leftKeys) {
            if (!state.placed.isSubset(key->allTables())) {
              usable = false;
              break;
            }
          }
          if (usable) {
            func(join, join->rightTable, join->lrFanout);
          }
        }
      }
    }
  });
}

JoinCandidate
reducingJoins(DerivedTablePtr dt, const PlanState& state, PlanObjectPtr start) {
  PlanObjectSet selected;
  PlanObjectSet visited = state.placed;
  // Look for reducing joins against tables not in the plan.
#if 0
  reducingJoinsRecursive(dt, start, selected, visited);
  visited = selected;
  forDirectJoins(table, [&](JoinPtr join, PlanObjectPtr other) {
    
  });
#endif
  return JoinCandidate();
}

JoinSide JoinCandidate::sideOf(PlanObjectPtr side, bool other) const {
  if (side == join->rightTable && !other) {
    return {
        join->rightTable,
        join->rightKeys,
        join->rightOptional,
        join->rightExists,
        join->rightNotExists};
  }
  return {
      join->leftTable,
      join->leftKeys,
      join->leftOptional,
      join->leftExists,
      false};
}

std::vector<JoinCandidate> Optimization::nextJoins(
    DerivedTablePtr dt,
    PlanState& state) {
  std::vector<JoinCandidate> candidates;
  forJoinedTables(
      dt, state, [&](JoinPtr join, PlanObjectPtr joined, float fanout) {
        candidates.emplace_back(join, joined, fanout);
      });

  std::vector<JoinCandidate> bushes;
  // Take the  first hand joined tables and bundle them with reducing joins that
  // can go on the build side.
  for (auto& candidate : candidates) {
    auto bush = reducingJoins(dt, state, candidate.tables[0]);
    if (!bush.tables.empty()) {
      bushes.push_back(std::move(bush));
    }
  }
  candidates.insert(candidates.begin(), bushes.begin(), bushes.end());
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const JoinCandidate& left, const JoinCandidate& right) {
        return left.fanout < right.fanout;
      });
  return candidates;
}

size_t MemoKey::hash() const {
  size_t hash = tables.hash();
  for (auto& exists : existences) {
    hash = bits::commutativeHashMix(hash, exists.hash());
  }
  return hash;
}

bool MemoKey::operator==(const MemoKey& other) const {
  if (columns == other.columns && tables == other.tables) {
    if (existences.size() != other.existences.size()) {
      return false;
    }
    for (auto& e : existences) {
      bool found = true;
      for (auto& e2 : other.existences) {
        if (e2 == e) {
          found = true;
          break;
        }
        if (!found) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

void Optimization::addPostprocess(
    DerivedTablePtr dt,
    RelationOpPtr& plan,
    PlanState& state) {
  if (dt->aggregation) {
    Define(Aggregation, newGroupBy, *dt->aggregation);
  }
}

std::vector<IndexPtr> chooseLeafIndex(BaseTablePtr table, DerivedTablePtr dt) {
  return {table->schemaTable->indices[0]};
}

template <typename V>
PtrSpan<Column> leadingColumns(V& exprs) {
  int32_t i = 0;
  for (; i < exprs.size(); ++i) {
    if (exprs[i]->type != PlanType::kColumn) {
      break;
    }
  }
  return PtrSpan<Column>(reinterpret_cast<ColumnPtr*>(&exprs[0]), i);
}

bool isIndexColocated(
    const IndexInfo& info,
    const ExprVector& lookupValues,
    RelationOpPtr input) {
  if (info.index->distribution.isBroadcast &&
      input->distribution.distributionType.locus ==
          info.index->distribution.distributionType.locus) {
    return true;
  }

  // True if 'input' is partitioned so that each partitioning key is joined to
  // the corresponding partition key in 'info'.
  if (!(input->distribution.distributionType ==
        info.index->distribution.distributionType)) {
    return false;
  }
  if (input->distribution.partition.empty()) {
    return false;
  }
  if (input->distribution.partition.size() !=
      info.index->distribution.partition.size()) {
    return false;
  }
  for (auto i = 0; i < input->distribution.partition.size(); ++i) {
    auto nthKey = position(lookupValues, *input->distribution.partition[i]);
    if (nthKey >= 0) {
      if (info.lookupKeys[nthKey] != info.index->distribution.partition[i]) {
        return false;
      }
    }
  }
  return true;
}

RelationOpPtr repartitionForIndex(
    const IndexInfo& info,
    const ExprVector& lookupValues,
    RelationOpPtr plan,
    PlanState& state) {
  if (isIndexColocated(info, lookupValues, plan)) {
    return plan;
  }
  ExprVector keyExprs{stl<ExprPtr>()};
  auto& partition = info.index->distribution.partition;
  for (auto key : partition) {
    auto nthKey = position(info.lookupKeys, *key);
    if (nthKey >= 0) {
      keyExprs.push_back(lookupValues[nthKey]);
    } else {
      return nullptr;
    }
  }

  Define(Repartition, repartition);
  repartition->distribution.partition = std::move(keyExprs);
  repartition->distribution.distributionType =
      info.index->distribution.distributionType;
  repartition->columns = plan->columns;
  return repartition;
}

void Optimization::joinByIndex(
    RelationOpPtr plan,
    const JoinCandidate& candidate,
    PlanState& state) {
  if (candidate.tables[0]->type != PlanType::kTable) {
    return;
  }
  auto right = candidate.tables[0]->as<BaseTablePtr>();
  auto left = candidate.sideOf(right, true);
  auto& keys = candidate.join->leftTable == right ? candidate.join->leftKeys
                                                  : candidate.join->rightKeys;
  auto keyColumns = leadingColumns(keys);
  if (keyColumns.empty()) {
    return;
  }
  for (auto& index : right->schemaTable->indices) {
    auto info = right->schemaTable->indexInfo(index, keyColumns);
    if (info.lookupKeys.empty()) {
      continue;
    }
    auto newPartition = repartitionForIndex(info, left.keys, plan, state);
    if (!newPartition) {
      continue;
    }
    StateSaver save(state);
    if (newPartition != plan) {
      state.addCost(*newPartition);
    }
    Define(TableScan, scan);
    scan->input = newPartition;
    scan->index = info.index;
    scan->keys = left.keys;
    appendToVector(scan->keys, info.lookupKeys);
    scan->fanout = info.scanCardinality;
    state.columns.unionSet(scan->availableColumns());

    PlanObjectSet c = state.downstreamColumns();
    c.intersect(state.columns);

    state.placed.add(candidate.tables[0]);
    c.forEach(
        [&](auto o) { scan->columns.push_back(reinterpret_cast<Column*>(o)); });
    if (scan->baseTable->filter) {
      scan->extractedColumns.unionSet(scan->baseTable->filter->columns);
    }
    auto join = candidate.join;
    if (join->filter) {
      scan->joinFilter = join->filter;
      scan->extractedColumns.unionSet(join->filter->columns);
      if (join->rightOptional) {
        scan->joinType = velox::core::JoinType::kLeft;
      } else if (join->rightExists)
        scan->joinType = velox::core::JoinType::kLeftSemiFilter;
    } else if (join->rightNotExists) {
      scan->joinType = velox::core::JoinType::kAnti;
    }
    state.addCost(*scan);
    makeJoins(scan, state);
  }
}

// Returns the positions in 'keys' for the expressions that determine the
// partition. empty if the partition is not decided by 'keys'
std::vector<int32_t> joinKeyPartition(
    RelationOpPtr op,
    const ExprVector& keys) {
  std::vector<int32_t> positions;
  for (auto i = 0; i < op->distribution.partition.size(); ++i) {
    auto nthKey = position(keys, *op->distribution.partition[i]);
    if (nthKey < 0) {
      return {};
    }
    positions.push_back(nthKey);
  }
  return positions;
}

PlanObjectSet availableColumns(PlanObjectPtr object) {
  PlanObjectSet set;
  if (object->type == PlanType::kTable) {
    for (auto& c : object->as<BaseTablePtr>()->columns) {
      set.add(c);
    }
  }
  return set;
}

bool isBroadcastable(PlanPtr build, PlanState& /*state*/) {
  return build->fanout < 100000;
}

void Optimization::joinByHash(
    RelationOpPtr plan,
    const JoinCandidate& candidate,
    PlanState& state) {
  auto build = candidate.sideOf(candidate.tables[0]);
  auto probe = candidate.sideOf(candidate.tables[0], true);
  auto partKeys = joinKeyPartition(plan, probe.keys);
  Distribution distribution;
  // If every partition key of the probe side is a join key, then the build
  // should be partitioned by the partition of the probe.
  for (auto nthKey : partKeys) {
    distribution.partition.push_back(build.keys[nthKey]);
  }
  if (partKeys.empty()) {
    // Prefer to make a build partitioned on join keys and shuffle probe to
    // align with build.
    distribution.partition = build.keys;
  }
  StateSaver save(state);
  PlanObjectSet buildTables;
  PlanObjectSet buildColumns;
  for (auto build : candidate.tables) {
    buildColumns.unionSet(availableColumns(build));
    state.placed.add(build);
    buildTables.add(build);
  }
  auto downstream = state.downstreamColumns();
  buildColumns.intersect(downstream);
  auto key = MemoKey{buildColumns, buildTables, candidate.existences};
  PlanObjectSet empty;
  bool needsShuffle = false;
  auto buildPlan = makePlan(key, distribution, empty, state, needsShuffle);
  bool partitionByProbe = !partKeys.empty();
  RelationOpPtr buildInput = nullptr;
  RepartitionPtr buildShuffle = nullptr;
  HashBuildPtr buildOp = nullptr;
  RelationOpPtr probeInput = plan;
  RepartitionPtr probeShuffle = nullptr;
  JoinOpPtr joinOp = nullptr;
  if (partitionByProbe) {
    if (needsShuffle) {
      Define(Repartition, shuffleTemp);
      buildShuffle = shuffleTemp;
      buildShuffle->distribution.distributionType =
          plan->distribution.distributionType;
      buildShuffle->distribution.partition = distribution.partition;
      buildShuffle->columns = buildPlan->op->columns;
      buildInput = buildShuffle;
    }
  } else if (isBroadcastable(buildPlan, state)) {
    Define(Repartition, broadcast);
    broadcast->distribution.isBroadcast = true;
    buildInput = broadcast;
    Define(JoinOp, joinTemp);
    joinOp = joinTemp;

  } else {
    // The probe gets shuffled to align with build. If build is not partitioned
    // on its keys, shuffle the build too.
    auto buildPart = joinKeyPartition(buildPlan->op, build.keys);
    buildInput = buildPlan->op;
    if (buildPart.empty()) {
      // The build is not aligned on join keys.
      Define(Repartition, shuffleTemp);
      buildShuffle = shuffleTemp;
      if (buildPart.empty()) {
        buildShuffle->distribution.distributionType =
            plan->distribution.distributionType;
        buildShuffle->distribution.partition = build.keys;
        buildInput = buildShuffle;
      }
    }
    Define(Repartition, probeTemp);
    probeShuffle = probeTemp;
    probeShuffle->input = plan;
    probeShuffle->distribution.distributionType =
        probeInput->distribution.distributionType;
    for (auto i = 0; i < probeInput->distribution.partition.size(); ++i) {
      auto key = buildInput->distribution.partition[i];
      auto nthKey = position(build.keys, *key);
      probeShuffle->distribution.partition.push_back(probe.keys[nthKey]);
    }
    probeInput = probeShuffle;
  }
  Define(HashBuild, buildTemp);
  buildOp = buildTemp;
  buildOp->input = buildInput;
  buildOp->keys = build.keys;
  Define(JoinOp, joinTemp);
  joinOp = joinTemp;
  joinOp->input = probeInput;
  joinOp->right = buildOp;
  joinOp->leftKeys = probe.keys;
  joinOp->rightKeys = build.keys;
  joinOp->distribution.distributionType =
      probeInput->distribution.distributionType;
  joinOp->distribution.partition = probeInput->distribution.partition;
  downstream.forEach([&](auto object) {
    joinOp->columns.push_back(reinterpret_cast<ColumnPtr>(object));
  });
  auto buildCost = buildPlan->unitCost;
  if (buildShuffle) {
    buildShuffle->setCost();
    buildCost += buildShuffle->unitCost * buildPlan->fanout;
  }
  buildOp->setCost();
  buildCost += buildOp->unitCost * buildPlan->fanout;
  if (probeShuffle) {
    state.addCost(*probeShuffle);
  }
  state.addCost(*joinOp);
  state.setupCost += buildCost;
  makeJoins(joinOp, state);
}

void Optimization::addJoin(
    DerivedTablePtr dt,
    const JoinCandidate& candidate,
    RelationOpPtr plan,
    PlanState& state) {
  joinByIndex(plan, candidate, state);
  joinByHash(plan, candidate, state);
}

void Optimization::makeJoins(RelationOpPtr plan, PlanState& state) {
  auto& dt = state.dt;
  if (!plan) {
    std::vector<float> scores(dt->tables.size());
    for (auto i = 0; i < dt->tables.size(); ++i) {
      auto table = dt->tables[i];
      scores[i] = startingScore(table, dt);
    }
    std::vector<int32_t> ids(dt->tables.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int32_t left, int32_t right) {
      return scores[left] > scores[right];
    });
    for (auto i : ids) {
      auto from = dt->tables[i];
      if (from->type == PlanType::kTable) {
        auto table = from->as<BaseTablePtr>();
        auto indices = chooseLeafIndex(table->as<BaseTablePtr>(), dt);
        // Make plan starting with each relevant index of the table.
        for (auto index : indices) {
          StateSaver save(state);
          state.placed.add(table);
          Define(TableScan, scan);
          scan->relType = RelType::kTableScan;
          scan->baseTable = table;
          scan->index = index;
          scan->setRelation(table->columns, table->schemaColumns);
          state.addCost(*scan);
          makeJoins(scan, state);
        }
      } else {
        // Start with a derived table.
        VELOX_NYI();
      }
    }
  } else {
    auto candidates = nextJoins(dt, state);
    if (candidates.empty()) {
      addPostprocess(dt, plan, state);
      state.plans.addPlan(plan, state);
    }
    for (auto& candidate : candidates) {
      addJoin(dt, candidate, plan, state);
    }
  }
}

PlanPtr Optimization::makePlan(
    const MemoKey& key,
    const Distribution& distribution,
    const PlanObjectSet& boundColumns,
    PlanState& state,
    bool needsShuffle) {
  auto it = memo_.find(key);
  PlanSet* plans;
  if (it == memo_.end()) {
    DerivedTable dt;
    dt.import(*state.dt, key.tables, key.existences);
    PlanState inner;
    inner.targetColumns = key.columns;
    inner.dt = &dt;
    makeJoins(nullptr, inner);
    memo_[key] = std::move(inner.plans);
    plans = &memo_[key];
  } else {
    plans = &it->second;
  }
  return plans->best(distribution, needsShuffle);
}

} // namespace facebook::verax
