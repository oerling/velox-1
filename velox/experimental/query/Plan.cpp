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

#include <iostream>

namespace facebook::verax {

using namespace facebook::velox;
using facebook::velox::core::JoinType;

Optimization::Optimization(
    const core::PlanNode& plan,
    const Schema& schema,
    int32_t traceFlags)
    : schema_(schema), inputPlan_(plan), traceFlags_(traceFlags) {
  root_ = makeQueryGraph();
  root_->addImpliedJoins();
  root_->linkTablesToJoins();
  setDerivedTableOutput(root_, inputPlan_);
}

void Optimization::trace(
    int32_t event,
    int32_t id,
    const Cost& cost,
    RelationOp& plan) {
  if (event & traceFlags_) {
    std::cout << (event == kRetained ? "Retained: " : "Abandoned: ") << id
              << " " << succinctNumber(cost.unitCost + cost.setupCost) << " "
              << plan.toString(true, false) << std::endl;
  }
}

PlanPtr Optimization::bestPlan() {
  topState_.dt = root_;
  for (auto expr : root_->exprs) {
    topState_.targetColumns.unionColumns(expr);
  }
  makeJoins(nullptr, topState_);
  Distribution empty;
  bool ignore;
  return topState_.plans.best(empty, ignore);
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
      cost(state.cost),
      tables(state.placed),
      columns(state.targetColumns) {}

bool Plan::isStateBetter(const PlanState& state) const {
  return cost.unitCost * cost.inputCardinality + cost.setupCost >
      state.cost.unitCost * state.cost.inputCardinality + state.cost.setupCost;
}

std::string Plan::printCost() const {
  return cost.toString(true, false);
}

std::string Plan::toString(bool detail) const {
  queryCtx()->contextPlan() = const_cast<Plan*>(this);
  auto result = op->toString(true, detail);
  queryCtx()->contextPlan() = nullptr;
  return result;
}

void PlanState::addCost(RelationOp& op) {
  if (!op.cost().unitCost) {
    op.setCost(*this);
  }
  cost.unitCost += cost.inputCardinality * cost.fanout * op.cost().unitCost;
  cost.setupCost += op.cost().setupCost;
  cost.fanout *= op.cost().fanout;
}

bool PlanState::addNextJoin(
    const JoinCandidate* candidate,
    RelationOpPtr plan,
    BuildSet builds,
    std::vector<NextJoin>& toTry) const {
  if (!isOverBest()) {
    toTry.emplace_back(candidate, plan, cost, placed, builds);
    return true;
  }
  return false;
}

void PlanState::addBuilds(const BuildSet& added) {
  for (auto build : added) {
    if (std::find(builds.begin(), builds.end(), build) == builds.end()) {
      builds.push_back(build);
    }
  }
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

std::string PlanState::printCost() const {
  return cost.toString(true, true);
}

std::string PlanState::printPlan(RelationOpPtr op, bool detail) const {
  auto plan = std::make_unique<Plan>(op, *this);
  return plan->toString(detail);
}

PlanPtr PlanSet::addPlan(RelationOpPtr plan, PlanState& state) {
  bool insert = plans.empty();
  int32_t replaceIndex = -1;
  if (!insert) {
    // Compare with existing. If there is one with same distribution
    // and new is better, replace. If there is one with a different
    // distribution and the new one can produce the same distribution
    // by repartition, for cheaper, add the new one and delete the old
    // one.
    for (auto i = 0; i < plans.size(); ++i) {
      auto old = plans[i].get();
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
    auto newPlan = std::make_unique<Plan>(plan, state);
    auto result = newPlan.get();
    if (replaceIndex >= 0) {
      plans[replaceIndex] = std::move(newPlan);
    } else {
      plans.push_back(std::move(newPlan));
    }
    if (!bestPlan ||
        bestPlan->cost.unitCost + bestPlan->cost.setupCost >
            result->cost.unitCost + result->cost.setupCost) {
      bestPlan = result;
      bestCostWithShuffle = result->cost.unitCost + result->cost.setupCost +
          shuffleCost(result->op->columns) * result->cost.fanout;
    }
    return result;
  }
  return nullptr;
}

PlanPtr PlanSet::best(const Distribution& distribution, bool& needsShuffle) {
  PlanPtr best = nullptr;
  PlanPtr match = nullptr;
  float bestCost = -1;
  float matchCost = -1;
  for (auto i = 0; i < plans.size(); ++i) {
    float cost = plans[i]->cost.fanout * plans[i]->cost.unitCost +
        plans[i]->cost.setupCost;
    if (!best || bestCost > cost) {
      best = plans[i].get();
      bestCost = cost;
    }
    if (plans[i]->op->distribution.isSamePartition(distribution)) {
      match = plans[i].get();
      matchCost = cost;
    }
  }
  if (best != match && match) {
    float shuffle = shuffleCost(best->op->columns) * best->cost.fanout;
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
  return join->leftTable == table && !join->leftOptional
      ? std::pair<PlanObjectPtr, float>{join->rightTable, join->lrFanout}
      : join->rightTable == table && !join->rightOptional && !join->rightExists
      ? std::pair<PlanObjectPtr, float>{join->leftTable, join->rlFanout}
      : std::pair<PlanObjectPtr, float>{nullptr, 0};
}

const JoinVector& joinedBy(PlanObjectPtr table) {
  if (table->type == PlanType::kTable) {
    return table->as<BaseTablePtr>()->joinedBy;
  }
  VELOX_DCHECK_EQ(table->type, PlanType::kDerivedTable);
  return table->as<DerivedTablePtr>()->joinedBy;
}

void reducingJoinsRecursive(
    const PlanState& state,
    PlanObjectPtr candidate,
    float fanout,
    float maxFanout,
    std::vector<PlanObjectPtr>& path,
    PlanObjectSet& visited,
    PlanObjectSet& result,
    float& reduction) {
  bool isLeaf = true;
  for (auto join : joinedBy(candidate)) {
    if (join->leftOptional || join->rightOptional) {
      continue;
    }
    JoinCandidate temp;
    temp.join = join;
    JoinSide other = temp.sideOf(candidate, true);
    if (!state.dt->tableSet.contains(other.table)) {
      continue;
    }
    if (other.table->type != PlanType::kTable) {
      continue;
    }
    if (visited.contains(other.table)) {
      continue;
    }
    float joinFanout =
        other.table == join->rightTable ? join->lrFanout : join->rlFanout;
    if (joinFanout > maxFanout) {
      continue;
    }
    visited.add(other.table);
    fanout *= joinFanout;
    if (fanout < 0.9) {
      result.add(other.table);
      for (auto step : path) {
        result.add(step);
        maxFanout = 1;
      }
    }
    path.push_back(other.table);
    isLeaf = false;
    reducingJoinsRecursive(
        state,
        other.table,
        fanout,
        maxFanout,
        path,
        visited,
        result,
        reduction);
    path.pop_back();
  }
  if (fanout < 1 && isLeaf) {
    reduction *= fanout;
  }
}

JoinCandidate reducingJoins(
    const PlanState& state,
    const JoinCandidate& candidate) {
  // For an inner join, see if can bundle reducing joins on the build.
  JoinCandidate reducing;
  reducing.join = candidate.join;
  PlanObjectSet reducingSet;
  if (candidate.join->isInner()) {
    PlanObjectSet visited = state.placed;
    visited.add(candidate.tables[0]);
    reducingSet.add(candidate.tables[0]);
    std::vector<PlanObjectPtr> path{candidate.tables[0]};
    float reduction = 1;
    reducingJoinsRecursive(
        state,
        candidate.tables[0],
        1,
        1.2,
        path,
        visited,
        reducingSet,
        reduction);
    if (reduction < 0.9) {
      // The only table in 'candidate' must be first in the bushy table list.
      reducing.tables = candidate.tables;
      reducingSet.forEach([&](auto object) {
        if (object != reducing.tables[0]) {
          reducing.tables.push_back(object);
        }
      });
      reducing.fanout = candidate.fanout * reduction;
    }
  }
  PlanObjectSet exists;
  float reduction = 1;
  std::vector<PlanObjectPtr> path{candidate.tables[0]};
  // Look for reducing joins that were not added before, also covering already
  // placed tables. This may copy reducing joins from a probe to the
  // corresponding build.
  reducingSet.add(candidate.tables[0]);
  reducingJoinsRecursive(
      state, candidate.tables[0], 1, 10, path, reducingSet, exists, reduction);
  if (reduction < 0.7) {
    reducing.existences.push_back(std::move(exists));
  }
  if (reducing.tables.empty() && reducing.existences.empty()) {
    // No reduction.
    return JoinCandidate{};
  }
  if (reducing.tables.empty()) {
    // No reducing joins but reducing existences from probe side.
    reducing.tables = candidate.tables;
  }
  return reducing;
}

// Calls 'func' with join, joined table and fanout for the joinable tables.
template <typename Func>
void forJoinedTables(DerivedTablePtr dt, const PlanState& state, Func func) {
  std::unordered_set<JoinPtr> visited;
  state.placed.forEach([&](PlanObjectPtr placedTable) {
    for (auto join : joinedBy(placedTable)) {
      if (join->isNonCommutative()) {
        if (!visited.insert(join).second) {
          continue;
        }
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
      } else {
        auto [table, fanout] = otherTable(join, placedTable);
        if (!state.dt->tableSet.contains(table)) {
          continue;
        }
        func(join, table, fanout);
      }
    }
  });
}

JoinSide JoinCandidate::sideOf(PlanObjectPtr side, bool other) const {
  if ((side == join->rightTable && !other) ||
      (side == join->leftTable && other)) {
    return {
        join->rightTable,
        join->rightKeys,
        join->rightOptional,
        join->rightExists,
        join->rightNotExists};
  }
  return {join->leftTable, join->leftKeys, join->leftOptional, false, false};
}

bool NextJoin::isWorse(const NextJoin& other) const {
  float shuffle = plan->distribution.isSamePartition(other.plan->distribution)
      ? 0
      : plan->cost().fanout * shuffleCost(plan->columns);
  return cost.unitCost + cost.setupCost + shuffle >
      other.cost.unitCost + other.cost.setupCost;
}

std::vector<JoinCandidate> Optimization::nextJoins(
    DerivedTablePtr dt,
    PlanState& state) {
  std::vector<JoinCandidate> candidates;
  forJoinedTables(
      dt, state, [&](JoinPtr join, PlanObjectPtr joined, float fanout) {
        if (!state.placed.contains(joined) && state.dt->hasTable(joined)) {
          candidates.emplace_back(join, joined, fanout);
        }
      });

  std::vector<JoinCandidate> bushes;
  // Take the  first hand joined tables and bundle them with reducing joins that
  // can go on the build side.
  for (auto& candidate : candidates) {
    auto bush = reducingJoins(state, candidate);
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
  if (firstTable == other.firstTable && columns == other.columns &&
      tables == other.tables) {
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

RelationOpPtr repartitionForAgg(
    const ExprVector& keyValues,
    const RelationOpPtr& plan,
    PlanState& state) {
  // No shuffle if all grouping keys are in partitioning.
  bool shuffle = false;
  for (auto& key : keyValues) {
    auto nthKey = position(plan->distribution.partition, *key);
    if (nthKey < 0) {
      shuffle = true;
      break;
    }
  }
  if (!shuffle) {
    return plan;
  }

  Distribution distribution(plan->distribution.distributionType, keyValues);
  Declare(
      Repartition, repartition, plan, std::move(distribution), plan->columns);
  state.addCost(*repartition);
  return repartition;
}

void Optimization::addPostprocess(
    DerivedTablePtr dt,
    RelationOpPtr& plan,
    PlanState& state) {
  if (dt->aggregation) {
    plan = repartitionForAgg(dt->aggregation->grouping, plan, state);
    Declare(Aggregation, newGroupBy, *dt->aggregation);
    newGroupBy->input = plan;
    state.addCost(*newGroupBy);
    plan = newGroupBy;
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
    const RelationOpPtr& input) {
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
      if (info.schemaColumn(info.lookupKeys[nthKey]) !=
          info.index->distribution.partition[i]) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

RelationOpPtr repartitionForIndex(
    const IndexInfo& info,
    const ExprVector& lookupValues,
    const RelationOpPtr& plan,
    PlanState& state) {
  if (isIndexColocated(info, lookupValues, plan)) {
    return plan;
  }
  ExprVector keyExprs;
  auto& partition = info.index->distribution.partition;
  for (auto key : partition) {
    // partition is in schema columns, lookupKeys is in BaseTable columns. Use
    // the schema column of lookup key for matching.
    auto nthKey = position(
        info.lookupKeys,
        [](auto c) {
          return c->type == PlanType::kColumn
              ? c->template as<ColumnPtr>()->schemaColumn
              : c;
        },
        *key);
    if (nthKey >= 0) {
      keyExprs.push_back(lookupValues[nthKey]);
    } else {
      return nullptr;
    }
  }

  Distribution distribution(
      info.index->distribution.distributionType, std::move(keyExprs));
  Declare(
      Repartition, repartition, plan, std::move(distribution), plan->columns);
  state.addCost(*repartition);
  return repartition;
}

float fanoutJoinTypeLimit(JoinType joinType, float fanout) {
  switch (joinType) {
    case JoinType::kLeft:
      return std::max<float>(1, fanout);
    case JoinType::kLeftSemiFilter:
      return std::min<float>(1, fanout);
    case JoinType::kAnti:
      return 1 - std::min<float>(1, fanout);
    default:
      return fanout;
  }
}

void Optimization::joinByIndex(
    const RelationOpPtr& plan,
    const JoinCandidate& candidate,
    PlanState& state,
    std::vector<NextJoin>& toTry) {
  if (candidate.tables[0]->type != PlanType::kTable ||
      candidate.tables.size() > 1 || !candidate.existences.empty()) {
    // Index applies to single base tables.
    return;
  }
  auto rightTable = candidate.tables[0]->as<BaseTablePtr>();
  auto left = candidate.sideOf(rightTable, true);
  auto right = candidate.sideOf(rightTable);
  auto& keys = right.keys;
  auto keyColumns = leadingColumns(keys);
  if (keyColumns.empty()) {
    return;
  }
  for (auto& index : rightTable->schemaTable->indices) {
    auto info = rightTable->schemaTable->indexInfo(index, keyColumns);
    if (info.lookupKeys.empty()) {
      continue;
    }
    StateSaver save(state);
    auto newPartition = repartitionForIndex(info, left.keys, plan, state);
    if (!newPartition) {
      continue;
    }
    state.placed.add(candidate.tables[0]);
    auto joinType = right.leftJoinType();
    auto fanout = fanoutJoinTypeLimit(
        joinType, info.scanCardinality * rightTable->filterSelectivity);
    Declare(
        TableScan,
        scan,
        newPartition,
        newPartition->distribution,
        rightTable,
        info.index,
        fanout);
    scan->joinType = joinType;
    scan->keys = left.keys;
    // The number of keys is  the prefix that matches index order.
    scan->keys.resize(info.lookupKeys.size());
    state.columns.unionSet(scan->availableColumns());
    PlanObjectSet c = state.downstreamColumns();
    c.intersect(state.columns);

    c.forEach(
        [&](auto o) { scan->columns.push_back(reinterpret_cast<Column*>(o)); });
    for (auto& filter : scan->baseTable->filter) {
      scan->extractedColumns.unionSet(filter->columns);
    }
    auto join = candidate.join;
    if (join->filter) {
      scan->joinFilter = join->filter;
      scan->extractedColumns.unionSet(join->filter->columns);
    }
    state.addCost(*scan);
    if (!state.addNextJoin(&candidate, scan, {}, toTry)) {
      trace(kExceededBest, state.dt->id, state.cost, *scan);
    }
  }
}

// Returns the positions in 'keys' for the expressions that determine the
// partition. empty if the partition is not decided by 'keys'
std::vector<int32_t> joinKeyPartition(
    const RelationOpPtr& op,
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
  return build->cost.fanout < 100000;
}

void Optimization::joinByHash(
    const RelationOpPtr& plan,
    const JoinCandidate& candidate,
    PlanState& state,
    std::vector<NextJoin>& toTry) {
  auto build = candidate.sideOf(candidate.tables[0]);
  auto probe = candidate.sideOf(candidate.tables[0], true);
  auto partKeys = joinKeyPartition(plan, probe.keys);
  Distribution distribution;
  distribution.distributionType = plan->distribution.distributionType;
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
  buildColumns.unionColumns(build.keys);
  auto key = MemoKey{
      candidate.tables[0], buildColumns, buildTables, candidate.existences};
  PlanObjectSet empty;
  bool needsShuffle = false;
  auto buildPlan = makePlan(key, distribution, empty, state, needsShuffle);
  PlanState buildState(buildPlan);
  bool partitionByProbe = !partKeys.empty();
  RelationOpPtr buildInput = buildPlan->op;
  RelationOpPtr probeInput = plan;
  if (partitionByProbe) {
    if (needsShuffle) {
      Distribution dist(
          plan->distribution.distributionType, distribution.partition);
      Declare(Repartition, shuffleTemp, buildInput, dist, buildInput->columns);
      buildState.addCost(*shuffleTemp);
      buildInput = shuffleTemp;
    }
  } else if (isBroadcastable(buildPlan, state)) {
    Distribution dist;
    dist.isBroadcast = true;
    Declare(
        Repartition,
        broadcast,
        buildInput,
        std::move(dist),
        buildInput->columns);
    buildState.addCost(*broadcast);
    buildInput = broadcast;
  } else {
    // The probe gets shuffled to align with build. If build is not partitioned
    // on its keys, shuffle the build too.
    auto buildPart = joinKeyPartition(buildInput, build.keys);
    if (buildPart.empty()) {
      // The build is not aligned on join keys.
      Distribution buildDist(plan->distribution.distributionType, build.keys);
      Declare(
          Repartition,
          buildShuffle,
          buildInput,
          buildDist,
          buildInput->columns);
      buildState.addCost(*buildShuffle);
      buildInput = buildShuffle;
    }

    ExprVector distCols;
    for (auto i = 0; i < probeInput->distribution.partition.size(); ++i) {
      auto key = buildInput->distribution.partition[i];
      auto nthKey = position(build.keys, *key);
      distCols.push_back(probe.keys[nthKey]);
    }

    Distribution probeDist(
        probeInput->distribution.distributionType, std::move(distCols));
    Declare(
        Repartition, probeShuffle, plan, std::move(probeDist), plan->columns);
    state.addCost(*probeShuffle);
    probeInput = probeShuffle;
  }
  Declare(
      HashBuild, buildOp, buildInput, ++buildCounter_, build.keys, buildPlan);
  buildState.addCost(*buildOp);

  ColumnVector columns;
  downstream.forEach([&](auto object) {
    columns.push_back(reinterpret_cast<ColumnPtr>(object));
  });

  auto joinType = velox::core::JoinType::kInner;
  auto fanout = fanoutJoinTypeLimit(joinType, candidate.fanout);
  Declare(
      JoinOp,
      joinOp,
      JoinMethod::kHash,
      joinType,
      probeInput,
      buildOp,
      probe.keys,
      build.keys,
      candidate.join->filter,
      fanout,
      std::move(columns));
  state.addCost(*joinOp);
  state.cost.setupCost += buildState.cost.unitCost;
  if (!state.addNextJoin(&candidate, joinOp, {buildOp}, toTry)) {
    trace(kExceededBest, state.dt->id, state.cost, *joinOp);
  }
}

void Optimization::addJoin(
    DerivedTablePtr dt,
    const JoinCandidate& candidate,
    const RelationOpPtr& plan,
    PlanState& state,
    std::vector<NextJoin>& result) {
  std::vector<NextJoin> toTry;
  joinByIndex(plan, candidate, state, toTry);
  joinByHash(plan, candidate, state, toTry);
  // If one is much better do not try the other.
  if (toTry.size() == 2 && candidate.tables.size() == 1) {
    if (toTry[0].isWorse(toTry[1])) {
      toTry.erase(toTry.begin());
    } else if (toTry[1].isWorse(toTry[0])) {
      toTry.erase(toTry.begin() + 1);
    }
  }
  result.insert(result.end(), toTry.begin(), toTry.end());
}

// Sets 'columns' and 'schemaColumns' to the columns in 'downstream' that exist
// in 'index' of 'table'.
void indexColumns(
    const PlanObjectSet& downstream,
    IndexPtr index,
    BaseTablePtr table,
    ColumnVector& columns,
    ColumnVector& schemaColumns) {
  for (auto& indexColumn : index->columns) {
    auto nthColumn = position(table->schemaColumns, *indexColumn);
    if (nthColumn >= 0 && downstream.contains(table->columns[nthColumn])) {
      columns.push_back(table->columns[nthColumn]);
      schemaColumns.push_back(table->schemaColumns[nthColumn]);
    }
  }
}

void Optimization::tryNextJoins(
    PlanState& state,
    const std::vector<NextJoin>& nextJoins) {
  for (auto& next : nextJoins) {
    StateSaver save(state);
    state.placed = next.placed;
    state.cost = next.cost;
    state.addBuilds(next.newBuilds);
    makeJoins(next.plan, state);
  }
}

void Optimization::makeJoins(RelationOpPtr plan, PlanState& state) {
  auto& dt = state.dt;
  if (!plan) {
    std::vector<PlanObjectPtr> firstTables;
    dt->startTables.forEach([&](auto table) {firstTables.push_back(table); });
    std::vector<float> scores(firstTables.size());
    for (auto i = 0; i < firstTables.size(); ++i) {
      auto table = firstTables[i];
      scores[i] = startingScore(table, dt);
    }
    std::vector<int32_t> ids(firstTables.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int32_t left, int32_t right) {
      return scores[left] > scores[right];
    });
    for (auto i : ids) {
      auto from = firstTables[i];
      if (from->type == PlanType::kTable) {
        auto table = from->as<BaseTablePtr>();
        auto indices = chooseLeafIndex(table->as<BaseTablePtr>(), dt);
        // Make plan starting with each relevant index of the table.
        auto downstream = state.downstreamColumns();
        for (auto index : indices) {
          StateSaver save(state);
          state.placed.add(table);
          Declare(
              TableScan,
              scan,
              nullptr,
              Distribution(),
              table,
              index,
              index->distribution.cardinality * table->filterSelectivity);
          ColumnVector columns;
          ColumnVector schemaColumns;
          indexColumns(downstream, index, table, columns, schemaColumns);
          scan->setRelation(columns, schemaColumns);

          state.addCost(*scan);
          makeJoins(scan, state);
        }
      } else {
        // Start with a derived table.
        VELOX_NYI();
      }
    }
  } else {
    if (state.isOverBest()) {
      trace(kExceededBest, dt->id, state.cost, *plan);
      return;
    }
    auto candidates = nextJoins(dt, state);
    if (candidates.empty()) {
      addPostprocess(dt, plan, state);
      auto kept = state.plans.addPlan(plan, state);
      if (kept) {
        trace(kRetained, dt->id, state.cost, *kept->op);
      }
    }
    std::vector<NextJoin> nextJoins;
    for (auto& candidate : candidates) {
      addJoin(dt, candidate, plan, state, nextJoins);
    }
    tryNextJoins(state, nextJoins);
  }
}

PlanPtr Optimization::makePlan(
    const MemoKey& key,
    const Distribution& distribution,
    const PlanObjectSet& boundColumns,
    PlanState& state,
    bool& needsShuffle) {
  auto it = memo_.find(key);
  PlanSet* plans;
  if (it == memo_.end()) {
    DerivedTable dt;
    dt.import(*state.dt, key.firstTable, key.tables, key.existences);
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
