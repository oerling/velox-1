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

#include "velox/experimental/query/PlanToGraph.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

using namespace facebook::velox;

using velox::connector::hive::HiveColumnHandle;
using velox::connector::hive::HiveTableHandle;

Optimization::Optimization(const core::PlanNode& plan, const Schema& schema)
    : schema_(schema), inputPlan_(plan) {
  root_ = makeQueryGraph();
  root_->expandJoins();
}

std::shared_ptr<const core::PlanNode> Optimization::bestPlan() {
  return nullptr;
}

std::unordered_map<std::string, float>& baseSelectivities() {
  static std::unordered_map<std::string, float> map;
  return map;
}

FunctionSet functionBits(Name name) {
  return FunctionSet(0);
}

DerivedTablePtr Optimization::makeQueryGraph() {
  Define(DerivedTable, root);
  root_ = root;
  currentSelect_ = root_;
  makeQueryGraph(inputPlan_);
  return root_;
}

float subfieldSelectivity(const HiveTableHandle& handle) {
  if (handle.subfieldFilters().empty()) {
    return 1;
  }
  auto string = handle.toString();
  auto it = baseSelectivities().find(string);
  if (it != baseSelectivities().end()) {
    return it->second;
  }
  return 0.1;
}

void BaseTable::setRelation(
    const Relation& relation,
    const ColumnVector& columns,
    const ColumnVector& schemaColumns) {
  // if all partitioning columns are projected, the output is partitioned.
  if (isSubset(
          toRange(schemaColumns),
          toRangeCast<ColumnPtr>(relation.distribution.partition))) {
    distribution.partition = relation.distribution.partition;
    distribution.distributionType = relation.distribution.distributionType;
  }
  auto numPrefix = prefixSize(
      toRangeCast<ColumnPtr>(relation.distribution.order),
      toRange(schemaColumns));
  if (numPrefix > 0) {
    distribution.order = relation.distribution.order;
    distribution.order.resize(numPrefix);
    distribution.orderType = relation.distribution.orderType;
    distribution.orderType.resize(numPrefix);
    if (relation.distribution.numKeysUnique <= numPrefix) {
      distribution.numKeysUnique = relation.distribution.numKeysUnique;
    }
  }
}

const std::string* FOLLY_NULLABLE columnName(const core::TypedExprPtr& expr) {
  if (auto column =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (dynamic_cast<const core::InputTypedExpr*>(column->inputs()[0].get())) {
      return &column->name();
    }
  }
  return nullptr;
}

ExprPtr Optimization::translateExpr(const core::TypedExprPtr& expr) {
  if (auto name = columnName(expr)) {
    return translateColumn(*name);
  }
  if (auto constant =
          dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    Define(
        Literal, literal, Value(constant->type().get(), 1), constant->value());
    return literal;
  }
  auto it = exprDedup_.find(expr.get());
  if (it != exprDedup_.end()) {
    return it->second;
  }
  ExprVector args{expr->inputs().size(), stl<ExprPtr>()};
  PlanObjectSet columns;
  FunctionSet funcs;
  auto& inputs = expr->inputs();
  float cardinality = 1;
  for (auto i = 0; i < inputs.size(); ++i) {
    args[i] = translateExpr(inputs[i]);
    cardinality = std::max(cardinality, args[i]->value.cardinality);
    if (args[i]->type == PlanType::kCall) {
      funcs = funcs | args[i]->as<CallPtr>()->functions;
    }
  }
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    auto name = toName(call->name());
    funcs = funcs | functionBits(name);

    Define(
        Call,
        callExpr,
        name,
        Value(call->type().get(), cardinality),
        args,
        funcs);
    exprDedup_[expr.get()] = callExpr;
    return callExpr;
  }
  VELOX_NYI();
  return nullptr;
}

ExprPtr Optimization::translateColumn(const std::string& name) {
  auto column = renames_.find(name);
  if (column != renames_.end()) {
    return column->second;
  }
  VELOX_FAIL("could not resolve name {}", name);
}

ExprVector Optimization::translateColumns(
    const std::vector<core::FieldAccessTypedExprPtr>& source) {
  ExprVector result{source.size(), stl<ExprPtr>()};
  for (auto i = 0; i < source.size(); ++i) {
    result[i] = translateColumn(source[i]->name());
  }
  return result;
}

GroupByPtr Optimization::translateGroupBy(
    const core::AggregationNode& aggregation) {
  return nullptr;
}

OrderByPtr Optimization::translateOrderBy(const core::OrderByNode& order) {
  return nullptr;
}

PlanObjectPtr singleTable(PlanObjectPtr object) {
  if (isExprType(object->type)) {
    return object->as<ExprPtr>()->singleTable();
  }
  return nullptr;
}

void makeJoin(DerivedTablePtr dt, ExprPtr left, ExprPtr right) {
  auto leftTable = singleTable(left);
  auto rightTable = singleTable(right);
  for (auto& join : dt->joins) {
    if (join->leftTable == leftTable && join->rightTable == rightTable) {
      join->leftKeys.push_back(left);
      join->rightKeys.push_back(right);
      join->guessFanout();
      return;
    } else if (join->rightTable == leftTable && join->leftTable == rightTable) {
      join->leftKeys.push_back(right);
      join->rightKeys.push_back(left);
      join->guessFanout();
      return;
    }
  }
  Define(Join, join);
  join->leftKeys.push_back(left);
  join->rightKeys.push_back(right);
  join->leftTable = leftTable;
  join->rightTable = rightTable;
  join->guessFanout();
  dt->joins.push_back(join);
}

void Optimization::translateJoin(const core::AbstractJoinNode& join) {
  makeQueryGraph(*join.sources()[0]);
  auto leftKeys = translateColumns(join.leftKeys());
  makeQueryGraph(*join.sources()[1]);
  auto rightKeys = translateColumns(join.rightKeys());
  if (join.isInnerJoin()) {
    // Every column to column equality adds to an equivalence class and is an
    // independent bidirectional join edge.
    for (auto i = 0; i < leftKeys.size(); ++i) {
      auto l = leftKeys[i];
      auto r = rightKeys[i];
      if (l->type == PlanType::kColumn && r->type == PlanType::kColumn) {
        l->as<ColumnPtr>()->equals(r->as<ColumnPtr>());
        makeJoin(currentSelect_, l, r);
      } else {
        VELOX_NYI("Only column to column inner joins");
      }
    }
  } else {
    VELOX_NYI("Only inner joins");
  }
}

PlanObjectPtr Optimization::makeQueryGraph(const core::PlanNode& node) {
  auto name = node.name();
  if (name == "TableScan") {
    auto tableScan = reinterpret_cast<const core::TableScanNode*>(&node);
    auto tableHandle =
        dynamic_cast<const HiveTableHandle*>(tableScan->tableHandle().get());
    VELOX_CHECK(tableHandle);
    auto assignments = tableScan->assignments();
    auto schemaTable = schema_.findTable(tableHandle->tableName());
    float selection = subfieldSelectivity(*tableHandle);
    auto cname = fmt::format("t{}", ++nameCounter_);

    Define(BaseTable, baseTable);
    baseTable->cname = toName(cname);
    baseTable->schemaTable = schemaTable;
    ColumnVector columns{stl<ColumnPtr>()};
    ColumnVector schemaColumns{stl<ColumnPtr>()};
    for (auto& pair : assignments) {
      auto handle =
          reinterpret_cast<const HiveColumnHandle*>(pair.second.get());
      auto schemaColumn = schemaTable->findColumn(handle->name());
      schemaColumns.push_back(schemaColumn);
      auto value = schemaColumn->value;
      Define(Column, column, toName(handle->name()), baseTable, value);
      columns.push_back(column);
      renames_[pair.first] = column;
    }
    baseTable->setRelation(*schemaTable->indices[0], columns, schemaColumns);
    baseTable->distribution.cardinality =
        schemaTable->indices[0]->distribution.cardinality * selection;
    currentSelect_->tables.push_back(baseTable);
    return baseTable;
  }
  if (name == "Project") {
    makeQueryGraph(*node.sources()[0]);
    auto project = reinterpret_cast<const core::ProjectNode*>(&node);
    auto names = project->names();
    auto exprs = project->projections();
    for (auto i = 0; i < names.size(); ++i) {
      Expr* expr = translateExpr(exprs[i]);
      renames_[names[i]] = expr;
    }
    return currentSelect_;
  }
  if (name == "Filter") {
    makeQueryGraph(*node.sources()[0]);
    auto filter = reinterpret_cast<const core::FilterNode*>(&node);
    Expr* expr = translateExpr(filter->filter());
    currentSelect_->conjuncts.push_back(expr);
    return currentSelect_;
  }
  if (name == "HashJoin" || name == "MergeJoin") {
    translateJoin(*reinterpret_cast<const core::AbstractJoinNode*>(&node));
    return currentSelect_;
  }
  if (name == "LocalPartition") {
    makeQueryGraph(*node.sources()[0]);
    return currentSelect_;
  }
  if (name == "Aggregation") {
    makeQueryGraph(*node.sources()[0]);
    currentSelect_->groupBy = translateGroupBy(
        *reinterpret_cast<const core::AggregationNode*>(&node));
    return currentSelect_;
  }
  if (name == "OrderBy") {
    makeQueryGraph(*node.sources()[0]);
    currentSelect_->orderBy =
        translateOrderBy(*reinterpret_cast<const core::OrderByNode*>(&node));
    return currentSelect_;
  }
  if (name == "Limit") {
    makeQueryGraph(*node.sources()[0]);
    auto limit = reinterpret_cast<const core::LimitNode*>(&node);
    currentSelect_->limit = limit->count();
    currentSelect_->offset = limit->offset();
  }
  return currentSelect_;
}

Plan::Plan(RelationOpPtr _op, const PlanState& state)
    : op(_op),
      unitCost(state.cost),
      setupCost(state.setupCost),
      fanout(state.fanout),
      tables(state.tables),
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

void PlanSet::addPlan(RelationOpPtr plan, PlanState& state) {
  if (!best || best->isStateBetter(state)) {
    Define(Plan, newPlan, plan, state);
    best = newPlan;
  }
}

float startingScore(PlanObjectPtr table, DerivedTablePtr dt) {
  if (table->type == PlanType::kTable) {
    return table->as<BaseTablePtr>()->distribution.cardinality;
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
  state.tables.forEach([&](PlanObjectPtr placedTable) {
    for (auto join : joinedBy(placedTable)) {
      auto [table, fanout] = otherTable(join, placedTable);
      if (table) {
        func(join, table, fanout);
      } else {
        if (!join->leftTable) {
          bool usable = true;
          for (auto key : join->leftKeys) {
            if (!state.tables.isSubset(key->allTables())) {
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
  PlanObjectSet visited = state.tables;
  // Look for reducing joins against tables not in the plan.
#if 0
  reducingJoinsRecursive(dt, start, selected, visited);
  visited = selected;
  forDirectJoins(table, [&](JoinPtr join, PlanObjectPtr other) {
    
  });
#endif
  return JoinCandidate();
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

void Optimization::addPostprocess(
    DerivedTablePtr dt,
    RelationOpPtr& plan,
    PlanState& state) {
  if (dt->groupBy) {
    Define(GroupBy, newGroupBy, *dt->groupBy);
  }
}

std::vector<IndexPtr> chooseLeafIndex(BaseTablePtr table, DerivedTablePtr dt) {
  return {table->schemaTable->indices[0]};
}

void Optimization::makeJoins(
    DerivedTablePtr dt,
    RelationOpPtr plan,
    PlanState& state) {
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
        state.tables.add(table);
        for (auto index : indices) {
          Define(TableScan, scan);
          scan->relType = RelType::kTableScan;
          scan->baseTable = table;
          scan->index = index;
          scan->distribution = index->distribution;
          scan->columns = table->columns;
          makeJoins(dt, scan, state);
        }
      } else {
        // Start with a derived table.
        VELOX_NYI();
      }
      state.tables.erase(from);
    }
  } else {
    auto candidates = nextJoins(dt, state);
    if (candidates.empty()) {
      addPostprocess(dt, plan, state);
      state.plans.addPlan(plan, state);
    }
    for (auto& candidate : candidates) {
    }
  }
}

#if 0
void Optimization::makePlans(
    derivedTablePtr table,
    RelationOpPtr* input,
    const PlanObjectSet& boundColumns) {}
#endif
} // namespace facebook::verax
