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

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

using namespace facebook::velox;

using velox::connector::hive::HiveColumnHandle;
using velox::connector::hive::HiveTableHandle;

void Optimization::setDerivedTableOutput(
    DerivedTablePtr dt,
    const velox::core::PlanNode& planNode) {
  auto& outputType = planNode.outputType();
  for (auto i = 0; i < outputType->size(); ++i) {
    auto fieldType = outputType->childAt(i);
    auto fieldName = outputType->nameOf(i);
    auto expr = translateColumn(fieldName);
    Value value(fieldType.get(), 0);
    Declare(Column, column, toName(fieldName), dt, value);
    dt->columns.push_back(column);
    dt->exprs.push_back(expr);
    renames_[fieldName] = column;
  }
}

DerivedTablePtr Optimization::makeQueryGraph() {
  Declare(DerivedTable, root);
  root_ = root;
  currentSelect_ = root_;
  root->cname = toName(fmt::format("dt{}", ++nameCounter_));
  ;
  makeQueryGraph(inputPlan_, kAllAllowedInDt);
  return root_;
}

float subfieldSelectivity(const HiveTableHandle& handle) {
  if (handle.subfieldFilters().empty() && !handle.remainingFilter()) {
    return 1;
  }
  auto string = handle.toString();
  auto it = baseSelectivities().find(string);
  if (it != baseSelectivities().end()) {
    return it->second;
  }
  return 0.1;
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

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    return call->name() == name;
  }
  return false;
}

void Optimization::translateConjuncts(
    const core::TypedExprPtr& input,
    ExprVector& flat) {
  if (!input) {
    return;
  }
  if (isCall(input, "and")) {
    for (auto& child : input->inputs()) {
      translateConjuncts(child, flat);
    }
  } else {
    flat.push_back(translateExpr(input));
  }
}

ExprPtr Optimization::translateExpr(const core::TypedExprPtr& expr) {
  if (auto name = columnName(expr)) {
    return translateColumn(*name);
  }
  if (auto constant =
          dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    Declare(
        Literal, literal, Value(constant->type().get(), 1), &constant->value());
    return literal;
  }
  auto it = exprDedup_.find(expr.get());
  if (it != exprDedup_.end()) {
    return it->second;
  }
  ExprVector args{expr->inputs().size()};
  PlanObjectSet columns;
  FunctionSet funcs;
  auto& inputs = expr->inputs();
  float cardinality = 1;
  for (auto i = 0; i < inputs.size(); ++i) {
    args[i] = translateExpr(inputs[i]);
    cardinality = std::max(cardinality, args[i]->value().cardinality);
    if (args[i]->type() == PlanType::kCall) {
      funcs = funcs | args[i]->as<Call>()->functions();
    }
  }
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    auto name = toName(call->name());
    funcs = funcs | functionBits(name);

    Declare(
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
  ExprVector result{source.size()};
  for (auto i = 0; i < source.size(); ++i) {
    result[i] = translateColumn(source[i]->name()); // NOLINT
  }
  return result;
}

AggregationPtr FOLLY_NULLABLE
Optimization::translateAggregation(const core::AggregationNode& source) {
  using velox::core::AggregationNode;

  if (source.step() == AggregationNode::Step::kPartial ||
      source.step() == AggregationNode::Step::kSingle) {
    Declare(
        Aggregation,
        aggregation,
        nullptr,
        translateColumns(source.groupingKeys()));
    for (auto i = 0; i < source.aggregateNames().size(); ++i) {
      auto rawFunc = translateExpr(source.aggregates()[i])->as<Call>();
      ExprPtr condition = nullptr;
      if (source.aggregateMasks()[i]) {
        condition = translateExpr(source.aggregateMasks()[i]);
      }
      Declare(
          Aggregate,
          agg,
          rawFunc->name(),
          rawFunc->value(),
          rawFunc->args(),
          rawFunc->functions(),
          false,
          condition,
          false);
      auto dedupped = queryCtx()->dedup(agg);
      aggregation->aggregates.push_back(dedupped->as<Aggregate>());
      auto name = toName(source.aggregateNames()[i]);
      renames_[name] = dedupped->as<Expr>();
    }
    return aggregation;
  }
  return nullptr;
}

OrderByPtr FOLLY_NULLABLE
Optimization::translateOrderBy(const core::OrderByNode& /*order*/) {
  return nullptr;
}

void Optimization::translateJoin(const core::AbstractJoinNode& join) {
  bool isInner = join.isInnerJoin();
  makeQueryGraph(*join.sources()[0], allow(PlanType::kJoin));
  auto leftKeys = translateColumns(join.leftKeys());
  // For an inner join a join tree on the right can be flattened, for all other
  // kinds it must be kept together in its own dt.
  makeQueryGraph(*join.sources()[1], isInner ? allow(PlanType::kJoin) : 0);
  auto rightKeys = translateColumns(join.rightKeys());
  ExprVector conjuncts;
  translateConjuncts(join.filter(), conjuncts);
  if (isInner) {
    // Every column to column equality adds to an equivalence class and is an
    // independent bidirectional join edge.
    for (auto i = 0; i < leftKeys.size(); ++i) {
      auto l = leftKeys[i];
      auto r = rightKeys.at(i);
      if (l->type() == PlanType::kColumn && r->type() == PlanType::kColumn) {
        l->as<Column>()->equals(r->as<Column>());
        currentSelect_->addJoinEquality(l, r, {}, false, false, false, false);
      } else {
        VELOX_NYI("Only column to column inner joins");
      }
    }
    currentSelect_->conjuncts.insert(
        currentSelect_->conjuncts.end(), conjuncts.begin(), conjuncts.end());
  } else {
    VELOX_NYI("Only inner joins");
  }
}

bool isJoin(const core::PlanNode& node) {
  auto name = node.name();
  if (name == "HashJoin" || name == "MergeJoin") {
    return true;
  }
  if (name == "Project" || name == "Filter") {
    return isJoin(*node.sources()[0]);
  }
  return false;
}

bool isDirectOver(const core::PlanNode& node, const std::string& name) {
  auto source = node.sources()[0];
  if (source && source->name() == name) {
    return true;
  }
  return false;
}

PlanObjectPtr Optimization::wrapInDt(const core::PlanNode& node) {
  DerivedTablePtr previousDt = currentSelect_;
  Declare(DerivedTable, newDt);
  auto cname = toName(fmt::format("dt{}", ++nameCounter_));
  newDt->cname = cname;
  currentSelect_ = newDt;
  makeQueryGraph(node, kAllAllowedInDt);

  currentSelect_ = previousDt;
  velox::RowTypePtr type =
      node.name() == "Aggregation" ? aggFinalType_ : node.outputType();
  for (auto i = 0; i < type->size(); ++i) {
    ExprPtr inner = translateColumn(type->nameOf(i));
    newDt->exprs.push_back(inner);
    Declare(Column, outer, toName(type->nameOf(i)), newDt, inner->value());
    newDt->columns.push_back(outer);
    renames_[type->nameOf(i)] = outer;
  }
  currentSelect_->tables.push_back(newDt);
  currentSelect_->tableSet.add(newDt);
  MemoKey key;
  key.firstTable = newDt;
  key.tables.add(newDt);
  newDt->addImpliedJoins();
  newDt->linkTablesToJoins();
  newDt->setStartTables();
  PlanState state(*this, newDt);
  for (auto expr : newDt->exprs) {
    state.targetColumns.unionColumns(expr);
  }

  makeJoins(nullptr, state);
  Distribution emptyDistribution;
  bool needsShuffle;
  auto plan = state.plans.best(emptyDistribution, needsShuffle)->op;
  auto& distribution = plan->distribution();
  ExprVector partition = distribution.partition;
  ExprVector order = distribution.order;
  auto orderType = distribution.orderType;
  replace(partition, newDt->exprs, newDt->columns.data());
  replace(order, newDt->exprs, newDt->columns.data());
  Declare(
      Distribution,
      dtDist,
      distribution.distributionType,
      distribution.cardinality,
      partition,
      order,
      orderType);
  newDt->distribution = dtDist;
  memo_[key] = std::move(state.plans);

  return newDt;
}

PlanObjectPtr Optimization::makeQueryGraph(
    const core::PlanNode& node,
    uint64_t allowedInDt) {
  auto name = node.name();
  if (isJoin(node) && !contains(allowedInDt, PlanType::kJoin)) {
    return wrapInDt(node);
  }
  if (name == "TableScan") {
    auto tableScan = reinterpret_cast<const core::TableScanNode*>(&node);
    auto tableHandle =
        dynamic_cast<const HiveTableHandle*>(tableScan->tableHandle().get());
    VELOX_CHECK(tableHandle);
    auto assignments = tableScan->assignments();
    auto schemaTable = schema_.findTable(tableHandle->tableName());
    float selection = subfieldSelectivity(*tableHandle);
    auto cname = fmt::format("t{}", ++nameCounter_);

    Declare(BaseTable, baseTable);
    baseTable->cname = toName(cname);
    baseTable->schemaTable = schemaTable;
    ColumnVector columns;
    ColumnVector schemaColumns;
    for (auto& pair : assignments) {
      auto handle =
          reinterpret_cast<const HiveColumnHandle*>(pair.second.get());
      auto schemaColumn = schemaTable->findColumn(handle->name());
      schemaColumns.push_back(schemaColumn);
      auto value = schemaColumn->value();
      Declare(Column, column, toName(handle->name()), baseTable, value);
      columns.push_back(column);
      renames_[pair.first] = column;
    }
    baseTable->columns = columns;
    baseTable->filterSelectivity = selection;
    currentSelect_->tables.push_back(baseTable);
    currentSelect_->tableSet.add(baseTable);
    return baseTable;
  }
  if (name == "Project") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    auto project = reinterpret_cast<const core::ProjectNode*>(&node);
    auto names = project->names();
    auto exprs = project->projections();
    for (auto i = 0; i < names.size(); ++i) {
      auto expr = translateExpr(exprs.at(i));
      renames_[names[i]] = expr;
    }
    return currentSelect_;
  }
  if (name == "Filter") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    auto filter = reinterpret_cast<const core::FilterNode*>(&node);
    ExprVector flat;
    translateConjuncts(filter->filter(), flat);
    if (isDirectOver(node, "Aggregation")) {
      VELOX_CHECK(
          currentSelect_->having.empty(),
          "Must have al;all HAVING in one filter");
      currentSelect_->having = flat;
    } else {
      currentSelect_->conjuncts.insert(
          currentSelect_->conjuncts.end(), flat.begin(), flat.end());
    }
    return currentSelect_;
  }
  if (name == "HashJoin" || name == "MergeJoin") {
    if (!contains(allowedInDt, PlanType::kJoin)) {
      return wrapInDt(node);
    }
    translateJoin(*reinterpret_cast<const core::AbstractJoinNode*>(&node));
    return currentSelect_;
  }
  if (name == "LocalPartition") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    return currentSelect_;
  }
  if (name == "Aggregation") {
    using AggregationNode = velox::core::AggregationNode;
    auto& aggNode = *reinterpret_cast<const core::AggregationNode*>(&node);
    if (aggNode.step() == AggregationNode::Step::kPartial ||
        aggNode.step() == AggregationNode::Step::kSingle) {
      if (!contains(allowedInDt, PlanType::kAggregation)) {
        return wrapInDt(node);
      }
      if (aggNode.step() == AggregationNode::Step::kSingle) {
        aggFinalType_ = aggNode.outputType();
      }
      makeQueryGraph(
          *node.sources()[0], makeDtIf(allowedInDt, PlanType::kAggregation));
      auto agg = translateAggregation(aggNode);
      if (agg) {
        currentSelect_->aggregation = agg;
      }
    } else {
      if (aggNode.step() == AggregationNode::Step::kFinal) {
        aggFinalType_ = aggNode.outputType();
      }
      makeQueryGraph(*aggNode.sources()[0], allowedInDt);
    }
    return currentSelect_;
  }
  if (name == "OrderBy") {
    if (!contains(allowedInDt, PlanType::kOrderBy)) {
      return wrapInDt(node);
    }
    makeQueryGraph(
        *node.sources()[0], makeDtIf(allowedInDt, PlanType::kOrderBy));
    currentSelect_->orderBy =
        translateOrderBy(*reinterpret_cast<const core::OrderByNode*>(&node));
    return currentSelect_;
  }
  if (name == "Limit") {
    if (!contains(allowedInDt, PlanType::kLimit)) {
      return wrapInDt(node);
    }
    makeQueryGraph(*node.sources()[0], makeDtIf(allowedInDt, PlanType::kLimit));
    auto limit = reinterpret_cast<const core::LimitNode*>(&node);
    currentSelect_->limit = limit->count();
    currentSelect_->offset = limit->offset();
  }
  return currentSelect_;
}

} // namespace facebook::verax
