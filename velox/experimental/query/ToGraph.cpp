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
    Declare(
        Literal, literal, Value(constant->type().get(), 1), constant->value());
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
    cardinality = std::max(cardinality, args[i]->value.cardinality);
    if (args[i]->type == PlanType::kCall) {
      funcs = funcs | args[i]->as<CallPtr>()->functions;
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
    result[i] = translateColumn(source[i]->name());
  }
  return result;
}

AggregationPtr Optimization::translateGroupBy(
    const core::AggregationNode& source) {
  using velox::core::AggregationNode;

  if (source.step() == AggregationNode::Step::kPartial ||
      source.step() == AggregationNode::Step::kSingle) {
    Declare(
        Aggregation,
        aggregation,
        nullptr,
        translateColumns(source.groupingKeys()));
    for (auto i = 0; i < source.aggregateNames().size(); ++i) {
      auto rawFunc = translateExpr(source.aggregates()[i])->as<CallPtr>();
      ExprPtr condition = nullptr;
      if (source.aggregateMasks()[i]) {
        condition = translateExpr(source.aggregateMasks()[i]);
      }
      Declare(
          Aggregate,
          agg,
          rawFunc->func,
          rawFunc->value,
          rawFunc->args,
          rawFunc->functions,
          false,
          condition,
          false);
      auto dedupped = queryCtx()->dedup(agg);
      aggregation->aggregates.push_back(dedupped->as<AggregatePtr>());
      auto name = toName(source.aggregateNames()[i]);
      renames_[name] = dedupped->as<ExprPtr>();
    }
    return aggregation;
  }
  return nullptr;
}

OrderByPtr Optimization::translateOrderBy(const core::OrderByNode& order) {
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
  Declare(Join, join);
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
      auto value = schemaColumn->value;
      Declare(Column, column, toName(handle->name()), baseTable, value);
      columns.push_back(column);
      renames_[pair.first] = column;
    }
    baseTable->columns = columns;
    baseTable->schemaColumns = schemaColumns;
    baseTable->filterSelectivity = selection;
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
    auto agg = translateGroupBy(
        *reinterpret_cast<const core::AggregationNode*>(&node));
    if (agg) {
      currentSelect_->aggregation = agg;
    }
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

} // namespace facebook::verax
