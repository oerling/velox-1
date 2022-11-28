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

namespace facebook::velox::query {

using velox::connector::hive::HiveColumnHandle;
using velox::connector::hive::HiveTableHandle;

Optimization::Optimization(const core::PlanNode& plan, const Schema& schema)
    : schema_(schema), inputPlan_(plan) {
  root_ = makeQueryGraph();
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
          toRange(schemaColumns), toRange(relation.distribution.partition))) {
    distribution.partition = relation.distribution.partition;
    distribution.distributionType = relation.distribution.distributionType;
  }
  auto numPrefix =
      prefixSize(toRange(relation.distribution.order), toRange(schemaColumns));
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
  if (it == exprDedup_.end()) {
    return it->second;
  }
  ExprVector args{expr->inputs().size(), stl<ExprPtr>()};
  PlanObjectSet columns;
  FunctionSet funcs;
  auto& inputs = expr->inputs();
  int64_t cardinality = 1;
  for (auto i = 0; i < inputs.size(); ++i) {
    args[i] = translateExpr(inputs[i]);
    columns.unionColumns(args[i]);
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
        columns,
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

  GroupByPtr Optimization::translateGroupBy(const core::AggregationNode& aggregation) {
  return nullptr;
}

  OrderByPtr Optimization::translateOrderBy(const core::OrderByNode& order) {
  return nullptr;
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
        DefineDefault(Join, edge);
        edge->leftKeys.push_back(l);
        edge->rightKeys.push_back(r);
        currentSelect_->joins.push_back(edge);
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
    ColumnVector columns{stl<ColumnPtr>()};
    ColumnVector schemaColumns{stl<ColumnPtr>()};
    for (auto& pair : assignments) {
      auto handle =
          reinterpret_cast<const HiveColumnHandle*>(pair.second.get());
      auto schemaColumn = schemaTable->findColumn(handle->name());
      schemaColumns.push_back(schemaColumn);
      auto value = schemaColumn->value;
      Define(
          Column,
          column,
          toName(fmt::format("{}.{}", cname, handle->name())),
          baseTable,
          value);
      columns.push_back(column);
      renames_[pair.first] = column;
    }
    // baseTable->setRelation(*schemaTable, columns, schemaColumns);
    currentSelect_->tables.push_back(baseTable);
    return baseTable;
  }
  if (name == "Project") {
    auto project = reinterpret_cast<const core::ProjectNode*>(&node);
    auto names = project->names();
    auto exprs = project->projections();
    for (auto i = 0; i < names.size(); ++i) {
      Expr* expr = translateExpr(exprs[i]);
      renames_[names[i]] = expr;
    }
  }
  if (name == "HashJoin" || name == "MergeJoin") {
    translateJoin(*reinterpret_cast<const core::AbstractJoinNode*>(&node));
    return currentSelect_;
  }
  if (name == "Aggregation") {
    makeQueryGraph(*node.sources()[0]);
    currentSelect_->groupBy =
      translateGroupBy(*reinterpret_cast<const core::AggregationNode*>(&node));
    return currentSelect_;
  }
  if (name == "OrderBy") {
    return makeQueryGraph(*node.sources()[0]);
    currentSelect_->orderBy = translateOrderBy(*reinterpret_cast<const core::OrderByNode*>(&node));
    return currentSelect_;
  }
  if (name == "Limit") {
    auto limit = reinterpret_cast<const core::LimitNode*>(&node);
    currentSelect_->limit = limit->count();
    currentSelect_->offset = limit->offset();
  }
  return nullptr;
}

} // namespace facebook::velox::query
