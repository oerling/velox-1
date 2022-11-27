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

PlanObjectPtr Optimization::makeQueryGraph(const core::PlanNode& node) {
  auto name = node.name();
  if (name == "TableScan") {
    auto tableScan = reinterpret_cast<const core::TableScanNode*>(&node);
    auto tableHandle = dynamic_cast<const HiveTableHandle*>(tableScan->tableHandle().get());
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
    //baseTable->setRelation(*schemaTable, columns, schemaColumns);
    currentSelect_->tables.push_back(baseTable);
    return baseTable;
  }
  if (name == "Project") {
  }
  if (name == "HashJoin") {
  }
  return nullptr;
}

} // namespace facebook::velox::query
