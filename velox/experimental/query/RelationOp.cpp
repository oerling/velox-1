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

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"
#include "velox/experimental/query/QueryGraph.h"

namespace facebook::verax {

void TableScan::setRelation(
    const ColumnVector& columns,
    const ColumnVector& schemaColumns) {
  auto cc =
      transform<ColumnVector>(columns, [](auto& c) { return c->schemaColumn; });
  distribution.cardinality =
      index->distribution.cardinality * baseTable->filterSelectivity;
  // if all partitioning columns are projected, the output is partitioned.
  if (isSubset(
          toRangeCast<ColumnPtr>(index->distribution.partition),
          toRange(schemaColumns))) {
    distribution.partition = index->distribution.partition;
    replace(
        toRangeCast<ColumnPtr>(distribution.partition),
        toRange(schemaColumns),
        &columns[0]);
    distribution.distributionType = index->distribution.distributionType;
  }
  auto numPrefix = prefixSize(
      toRangeCast<ColumnPtr>(index->distribution.order),
      toRange(schemaColumns));
  if (numPrefix > 0) {
    distribution.order = index->distribution.order;
    distribution.order.resize(numPrefix);
    distribution.orderType = index->distribution.orderType;
    distribution.orderType.resize(numPrefix);
    replace(
        toRangeCast<ColumnPtr>(distribution.order),
        toRange(schemaColumns),
        &columns[0]);
    if (index->distribution.numKeysUnique <= numPrefix) {
      distribution.numKeysUnique = index->distribution.numKeysUnique;
    }
  }
  this->columns = columns;
}

PlanObjectSet TableScan::availableColumns() {
  // The columns of base table that exist in 'index'.
  PlanObjectSet result;
  for (auto column : index->columns) {
    for (auto baseColumn : baseTable->columns) {
      if (baseColumn->name == column->name) {
        result.add(baseColumn);
        break;
      }
    }
  }
  return result;
}

std::string Cost::toString(bool detail, bool isUnit) const {
  std::stringstream out;
  float multiplier = isUnit ? 1 : inputCardinality;
  out << succinctNumber(fanout * multiplier) << " rows "
      << succinctNumber(unitCost * multiplier) << "CU";
  if (setupCost > 0) {
    out << ", setup " << succinctNumber(setupCost) << "CU";
  }
  if (totalBytes) {
    out << " " << velox::succinctBytes(totalBytes);
  }
  return out.str();
}

void RelationOp::printCost(bool detail, std::stringstream& out) const {
  auto ctx = queryCtx();
  if (ctx && ctx->contextPlan()) {
    auto plan = ctx->contextPlan();
    auto totalCost = plan->cost.unitCost + plan->cost.setupCost;
    auto pct = 100 * cost_.inputCardinality * cost_.unitCost / totalCost;
    out << " " << std::fixed << std::setprecision(2) << pct << "% ";
  }
  if (detail) {
    out << " " << cost_.toString(detail, false) << std::endl;
  }
}

const char* joinTypeLabel(velox::core::JoinType type) {
  switch (type) {
    case velox::core::JoinType::kLeft:
      return "left";
    case velox::core::JoinType::kRight:
      return "right";
    case velox::core::JoinType::kLeftSemiFilter:
      return "exists";
    case velox::core::JoinType::kLeftSemiProject:
      return "exists-flag";
    case velox::core::JoinType::kAnti:
      return "not exists";
    default:
      return "";
  }
}

std::string TableScan::toString(bool /*recursive*/, bool detail) const {
  std::stringstream out;
  if (input) {
    out << input->toString(true, detail);
    out << " *I " << joinTypeLabel(joinType);
  }
  out << baseTable->schemaTable->name << " " << baseTable->cname;
  if (detail) {
    printCost(detail, out);
    if (!input) {
      out << distribution.toString() << std::endl;
    }
  }
  return out.str();
}

std::string JoinOp::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input->toString(true, detail);
  }
  out << "*" << (method == JoinMethod::kHash ? "H" : "M") << " "
      << joinTypeLabel(joinType);
  printCost(detail, out);
  if (recursive) {
    out << " (" << right->toString(true, detail) << ")";
    if (detail) {
      out << std::endl;
    }
  }
  return out.str();
}

std::string Repartition::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input->toString(true, detail) << " ";
  }
  out << (distribution.isBroadcast ? "broadcast" : "shuffle") << " ";
  if (detail && !distribution.isBroadcast) {
    out << distribution.toString();
    printCost(detail, out);
  } else if (detail) {
    printCost(detail, out);
  }
  return out.str();
}

std::string Aggregation::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input->toString(true, detail) << " ";
  }
  out << velox::core::AggregationNode::stepName(step) << " agg";
  printCost(detail, out);
  return out.str();
}

std::string HashBuild::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input->toString(true, detail) << " ";
  }
  out << " Build ";
  printCost(detail, out);
  return out.str();
}

} // namespace facebook::verax
