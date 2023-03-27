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
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"
#include "velox/expression/ExprToSubfieldFilter.h"

namespace facebook::verax {

using namespace facebook::velox;
using namespace facebook::velox::exec;

void filterUpdated(BaseTablePtr table) {
  auto optimization = queryCtx()->optimization();
  std::vector<core::TypedExprPtr> remainingConjuncts;
  connector::hive::SubfieldFilters subfieldFilters;
  for (auto filter : table->columnFilters) {
    auto typedExpr = optimization->toTypedExpr(filter);
    auto pair = velox::exec::toSubfieldFilter(typedExpr);
    if (!pair.second) {
      remainingConjuncts.push_back(std::move(typedExpr));
      continue;
    }
    subfieldFilters[std::move(pair.first)] = std::move(pair.second);
  }
  for (auto expr : table->filter) {
    remainingConjuncts.push_back(optimization->toTypedExpr(expr));
  }
  core::TypedExprPtr remainingFilter;
  for (auto conjunct : remainingConjuncts) {
    if (!remainingFilter) {
      remainingFilter = conjunct;
    } else {
      remainingFilter = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{remainingFilter, conjunct},
          "and");
    }
  }
  const char* connector = table->schemaTable->indices[0]
                              ->distribution()
                              .distributionType.locus->name();
  auto handle = std::make_shared<connector::hive::HiveTableHandle>(
      connector,
      table->schemaTable->name,
      true,
      std::move(subfieldFilters),
      remainingFilter);
  optimization->setLeafHandle(table->id(), handle);
}

RelationOpPtr addGather(RelationOpPtr op) {
  if (op->relType() == RelType::kOrderBy) {
    auto order = op->distribution();
    Distribution final = Distribution::gather(
        op->distribution().distributionType, order.order, order.orderType);
    Declare(Repartition, gather, op, final, op->columns());
    Declare(OrderBy, orderBy, gather, order.order, order.orderType);
    return orderBy;
  }
  Declare(
      Repartition,
      gather,
      op,
      Distribution::gather(op->distribution().distributionType),
      op->columns());
  return gather;
}

std::vector<ExecutableFragment> Optimization::toVeloxPlan(
    RelationOpPtr plan,
    const ExecutablePlanOptions& options) {
  options_ = options;
  std::vector<ExecutableFragment> stages;
  if (options_.numWorkers > 1) {
    plan = addGather(plan);
  }
  ExecutableFragment top;
  makeFragment(plan, top, stages);
  stages.push_back(std::move(top));
  return stages;
}

RowTypePtr Optimization::makeOutputType(const ColumnVector& columns) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < columns.size(); ++i) {
    names.push_back(columns[i]->name());
    types.push_back(toTypePtr(columns[i]->value().type));
  }
  return ROW(std::move(names), std::move(types));
}

core::TypedExprPtr Optimization::toTypedExpr(ExprPtr expr) {
  switch (expr->type()) {
    case PlanType::kColumn: {
      return std::make_shared<core::FieldAccessTypedExpr>(
          toTypePtr(expr->value().type), expr->as<Column>()->name());
    }
    case PlanType::kCall: {
      std::vector<core::TypedExprPtr> inputs;
      auto call = expr->as<Call>();
      for (auto arg : call->args()) {
        inputs.push_back(toTypedExpr(arg));
      }
      return std::make_shared<core::CallTypedExpr>(
          toTypePtr(expr->value().type), std::move(inputs), call->name());
    }
    case PlanType::kLiteral: {
      auto literal = expr->as<Literal>();
      return std::make_shared<core::ConstantTypedExpr>(
          toTypePtr(literal->value().type), literal->literal());
    }

    default:
      VELOX_FAIL("Cannot translate {} to TypeExpr", expr->toString());
  }
}

core::PlanNodePtr Optimization::makeFragment(
    RelationOpPtr op,
    ExecutableFragment& fragment,
    std::vector<ExecutableFragment>& stages) {
  switch (op->relType()) {
    case RelType::kProject: {
      auto input = makeFragment(op->input(), fragment, stages);
    }
    case RelType::kRepartition: {
      ExecutableFragment source;
      source.taskPrefix = fmt::format("stage{}", ++stageCounter_);
      source.fragment.planNode = makeFragment(op->input(), source, stages);
      stages.push_back(std::move(source));
      break;
    }
    case RelType::kTableScan: {
      auto scan = op->as<TableScan>();
      auto handle = leafHandle(scan->baseTable->id());
      if (!handle) {
        filterUpdated(scan->baseTable);
        handle = leafHandle(scan->baseTable->id());
        VELOX_CHECK(handle, "No table for scan {}", scan->toString(true, true));
      }
      std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          assignments;
      for (auto column : scan->columns()) {
        assignments[column->name()] =
            std::make_shared<connector::hive::HiveColumnHandle>(
								column->name(),
                connector::hive::HiveColumnHandle::ColumnType::kRegular,
                toTypePtr(column->value().type));
      }
      return std::make_shared<core::TableScanNode>(
          idGenerator_.next(),
          makeOutputType(scan->columns()),
          handle,
          assignments);
    }
    default:
      VELOX_FAIL("Unsupported RelationOp {}", op->relType());
  }
  return nullptr;
}

} // namespace facebook::verax
