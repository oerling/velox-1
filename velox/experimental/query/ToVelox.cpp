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
#include "velox/exec/test/utils/PlanBuilder.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

using namespace facebook::velox;

RelationOpPtr addGather(RelationOpPtr op) {
  if (op->relType() = RelType::kOrderBy) {
    auto order = op->distribution();
    Distribution final =
        Distribution::gather(plan, order->order, order->orderType);
    Declare(Repartition, gather, plan, final, plan->columns());
    Declare(OrderBy, orderBy, gather, order->order(), order->orderType());
    return orderBy;
  }
  Declare(
      Repartition,
      gather,
      plan,
      Distribution::gather(plan->distribution()->distributionType),
      plan->columns());
  return gather;
}

std::vector<ExecutablePlan> Optimization::toVeloxPlan(
    RelationOpPtr plan,
    const ExecutablePlanOptions& options) {
  options_ = options;
  std::vector<ExecutablePlan> stages;
  if (options_.numWorkers > 1) {
    plan = addGather(plan);
  }
  ExecutableFragment top;
  makeFragment(plan, top, stages);
  stages.push_back(std::move(top));
  return stages;
}

core::TypedExprPtr Optimization::toTypedExpr(ExprPtr expr) {
  switch (expr->type()) {
    case PlanType::kColumn: {
      return std::make_shared<core::FieldAccessTypedExpr>(
          toTypePtr(expr->type()), expr->as<Column>()->name());
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
    case RelType::Project: {
      auto input = makeFragment(op->sources()[0], fragment, stages);
    }
    default:
      VELOX_FAIL("Unsupported RelationOp {}", op->relType());
  }
}
