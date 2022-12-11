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
#pragma once

#include "velox/common/base/SimdUtil.h"
#include "velox/core/PlanNode.h"
#include "velox/experimental/query/QueryGraph.h"

namespace facebook::velox::query {

struct ITypedExprHasher {
  size_t operator()(const core::ITypedExpr* expr) const {
    return expr->hash();
  }
};

struct ITypedExprComparer {
  bool operator()(const core::ITypedExpr* lhs, const core::ITypedExpr* rhs)
      const {
    return *lhs == *rhs;
  }
};

// Map for deduplicating ITypedExpr trees.
using ExprDedupMap = folly::F14FastMap<
    const core::ITypedExpr*,
    ExprPtr,
    ITypedExprHasher,
    ITypedExprComparer>;

class Plan;

/// Instance of query optimization. Comverts a plan and schema into an optimized
/// plan. Depends on QueryGraphContext being set on the calling thread.
class Optimization {
 public:
  Optimization(const core::PlanNode& plan, const Schema& schema);

  std::shared_ptr<const core::PlanNode> bestPlan();

 private:
  DerivedTablePtr makeQueryGraph();

  PlanObjectPtr makeQueryGraph(const core::PlanNode& node);
  ExprPtr translateExpr(const core::TypedExprPtr& expr);
  ExprPtr translateColumn(const std::string& name);
  ExprVector translateColumns(
      const std::vector<core::FieldAccessTypedExprPtr>& source);
  void translateJoin(const core::AbstractJoinNode& join);

  OrderByPtr translateOrderBy(const core::OrderByNode& order);
  GroupByPtr translateGroupBy(const core::AggregationNode& aggregation);

  void makePlans(
      DerivedTablePtr table,
      RelationOpPtr* FOLLY_NULLABLE left,
      const PlanObjectSet& boundColumns);

  const Schema& schema_;
  const core::PlanNode& inputPlan_;
  DerivedTablePtr root_;

  DerivedTablePtr currentSelect_;

  std::unordered_map<std::string, ExprPtr> renames_;

  ExprDedupMap exprDedup_;

  int32_t nameCounter_{0};

  std::unordered_map<PlanObjectSet, std::vector<Plan * FOLLY_NONNULL>> memo_;
};

/// Cheat sheet for selectivity keyed on ConnectorTableHandle::toString().
/// Values between 0 and 1.
std::unordered_map<std::string, float>& baseSelectivities();

/// Returns bits describing function 'name'.
FunctionSet functionBits(Name name);

class Plan {
  // The tables from original join graph that are included in this
  // plan. If this is a derived table in the original plan, the
  // covered object is the derived table, not its constituent
  // tables.
  PlanObjectSet tables_;

  // The produced columns. Includes input columns.
  PlanObjectSet columns_;

  // Columns that are fixed on input. Applies to index path for a derived table,
  // e.g. a left (t1 left t2) dt on dt.t1pk = a.fk. In a memo of dt inputs is
  // dt.pkt1.
  PlanObjectSet input_;

  float setupCost_{0};
  float perInputCost_{0};

  // The plan is made assuming that it will be applied to
  // 'planInputCardinality_' rows of input.
  float plannedInputCardinality_{1};

  RelationPtr root_;
};


} // namespace facebook::velox::query
