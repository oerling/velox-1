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
#include "velox/experimental/query/RelationOp.h"

namespace facebook::verax {

struct ITypedExprHasher {
  size_t operator()(const velox::core::ITypedExpr* expr) const {
    return expr->hash();
  }
};

struct ITypedExprComparer {
  bool operator()(
      const velox::core::ITypedExpr* lhs,
      const velox::core::ITypedExpr* rhs) const {
    return *lhs == *rhs;
  }
};

// Map for deduplicating ITypedExpr trees.
using ExprDedupMap = folly::F14FastMap<
    const velox::core::ITypedExpr*,
    ExprPtr,
    ITypedExprHasher,
    ITypedExprComparer>;

struct PlanState;

/// Item produced by optimization and kept in memo. Corresponds to
/// pre-costed physical plan with costs and data properties.
struct Plan {
  Plan(RelationOpPtr op, const PlanState& state);

  bool isStateBetter(const PlanState& state) const;

  RelationOpPtr op;

  Cost cost;

  // The tables from original join graph that are included in this
  // plan. If this is a derived table in the original plan, the
  // covered object is the derived table, not its constituent
  // tables.
  PlanObjectSet tables;

  // The produced columns. Includes input columns.
  PlanObjectSet columns;

  // Columns that are fixed on input. Applies to index path for a derived table,
  // e.g. a left (t1 left t2) dt on dt.t1pk = a.fk. In a memo of dt inputs is
  // dt.pkt1.
  PlanObjectSet input;

  std::string printCost() const;
  std::string toString(bool detail) const;
};

using PlanPtr = Plan*;

struct PlanSet {
  std::vector<std::unique_ptr<Plan>> plans;

  // Returns the best plan that produces 'distribution'. If the best plan has
  // some other distribution, sets 'needsShuffle ' to true.
  PlanPtr best(const Distribution& distribution, bool& needShuffle);

  /// Compares 'plan' to already seen plans and retains it if it is interesting,
  /// e.g. better than the best so far or has an interesting order. Returns the plan
  /// if retained, nullptr if not.
  PlanPtr addPlan(RelationOpPtr plan, PlanState& state);
};

struct PlanState {
  PlanState() = default;

  PlanState(PlanPtr plan) : cost(plan->cost) {}

  // The derived table from which the tables are drawn.
  DerivedTablePtr dt{nullptr};

  // The tables that have been placed so far.
  PlanObjectSet placed;

  // The columns that have a value.
  PlanObjectSet columns;

  // The columns that need a value at the end of the plan. A dt can
  // be planned for just join columns or all payload.
  PlanObjectSet targetColumns;

  // lookup keys for an index based derived table.
  PlanObjectSet input;

  Cost cost;

  bool HasCutoff{true};

  // Interesting completed plans for the dt being planned. For
  // example, best by cost and maybe plans with interesting orders.
  PlanSet plans;

  // Caches results of downstreamColumns(). This is a pure function of 'placed'
  // a'targetColumns' and 'dt'.
  mutable std::unordered_map<PlanObjectSet, PlanObjectSet>
      downstreamPrecomputed;

  void addCost(RelationOp& op);

  /// The set of columns referenced in unplaced joins/filters union
  /// targetColumns. Gets smaller as more tables are placed.
  PlanObjectSet downstreamColumns() const;

  std::string printCost() const;
  std::string printPlan(RelationOpPtr op, bool detail) const;
};

struct StateSaver {
 public:
  StateSaver(PlanState& state)
      : state_(state),
        placed_(state.placed),
        columns_(state.columns),
        cost_(state.cost) {}

  ~StateSaver() {
    state_.placed = std::move(placed_);
    state_.columns = std::move(columns_);
    state_.cost = cost_;
  }

 private:
  PlanState& state_;
  PlanObjectSet placed_;
  PlanObjectSet columns_;
  const Cost cost_;
};

struct JoinSide {
  PlanObjectPtr table;
  ExprVector& keys;
  bool isOptional;
  bool isExists;
  bool isNotExists;

  /// Returns the join type to use if 'this' is the right side.
  velox::core::JoinType leftJoinType() {
    if (isNotExists) {
      return velox::core::JoinType::kAnti;
    }
    if (isExists) {
      return velox::core::JoinType::kLeftSemiFilter;
    }
    if (isOptional) {
      return velox::core::JoinType::kLeft;
    }
    return velox::core::JoinType::kInner;
  }
};

// Represents the next table/derived table to join. May consist of several
// tables for a bushy build side.
struct JoinCandidate {
  JoinCandidate() = default;

  JoinCandidate(JoinPtr _join, PlanObjectPtr _right, float _fanout)
      : join(_join), tables({_right}), fanout(_fanout) {}

  // Returns the join side info for 'table'. If 'other' is set, returns the
  // other side.
  JoinSide sideOf(PlanObjectPtr side, bool other = false) const;

  JoinPtr join{nullptr};

  // Tables to join on the build side. The tables must not occur on the left
  // side.
  std::vector<PlanObjectPtr> tables;

  // Joins imported from the left side for reducing a build
  // size. These could be ignored without affecting the result but can
  // be included to restrict the size of build, e.g. lineitem join
  // part left (partsupp exists part) would have the second part in
  // 'existences' and partsupp in 'tables' because we know that
  // partsupp will not be probed with keys that are not in part, so
  // there is no point building with these.
  std::vector<PlanObjectSet> existences;

  // Number of right side hits for one row on the left. The join
  // selectivity in 'tables' affects this but the selectivity in
  // 'existences' does not.
  float fanout;
};

struct MemoKey {
  bool operator==(const MemoKey& other) const;
  size_t hash() const;

  PlanObjectSet columns;
  PlanObjectSet tables;
  std::vector<PlanObjectSet> existences;
};

} // namespace facebook::verax

namespace std {
template <>
struct hash<::facebook::verax::MemoKey> {
  size_t operator()(const ::facebook::verax::MemoKey& key) const {
    return key.hash();
  }
};
} // namespace std

namespace facebook::verax {

/// Instance of query optimization. Comverts a plan and schema into an optimized
/// plan. Depends on QueryGraphContext being set on the calling thread.
class Optimization {
 public:
  Optimization(
      const velox::core::PlanNode& plan,
      const Schema& schema,
      int32_t traceFlags = 0);

  PlanPtr bestPlan();

  std::shared_ptr<const velox::core::PlanNode> toVeloxPlan(RelationOpPtr plan) {
    return nullptr;
  }

 private:
  DerivedTablePtr makeQueryGraph();

  PlanObjectPtr makeQueryGraph(const velox::core::PlanNode& node);
  void setDerivedTableOutput(
      DerivedTablePtr dt,
      const velox::core::PlanNode& planNode);
  ExprPtr translateExpr(const velox::core::TypedExprPtr& expr);

  ExprPtr translateColumn(const std::string& name);
  ExprVector translateColumns(
      const std::vector<velox::core::FieldAccessTypedExprPtr>& source);
  void translateJoin(const velox::core::AbstractJoinNode& join);

  OrderByPtr translateOrderBy(const velox::core::OrderByNode& order);
  AggregationPtr translateGroupBy(
      const velox::core::AggregationNode& aggregation);

  PlanPtr makePlan(
      const MemoKey& key,
      const Distribution& distribution,
      const PlanObjectSet& boundColumns,
      PlanState& state,
      bool& needsShuffle);

  std::vector<JoinCandidate> nextJoins(DerivedTablePtr dt, PlanState& state);

  // Adds group by, order by, top k to 'plan'. Updates 'plan' if
  // relation ops added.  Sets cost in 'state'.
  void
  addPostprocess(DerivedTablePtr dt, RelationOpPtr& plan, PlanState& state);

  const Schema& schema_;
  const velox::core::PlanNode& inputPlan_;
  DerivedTablePtr root_;

  void makeJoins(RelationOpPtr plan, PlanState& state);

  void addJoin(
      DerivedTablePtr dt,
      const JoinCandidate& candidate,
      const RelationOpPtr& plan,
      PlanState& state);
  void joinByIndex(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state);

  void joinByHash(
      const RelationOpPtr& plan,
      const JoinCandidate& candidate,
      PlanState& state);

  DerivedTablePtr currentSelect_;

  std::unordered_map<std::string, ExprPtr> renames_;

  ExprDedupMap exprDedup_;

  int32_t nameCounter_{0};

  std::unordered_map<MemoKey, PlanSet> memo_;
  int32_t traceFlags_{0};
};

/// Cheat sheet for selectivity keyed on ConnectorTableHandle::toString().
/// Values between 0 and 1.
std::unordered_map<std::string, float>& baseSelectivities();

/// Returns bits describing function 'name'.
FunctionSet functionBits(Name name);

} // namespace facebook::verax
