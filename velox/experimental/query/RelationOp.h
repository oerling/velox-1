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

#include "velox/experimental/query/QueryGraph.h"
#include "velox/experimental/query/Schema.h"

namespace facebook::verax {

struct PlanState;

// Plan candidates.
//
// A candidate plan is constructed from  the above join graph/derived table
// tree specification.

// A physical operation on a relation. Has a per-row cost, a per-row
// fanout and a one-time setup cost. For example, a hash join probe
// has a fanout of 0.3 if 3 of 10 input rows are expected to hit, a
// constant small per-row cost that is fixed and a setup cost that is
// the one time cost of the build side subplan. The inputCardinality
// is a precalculated product of the left deep inputs for the hash
// probe. For a leaf table scan, input cardinality is 1 and the fanout
// is the estimated cardinality after filters, the unitCost is the
// cost of the scan and all filters. For an index lookup, the unit
// cost is a function of the index size and the input spacing and
// input cardinality. A lookup that hits densely is cheaper than one
// that hits sparsely. An index lookup has no setup cost.

 struct Cost {
// Cardinality of the output of the left deep input tree. 1 for a leaf
  // scan.
  float inputCardinality{1};

  // Cost of processing one input tuple. Complete cost of the operation for a
  // leaf.
  float unitCost{0};

  // 'fanout * inputCardinality' is the number of result rows. For a leaf scan,
  // this is the number of rows.
  float fanout{1};

  // One time setup cost. Cost of build subplan for the first use of a hash
  // build side. 0 for the second use of a hash build side. 0 for table scan
  // or index access.
  float setupCost{0};

  // Estimate of total data volume  for a hash join build or group/order
  // by/distinct / repartition. The memory footprint may not be this if the
  // operation is streaming or spills.
  float totalBytes{0};

  // Maximum memory occupancy. If the operation is blocking, e.g. group by, the
  // amount of spill is 'totalBytes' - 'peakResidentBytes'.
  float peakResidentBytes{0};

   /// If 'isUnit' shows the cost/cardinality for one row, else for 'inputCardinality' rows.
   std::string toString(bool detail, bool isUnit = false) const;
 };


 struct RelationOp : public Relation {
  RelationOp(
      RelType type,
      boost::intrusive_ptr<RelationOp> input,
      Distribution _distribution)
      : Relation(relType, _distribution, ColumnVector{}),
        input(std::move(input)) {}

  virtual ~RelationOp() = default;

  void operator delete(void* ptr) {
    queryCtx()->allocator().free(velox::HashStringAllocator::headerOf(ptr));
  }

  const Cost& cost() const {
    return cost_;
  }

  virtual void setCost(const PlanState& input);

  virtual std::string toString(bool recursive, bool detail) const {
    if (input && recursive) {
      return input->toString(true, detail);
    }
    return "";
  }

  // adds a line of cost information to 'out'
  void printCost(bool detail, std::stringstream& out) const;

  
  // thread local reference count. PlanObjects are freed when the
  // QueryGraphContext arena is freed, candidate plans are freed when no longer
  // referenced.
  mutable int32_t refCount{0};

  // Input of filter/project/group by etc., Left side of join, nullptr for a
  // leaf table scan.
  boost::intrusive_ptr<struct RelationOp> input;


 protected:
  Cost cost_;
};

using RelationOpPtr = boost::intrusive_ptr<RelationOp>;

 static inline void intrusive_ptr_add_ref(RelationOp* op) {
  ++op->refCount;
}

static inline void intrusive_ptr_release(RelationOp* op) {
  if (0 == --op->refCount) {
    delete op;
  }
}

struct Index;
using IndexPtr = Index*;

struct TableScan : public RelationOp {
  TableScan(
      RelationOpPtr input,
      Distribution _distribution,
      BaseTablePtr table,
      IndexPtr _index,
	    float fanout)
      : RelationOp(RelType::kTableScan, input, _distribution),
        baseTable(table),
        index(_index) {
    cost_.fanout = fanout;
  }

  
  /// Columns of base table available in 'index'.
  PlanObjectSet availableColumns();

  void setRelation(
      const ColumnVector& columns,
      const ColumnVector& schemaColumns);

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;

  // The base table reference. May occur in multiple scans if the base
  // table decomposes into access via secondary index joined to pk or
  // if doing another pass for late materialization.
  BaseTablePtr baseTable;

  // Index (or other materialization of table) used for the physical data
  // access.
  IndexPtr index;

  // Columns read from 'baseTable'. Can be more than 'columns' if
  // there are filters that need columns that are not projected out to
  // next op.
  PlanObjectSet extractedColumns;
  
  // Lookup keys, empty if full table scan.
  ExprVector keys;

  // If this is a lookup, 'joinType' can  be inner, left or anti.
  velox::core::JoinType joinType{velox::core::JoinType::kInner};

  ExprPtr joinFilter{nullptr};
};

struct Repartition : public RelationOp {
  Repartition(
      RelationOpPtr input,
      Distribution _distribution,
      const ColumnVector& _columns)
      : RelationOp(
            RelType::kRepartition,
            std::move(input),
            std::move(_distribution)) {
    columns = std::move(_columns);
  }

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

using RepartitionPtr = Repartition*;

struct Filter : public RelationOp {
  ExprPtr expr;
};

struct Project : public RelationOp {
  // Exprs. Output description is inherited from Relation.
  ExprVector exprs;
};

enum class JoinMethod { kHash, kMerge };

struct JoinOp : public RelationOp {
  JoinOp(
      JoinMethod _method,
      velox::core::JoinType _joinType,
      RelationOpPtr input,
      RelationOpPtr right,
      ExprVector leftKeys,
      ExprVector rightKeys,
      ExprPtr filter,
      float fanout,
      ColumnVector _columns)
    : RelationOp(RelType::kJoin, input, input->distribution),
        method(_method),
        joinType(_joinType),
    right(std::move(right)),
    leftKeys(std::move(leftKeys)),
    rightKeys(std::move(rightKeys)),
    filter(filter) {
    cost_.fanout = fanout;
    columns = std::move(_columns);
  }

  JoinMethod method;
  velox::core::JoinType joinType;
  RelationOpPtr right;
  ExprVector leftKeys;
  ExprVector rightKeys;
  ExprPtr filter;

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

using JoinOpPtr = JoinOp*;

/// Occurs as right input of JoinOp with type kHash. Contains the
/// cost and memory specific to building the table. Can be
/// referenced from multiple JoinOps. The unit cost * input
/// cardinality of this is counted as setup cost in the first
/// referencing join and not counted in subsequent ones.
struct HashBuild : public RelationOp {
  HashBuild(RelationOpPtr input, ExprVector _keys)
      : RelationOp(RelType::kHashBuild, input, input->distribution),
        keys(std::move(_keys)) {}

  int32_t buildId{0};
  ExprVector keys;

  void setCost(const PlanState& input) override;
};

using HashBuildPtr = HashBuild*;

struct Aggregation : public RelationOp {
  Aggregation(RelationOpPtr input, ExprVector _grouping)
      : RelationOp(
            RelType::kAggregation,
            input,
            input ? input->distribution : Distribution()),
        grouping(std::move(_grouping)) {}

  ExprVector grouping;

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to any() aggregates.
  PlanObjectSet dependentKeys;

  std::vector<AggregatePtr, QGAllocator<AggregatePtr>> aggregates;

  velox::core::AggregationNode::Step step{
      velox::core::AggregationNode::Step::kSingle};

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

} // namespace facebook::verax
