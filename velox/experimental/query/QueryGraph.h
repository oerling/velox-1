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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"

namespace facebook::velox::query {

/// Base data structures for plan candidate generation.

struct PlanObject;

using PlanObjectPtr = PlanObject* FOLLY_NONNULL;

class QueryGraphContext {
 public:
  QueryGraphContext() {}

  int32_t newId() {
    return lastId_++;
  }

  HashStringAllocator* allocator_;
  int32_t lastId_{0};

  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectPtr> objects_;
};

QueryGraphContext& ctx() {
  thread_local QueryGraphContext context;
  return context;
}

#define create(T, destination, ...)                         \
  T* destination = reinterpret_cast<T*>(malloc(sizeof(T))); \
  destination = new (data) T(__VA_ARGS__);

/// Pointers are name <type>Ptr and defined to be raw pointers. We
/// expect arena allocation with a whole areena freed after plan
/// selection. The different pointers could also be declared as smart
/// pointers for e.g. reference counting. std::shared_ptr would be
/// expensive but a non thread safe intrusive_ptr could be an
/// alternative.

/// The join structure is described as a tree of derived tables with
/// base tables as leaves. Joins are described as join graph
/// edges. Edges describe direction for non-inner joins. Scalar and
/// existence subqueries are flattened into derived tables or base
/// tables. The join graph would represent select ... form t where
/// exists(x) or exists(y) as a derived table of three joined tables
/// where the edge from t to x and t to y is directed and qualified as
/// left semijoin. The semijoins project out one column, an existence
/// flag. The filter would be expresssed as a conjunct under the top
/// derived table with x-exists or y-exists.

// Enum for types of plan candidate nodes.

enum class PlanType { kTable, kDerivedTable, kExpr, kProject, kFilter, kJoin };

struct PlanObject {
  PlanType type;
  int32_t id;
};

struct Value : public PlanObject {
  const Type* type;
  variant min;
  variant max;
  int64_t cardinality;
  bool nullable;
};
struct Expr : public Value {};
struct Relation;
using RelationPtr = Relation*;

struct Column : public Expr {
  RelationPtr relation;
  std::string name;
};

using ColumnPtr = Column*;
using ExprPtr = Expr*;

struct Call : public Expr {
  char* func;
  std::vector<ExprPtr> args;
};

using ExprPtr = Expr*;

struct Equivalence {
  std::vector<ExprPtr> exprs;
  // Corresponds pairwise to 'exprs'. True if the Expr comes from an
  // outer optional side key join and is therefore null or equal.
  std::vector<bool> nullable;
};

enum class OrderType {
  kAscNullsFirst,
  kAscNullsLast,
  kDescNullsFirst,
  kDescNullsLast
};

// Describes output of relational operator. If base table, cardinality is after
// filtering, column value ranges are after filtering.
struct Distribution : public PlanObject {
  int64_t cardinality;

  // Number of workers producing the data, i.e. width of shuffle. 1 regardless f
  // parallelism if the data has no partitioning key.
  int32_t numPartitions;

  // Partitioning columns. The values of these columns determine which of
  // 'numPartitions' contains any given row. This does not specify the partition
  // function (e.g. Hive bucket or range partition).
  std::vector<ColumnPtr> partition;

  // Ordering columns. Each partition is ordered by these. Specifies that
  // streaming group by or merge join are possible.
  std::vector<ColumnPtr> order;

  // Corresponds 1:1 to 'order'
  std::vector<OrderType> orderType;

  // Specifies the selectivity between the source of the ordered data
  // and 'this'. For example, if orders join lineitem and both are
  // ordered on orderkey and there is a 1/1000 selection on orders,
  // the distribution after the filter would have a spacing of 1000,
  // meaning that lineitem is hit every 1000 irder, meaning that an
  // index join with lineitem would skip 4000 rows between hits
  // because lineitem has an average of 4 repeats of orderkey.
  int64_t spacing{-1};

  // specifies that data in each partition is clustered by these columns, i.e.
  // all rows with any value k1=v1,...kn =vn are consecutive. Specifies that
  // once the value of any clustering column changes between consecutive rows,
  // the same combination of clustering columns will not repeat. means that a
  // final group by can be flushed when seeing a change in clustering columns.
  std::vector<ColumnPtr> clustering;

  // True if the data is replicated to 'numPartitions'.
  bool isBroadcast;
};

struct FilteredColumn {
  // The single column on which 'expr' depends.
  ColumnPtr column;
  // Filter normalized so that Column is leftmost argument if possible.
  ExprPtr filter;
  // e.g 0.2 for 1/5 passing.
  float selectivity;
};

using FilteredColumnPtr = FilteredColumn*;

// Represents a possibly directional binary join edge. A join
// hyperedge like a.k = b.k + c.k is represented as non-join filters
// in the containing derived table. a.k can be used as a lookup key
// if b and c are placed to the left of a.  if left or right are
// semi, anti or optional, the join can only be placed after the
// inner side is placed. If neither side is optional, the edge is
// non-directional and whichever side is not placed can be added. If
// both sides are optional (full outer join) then the edge is
// non-directional.
struct JoinEdge {
  Relation* left;
  Relation* right;
  // Leading left side join keys.
  std::vector<ExprPtr> leftKeys;
  // Leading right side join keys, compared equals to 1:1 to 'leftKeys'.
  std::vector<ExprPtr> rightKeys;

  // Join condition for any non-equality  conditions for non-inner joins.
  Expr* condition;

  // True if an unprobed right side row produces a result with right side
  // columns set and left side columns as null. Possible only be hash or merge.
  bool leftOptional;

  // True if a right side miss produces a row with left side columns
  // ad a null for right side columns (left outer join). A full outer
  // join has both left and right optional.
  bool rightOptional;

  // True if the right side is only checked for existence of a match. If
  // rightOptional is set, this can project out a null for misses.
  bool rightSemi;

  // True if the join is a right semijoin. This is possible only by hash or
  // merge.
  bool leftSemi;
};

using JoinEdgePtr = JoinEdge*;

struct Relation : public PlanObject {
  Distribution distribution;
  std::vector<ColumnPtr> columns;
};

struct Table : public Relation {
  // Correlation name, i.e. the AS in the FROM or the 'table' in table.column
  // notation.
  std::string cname;
};

using TablePtr = Table*;

struct BaseTable : public Table {
  // Name with catalog.schema.tablename.
  std::string name;
};

// Aggregate function. The aggregation and arguments are in the
// inherited Expr. The Value pertains to the aggregation
// result. adds aggregate-only attributes to Expr.
struct Aggregate : public Expr {
  bool distinct{false};
  ExprPtr condition;
};

using AggregatePtr = Aggregate*;

struct GroupBy : public Relation {
  std::vector<ExprPtr> grouping;
  std::vector<AggregatePtr> aggregates;
};
using GroupByPtr = GroupBy*;

struct OrderBy : public Relation {
  std::vector<ExprPtr> keys;
};

using OrderByPtr = OrderBy*;

struct DerivedTable : public Table {
  // Columns projected out. Visible in the enclosing query.
  std::vector<ColumnPtr> columns;

  // Exprs projected out.
  std::vector<ExprPtr> exprs;

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are under the table.
  std::vector<TablePtr> tables;
  // Join edges between 'tables'. Captures join structure. e.g. a left
  // join (b join c) has a and a derived table with (b join c) in
  // 'tables' and a join edge between these in 'joins'. The join edge
  // has the right side as optional. Join filters are in join edges.
  std::vector<JoinEdgePtr> joins;
  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  std::vector<ExprPtr> conjuncts;

  GroupByPtr groupBy;
  OrderByPtr orderBy;
  int32_t limit{-1};
  int32_t offset{0};
};

using DerivedTablePtr = DerivedTable*;

// Plan candidates.
//
// A candidate plan is constructed from  the above join graph/derived table tree
// specification.

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
struct RelationOp : public Relation {
  // Cost Cardinality of the output of the left deep input tree. 1 for a leaf
  // scan.
  float inputCardinality;

  // Cost of processing one input tuple. Complete cost of the operation for a
  // leaf.
  float unitCost;

  // 'fanout * inputCardinality' is the number of result riws.
  float fanout;

  // One time setup cost. Cost of build subplan for the first use of a hash
  // build side. 0 for the second use of a hash build side. 0 for table scan or
  // index access.
  float setupCost;

  // Estimate of total size for a hash join build or group/order
  // by/distinct. This does not account for spilling.
  float peakSize;

  // Maximum memory occupancy. Smaller than 'peakSize' if spilling is
  // expected. Ratio of peakSize / peakResident gives the spill, which
  // is reflected in higher per-row unit cost.
  float peakResident;
};

using RelationOpPtr = RelationOp*;

struct TableScan : public RelationOp {
  // Left side of index join, nullptr for a leaf table scan.
  RelationPtr input;

  // Base table. If the table is vertically partitioned (e.g. side tables) this
  // will be apparent at run time from the splits for the scan, not included
  // here.
  std::string table;

  // Index (or other materialization of table) used for the physical data
  // access. Empty for e.g. hive base table scan.
  std::string index;

  // Lookup keys, empty if full table scan.
  std::vector<ExprPtr> keys;

  // Leading key parts of index, 1:1 equal to 'keys'.
  std::vector<ColumnPtr> keyColumns;

  // Projected columns, does not necessarily include columns in keys or filters.
  std::vector<ColumnPtr> projectedColumns;

  // Filters involving only columns of this.
  std::vector<FilteredColumnPtr> filters;
};

struct Filter : public RelationOp {
  ExprPtr expr;
};

struct Project : public RelationOp {
  // Exprs. Output description is inherited from Relation.
  std::vector<ExprPtr> exprs;
};

struct HashJoin : public RelationOp {
  core::JoinType joinType;
  RelationOpPtr left;
  RelationOpPtr right;
  std::vector<ColumnPtr> leftKeys;
  std::vector<ColumnPtr> rightKeys;
  ExprPtr filter;
};

class Index : public Relation {};

class SchemaTable {
  std::vector<Index> indices_;
};

class Schema {
  std::unordered_map<std::string, std::unique_ptr<SchemaTable>> tables;
};

} // namespace facebook::velox::query
