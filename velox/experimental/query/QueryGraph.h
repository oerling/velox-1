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

using Name = const char*;

template <typename T>
using PtrSpan = folly::Range<T**>;

struct PlanObject;

using PlanObjectPtr = PlanObject* FOLLY_NONNULL;

class QueryGraphContext {
 public:
  QueryGraphContext(HashStringAllocator& allocator)
      : allocator_(allocator), stlAllocator_(StlAllocator<void*>(&allocator)) {}

  Name toName(std::string_view str);

  int32_t newId(PlanObject* FOLLY_NONNULL object) {
    objects_.push_back(object);
    return objects_.size() - 1;
  }

  StlAllocator<void*>* stlAllocator() & {
    return &stlAllocator_;
  }

  HashStringAllocator& allocator() {
    return allocator_;
  }

  HashStringAllocator& allocator_;
  StlAllocator<void*> stlAllocator_;

  PlanObjectPtr objectAt(int32_t id) {
    return objects_[id];
  }

  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectPtr> objects_;
  std::unordered_set<std::string_view> names_;
};

inline QueryGraphContext*& queryCtx() {
  thread_local QueryGraphContext* context;
  return context;
}

template <typename T>
StlAllocator<T> stl() {
  return *reinterpret_cast<StlAllocator<T>*>(queryCtx()->stlAllocator());
}

#define Define(T, destination, ...)                         \
  T* destination = reinterpret_cast<T*>(malloc(sizeof(T))); \
  new (destination) T(__VA_ARGS__);
#define DefineDefault(T, destination)                       \
  T* destination = reinterpret_cast<T*>(malloc(sizeof(T))); \
  new (destination) T();

/// Converts std::string to name used in query graph objects. raw pointer to
/// arena allocated const chars.
Name toName(const std::string& string);

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

enum class PlanType {
  kTable,
  kDerivedTable,
  kColumn,
  kLiteral,
  kCall,
  kProject,
  kFilter,
  kJoin
};

Name planTypeName(PlanType type);

inline bool isExprType(PlanType type) {
  return type == PlanType::kColumn || type == PlanType::kCall ||
      type == PlanType::kLiteral;
}

struct PlanObject {
  PlanObject(PlanType _type) : type(_type) {
    id = queryCtx()->newId(this);
  }

  template <typename T>
  T as() {
    return reinterpret_cast<T>(this);
  }

  template <typename T>
  const T as() const {
    return reinterpret_cast<const T>(this);
  }

  virtual folly::Range<PlanObjectPtr*> children() {
    return folly::Range<PlanObjectPtr*>(nullptr, nullptr);
  }

  template <typename Func>
  void preorderVisit(Func func) {
    func(this);
    for (auto child : children()) {
      child->preorderVisit(func);
    }
  }

  std::string toString() {
    return fmt::format("#{}", id);
  }
  PlanType type;
  int32_t id;
};

struct Expr;
using ExprPtr = Expr*;
struct Column;
using ColumnPtr = Column*;

class PlanObjectSet {
 public:
  bool contains(PlanObjectPtr object) {
    return object->id < end_ && bits::isBitSet(bits_.data(), object->id);
  }

  bool operator==(const PlanObjectSet& other) const;

  size_t hash() const;

  void add(PlanObjectPtr ptr) {
    auto id = ptr->id;
    ensureSize(id);
    adjustRange(id);
    bits::setBit(bits_.data(), id);
  }

  void erase(PlanObjectPtr object) {
    if (object->id < end_) {
      bits::clearBit(bits_.data(), object->id);
    }
  }

  void unionColumns(ExprPtr expr);

  void unionSet(const PlanObjectSet& other);

  template <typename Func>
  void forEach(Func func) const {
    auto ctx = queryCtx();
    bits::forEachSetBit(
        bits_.data(), begin_, end_, [&](auto i) { func(ctx->objectAt(i)); });
  }

  std::vector<PlanObjectPtr> objects() const {
    std::vector<PlanObjectPtr> result;
    forEach([&](auto object) { result.push_back(object); });
    return result;
  }

 private:
  void ensureSize(int32_t id) {
    ensureWords(bits::nwords(id + 1));
  }

  void adjustRange(int32_t id) {
    if (id < begin_) {
      begin_ = id;
    }
    if (id >= end_) {
      end_ = id + 1;
    }
  }
  void ensureWords(int32_t size) {
    if (bits_.size() < size) {
      bits_.resize(size);
    }
  }

  std::vector<uint64_t, StlAllocator<uint64_t>> bits_{stl<uint64_t>()};
  int32_t begin_{0};
  int32_t end_{0};
};

struct Value {
  Value() = default;
  Value(const Type* _type, int64_t _cardinality)
      : type(_type), cardinality(_cardinality) {}

  const Type* FOLLY_NONNULL type;
  variant min;
  variant max;
  const float cardinality;
  // 0 means no nulls, 0.5 means half are null.
  float nullFraction{0};
  bool nullable{true};
};

struct Relation;
using RelationPtr = Relation*;

struct Expr : public PlanObject {
  Expr(PlanType type, Value _value) : PlanObject(type), value(_value) {}

  // Returns the single base or derived table 'this' depends on, nullptr if
  // 'this' depends on none or multiple tables.
  PlanObjectPtr singleTable();

  PlanObjectSet allTables() const;

  PlanObjectSet columns;
  Value value;
};

using ExprVector = std::vector<ExprPtr, StlAllocator<ExprPtr>>;

struct Equivalence;
using EquivalencePtr = Equivalence*;

struct Literal : public Expr {
  Literal(Value value, variant _literal)
      : Expr(PlanType::kLiteral, value), literal(_literal) {}
  variant literal;
};

struct Column : public Expr {
  Column(Name _name, PlanObjectPtr _relation, Value value)
      : Expr(PlanType::kColumn, value), name(_name), relation(_relation) {
    columns.add(this);
  }

  void equals(ColumnPtr other);

  Name name;
  PlanObjectPtr relation;
  EquivalencePtr equivalence;
};

template <typename T>
inline folly::Range<T*> toRange(const std::vector<T, StlAllocator<T>>& v) {
  return folly::Range<T*>(const_cast<T*>(v.data()), v.size());
}

template <typename T, typename U>
inline folly::Range<T*> toRangeCast(const std::vector<U, StlAllocator<U>>& v) {
  return folly::Range<T*>(
      reinterpret_cast<T*>(const_cast<U*>(v.data())), v.size());
}

using ColumnVector = std::vector<ColumnPtr, StlAllocator<ColumnPtr>>;

class FunctionSet {
 public:
  FunctionSet() : set_(0) {}
  FunctionSet(uint32_t set) : set_(set) {}

  bool contains(int32_t item) {
    return 0 != (set_ & (1UL << item));
  }

  FunctionSet operator|(const FunctionSet& other) {
    return FunctionSet(set_ | other.set_);
  }

 private:
  uint64_t set_;
};

struct Call : public Expr {
  Call(Name _func, Value value, ExprVector _args, FunctionSet _functions)
      : Expr(PlanType::kCall, value),
        func(_func),
        args(std::move(_args)),
        functions(_functions) {
    for (auto arg : args) {
      columns.unionSet(arg->columns);
    }
  }

  Name func;
  ExprVector args;

  // Columns this depends on.
  PlanObjectSet columns;
  // Set of functions used in 'this' and 'args'.
  FunctionSet functions;
};

using CallPtr = Call*;

struct Equivalence {
  ColumnVector columns{stl<ColumnPtr>()};
  ;
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

/// Method for determining a partition given an ordered list of partitioning
/// keys. Hive hash is an example, range partitioning is another. Add values
/// here for more types.
enum class ShuffleMode { kNone, kHive };

/// Distribution of data. 'numPartitions' is 1 if the data is not partitioned.
/// There is copartitioning if the DistributionType is the same on both sides
/// and both sides have an equal number of 1:1 type matched partitioning keys.
struct DistributionType {
  bool operator==(const DistributionType& other) {
    return mode == other.mode && numPartitions == other.numPartitions;
  }

  ShuffleMode mode{ShuffleMode::kNone};
  int32_t numPartitions{1};
};

// Describes output of relational operator. If base table, cardinality is
// after filtering, column value ranges are after filtering.
struct Distribution {
  int64_t cardinality;

  DistributionType distributionType;

  // Partitioning columns. The values of these columns determine which of
  // 'numPartitions' contains any given row. This does not specify the
  // partition function (e.g. Hive bucket or range partition).
  ExprVector partition{stl<ExprPtr>()};

  // Ordering columns. Each partition is ordered by these. Specifies that
  // streaming group by or merge join are possible.
  ExprVector order{stl<ExprPtr>()};

  // Corresponds 1:1 to 'order'
  std::vector<OrderType> orderType;

  // Number of leading elements of 'order' such that these uniquely
  // identify a row. 0 if there is no uniqueness.
  int32_t numKeysUnique{0};

  // Specifies the selectivity between the source of the ordered data
  // and 'this'. For example, if orders join lineitem and both are
  // ordered on orderkey and there is a 1/1000 selection on orders,
  // the distribution after the filter would have a spacing of 1000,
  // meaning that lineitem is hit every 1000 orders, meaning that an
  // index join with lineitem would skip 4000 rows between hits
  // because lineitem has an average of 4 repeats of orderkey.
  int64_t spacing{-1};

  // specifies that data in each partition is clustered by these columns, i.e.
  // all rows with any value k1=v1,...kn =vn are consecutive. Specifies that
  // once the value of any clustering column changes between consecutive rows,
  // the same combination of clustering columns will not repeat. means that a
  // final group by can be flushed when seeing a change in clustering columns.
  ColumnVector clustering{stl<ColumnPtr>()};

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
struct Join {
  // Leading left side join keys.
  ExprVector leftKeys{stl<ExprPtr>()};
  // Leading right side join keys, compared equals to 1:1 to 'leftKeys'.
  ExprVector rightKeys{stl<ExprPtr>()};

  PlanObjectPtr leftTable{nullptr};
  PlanObjectPtr rightTable{nullptr};

  // 'rightKeys' select max 1 'leftTable' row.
  bool leftUnique{false};

  // 'leftKeys' select max 1 'rightTable' row.
  bool rightUnique{false};
  // number of right side rows selected for one row on the left.
  float lrFanout{1};

  // Number of left side rows selected for one row on the right.
  float rlFanout{1};

  // Join condition for any non-equality  conditions for non-inner joins.
  Expr* condition;

  // True if an unprobed right side row produces a result with right side
  // columns set and left side columns as null. Possible only be hash or
  // merge.
  bool leftOptional{false};

  // True if a right side miss produces a row with left side columns
  // ad a null for right side columns (left outer join). A full outer
  // join has both left and right optional.
  bool rightOptional{false};

  // True if the right side is only checked for existence of a match. If
  // rightOptional is set, this can project out a null for misses.
  bool rightSemi{false};

  // True if the join is a right semijoin. This is possible only by hash or
  // merge.
  bool leftSemi{false};

  void guessFanout();
};

using JoinPtr = Join*;

/// Differentiates between base tables and operator results.
enum class RelType { kBase, kOperator };

struct Relation {
  Relation() = default;

  Relation(
      RelType _relType,
      Distribution _distribution,
      const ColumnVector& _columns)
      : relType(_relType), distribution(_distribution), columns(_columns) {}

  RelType relType;
  Distribution distribution;
  RowTypePtr type;
  ColumnVector columns{stl<ColumnPtr>()};

  // Correlation name for base table or derived table in  a plan. nullptr for
  // schema table.
  Name cname;
};

struct SchemaTable;
using SchemaTablePtr = SchemaTable*;

struct BaseTable : public PlanObject, public Relation {
  BaseTable() : PlanObject(PlanType::kTable) {}

  void setRelation(
      const Relation& relation,
      const ColumnVector& columns,
      const ColumnVector& schemaColumns);

  SchemaTablePtr schemaTable;
};

using BaseTablePtr = BaseTable*;

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

struct DerivedTable : public PlanObject, public Relation {
  DerivedTable() : PlanObject(PlanType::kDerivedTable) {}

  // Columns projected out. Visible in the enclosing query.
  ColumnVector columns{stl<ColumnPtr>()};

  // Exprs projected out.1:1 to 'columns'.
  ExprVector exprs{stl<ExprPtr>()};

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of Table.
  std::vector<PlanObjectPtr, StlAllocator<PlanObjectPtr>> tables{
      stl<PlanObjectPtr>()};

  std::vector<JoinPtr, StlAllocator<JoinPtr>> joins{stl<JoinPtr>()};
  ;

  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  ExprVector conjuncts{stl<ExprPtr>()};

  GroupByPtr groupBy;
  OrderByPtr orderBy;
  int32_t limit{-1};
  int32_t offset{0};
};

using DerivedTablePtr = DerivedTable*;

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
  // build side. 0 for the second use of a hash build side. 0 for table scan
  // or index access.
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
struct Index;

struct TableScan : public RelationOp {
  // Left side of index join, nullptr for a leaf table scan.
  RelationPtr input;

  // Index (or other materialization of table) used for the physical data
  // access.
  Index* index;

  // Lookup keys, empty if full table scan.
  ExprVector keys{0, stl<ExprPtr>()};

  // Leading key parts of index, 1:1 equal to 'keys'.
  ColumnVector keyColumns{stl<ColumnPtr>()};

  // Projected columns, does not necessarily include columns in keys or
  // filters.
  ColumnVector projectedColumns{stl<ColumnPtr>()};

  // Filters involving only columns of this.
  std::vector<FilteredColumnPtr> filters;
};

struct Filter : public RelationOp {
  ExprPtr expr;
};

struct Project : public RelationOp {
  // Exprs. Output description is inherited from Relation.
  ExprVector exprs{stl<ExprPtr>()};
};

struct HashJoin : public RelationOp {
  core::JoinType joinType;
  RelationOpPtr left;
  RelationOpPtr right;
  std::vector<ColumnPtr> leftKeys;
  std::vector<ColumnPtr> rightKeys;
  ExprPtr filter;
};

struct Index : public Relation {
  Index(
      Name _name,
      SchemaTablePtr _table,
      Distribution distribution,
      const ColumnVector& _columns)
      : Relation(RelType::kBase, distribution, _columns),
        name(_name),
        table(_table) {}

  Name name;
  SchemaTablePtr table;
};

using IndexPtr = Index*;

// Describes the number of rows to look at and the number of expected matches
// given an arbitrary set of values for a set of columns.
struct IndexInfo {
  // Index chosen based on columns.
  IndexPtr index;

  // True if the column combination is unique. This can be true even if there is
  // no key order in 'index'.
  bool unique;

  // The number of rows selected after index lookup based on 'lookupKeys'. For
  // empty 'lookupKeys', this is the cardinality of 'index'.
  float scanCardinality;

  // The expected number of hits for an equality match of lookup keys. This is
  // the expected number of rows given the lookup column combination regardless
  // of whether an index order can be used.
  float joinCardinality;

  // The lookup columns that match 'index'. These match 1:1 the leading keys of
  // 'index'. If 'index' has no ordering columns or if the lookup columns are
  // not a prefix of these, this is empty.
  std::vector<ColumnPtr> lookupKeys;
  PlanObjectSet coveredColumns;
};

struct SchemaTable {
  SchemaTable(Name _name, const RowTypePtr& _type) : name(_name), type(_type) {}

  void addIndex(
      Name name,
      int64_t cardinality,
      int32_t numKeysUnique,
      int32_t numOrdering,
      const ColumnVector& keys,
      DistributionType distType,
      const ColumnVector& partition,
      const ColumnVector& columns);

  ColumnPtr column(const std::string& name, Value value);

  ColumnPtr findColumn(const std::string& name) const;
  bool isUnique(folly::Range<ColumnPtr*> columns);

  IndexInfo indexInfo(IndexPtr index, folly::Range<ColumnPtr*> columns);

  IndexInfo indexByColumns(folly::Range<ColumnPtr*> columns);

  // private:
  std::vector<ColumnPtr> toColumns(const std::vector<std::string>& names);
  const std::string name;
  const RowTypePtr type;
  std::unordered_map<std::string, ColumnPtr> columns;
  std::vector<IndexPtr> indices;
};

class Schema {
 public:
  Schema(Name _name, std::vector<SchemaTablePtr> tables);

  SchemaTablePtr findTable(const std::string& name) const;

 private:
  Name name;
  std::unordered_map<std::string, SchemaTablePtr> tables_;
};

using SchemaPtr = Schema*;
/// Returns all distinct tables 'exprs' depend on.
PlanObjectSet allTables(PtrSpan<Expr> exprs);

} // namespace facebook::velox::query

namespace std {
template <>
struct hash<::facebook::velox::query::PlanObjectSet> {
  size_t operator()(const ::facebook::velox::query::PlanObjectSet set) const {
    return set.hash();
  }
};
} // namespace std
