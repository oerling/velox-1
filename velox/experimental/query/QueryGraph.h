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

namespace facebook::verax {

/// Base data structures for plan candidate generation.

using Name = const char*;

template <typename T>
using PtrSpan = folly::Range<T**>;

struct PlanObject;

using PlanObjectPtr = PlanObject* FOLLY_NONNULL;
using PlanObjectConstPtr = const PlanObject* FOLLY_NONNULL;

struct PlanObjectPtrHasher {
  size_t operator()(const PlanObjectPtr& object) const;
};

struct PlanObjectPtrComparer {
  bool operator()(const PlanObjectPtr& lhs, const PlanObjectPtr& rhs) const;
};

class Plan;
  
class QueryGraphContext {
 public:
  QueryGraphContext(velox::HashStringAllocator& allocator)
      : allocator_(allocator),
        stlAllocator_(velox::StlAllocator<void*>(&allocator)) {}

  Name toName(std::string_view str);

  int32_t newId(PlanObject* FOLLY_NONNULL object) {
    objects_.push_back(object);
    return objects_.size() - 1;
  }

  velox::StlAllocator<void*>* stlAllocator() & {
    return &stlAllocator_;
  }

  velox::HashStringAllocator& allocator() {
    return allocator_;
  }

  velox::HashStringAllocator& allocator_;
  velox::StlAllocator<void*> stlAllocator_;

  /// Returns a canonical instance for all logically equal values of 'object'.
  /// Returns 'object' on first call with object, thereafter the same physical
  /// object if the argument is equal.
  PlanObjectPtr dedup(PlanObjectPtr object);

  PlanObjectPtr objectAt(int32_t id) {
    return objects_[id];
  }

  /// Returns the top level plan being processed when printing operator trees. If non-null, allows showing percentages.
  Plan*& contextPlan() {
    return contextPlan_;
  }
  
  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectPtr> objects_;
  std::unordered_set<std::string_view> names_;
  std::unordered_set<PlanObjectPtr, PlanObjectPtrHasher, PlanObjectPtrComparer>
      deduppedObjects_;
  Plan* FOLLY_NULLABLE contextPlan_{nullptr};
};

inline QueryGraphContext*& queryCtx() {
  thread_local QueryGraphContext* context;
  return context;
}

template <typename T>
velox::StlAllocator<T> stl() {
  return *reinterpret_cast<velox::StlAllocator<T>*>(queryCtx()->stlAllocator());
}

#define Define(T, destination, ...)                          \
  T* destination = reinterpret_cast<T*>(                     \
      queryCtx()->allocator().allocate(sizeof(T))->begin()); \
  new (destination) T(__VA_ARGS__);

#define DefineDefault(T, destination)                        \
  T* destination = reinterpret_cast<T*>(                     \
      queryCtx()->allocator().allocate(sizeof(T))->begin()); \
  new (destination) T();

/// Converts std::string to name used in query graph objects. raw pointer to
/// arena allocated const chars.
Name toName(const std::string& string);

/// Pointers are name <type>Ptr and defined to be raw pointers. We
/// expect arena allocation with a whole areena freed after plan
/// selection.
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

/// Enum for types of query graph nodes.
enum class PlanType {
  kTable,
  kDerivedTable,
  kColumn,
  kLiteral,
  kCall,
  kAggregate,
  kProject,
  kFilter
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

  void operator delete(void* ptr) {
    LOG(FATAL) << "Plan objects are not deletable";
  }

  template <typename T>
  T as() {
    return reinterpret_cast<T>(this);
  }

  template <typename T>
  const T as() const {
    return reinterpret_cast<const T>(this);
  }

  virtual PtrSpan<PlanObject> children() const {
    return PtrSpan<PlanObject>(nullptr, nullptr);
  }

  template <typename Func>
  void preorderVisit(Func func) {
    func(this);
    for (auto child : children()) {
      child->preorderVisit(func);
    }
  }

  virtual bool isExpr() const {
    return false;
  }

  size_t hash() const;

  virtual std::string toString() const {
    return fmt::format("#{}", id);
  }
  PlanType type;
  int32_t id;
};

struct Expr;
using ExprPtr = Expr*;
struct Column;
using ColumnPtr = Column*;
using ExprVector = std::vector<ExprPtr, velox::StlAllocator<ExprPtr>>;

class PlanObjectSet {
 public:
  bool contains(PlanObjectConstPtr object) const {
    return object->id < bits_.size() * 64 &&
        velox::bits::isBitSet(bits_.data(), object->id);
  }

  bool operator==(const PlanObjectSet& other) const;

  size_t hash() const;

  void add(PlanObjectPtr ptr) {
    auto id = ptr->id;
    ensureSize(id);
    velox::bits::setBit(bits_.data(), id);
  }

  /// Returns true if 'this' is a subset of 'super'.
  bool isSubset(const PlanObjectSet& super) const;

  void erase(PlanObjectPtr object) {
    if (object->id < bits_.size() * 64) {
      velox::bits::clearBit(bits_.data(), object->id);
    }
  }

  void unionColumns(ExprPtr expr);

  void unionColumns(const ExprVector& exprs);

  void unionSet(const PlanObjectSet& other);

  void intersect(const PlanObjectSet& other);

  template <typename V>
  void unionObjects(const V& objects) {
    for (PlanObjectPtr& object : objects) {
      add(object);
    }
  }

  template <typename Func>
  void forEach(Func func) const {
    auto ctx = queryCtx();
    velox::bits::forEachSetBit(bits_.data(), 0, bits_.size() * 64, [&](auto i) {
      func(ctx->objectAt(i));
    });
  }

  template <typename T = PlanObjectPtr>
  std::vector<T> objects() const {
    std::vector<T> result;
    forEach(
        [&](auto object) { result.push_back(reinterpret_cast<T>(object)); });
    return result;
  }

  std::string toString(bool names) const;

 private:
  void ensureSize(int32_t id) {
    ensureWords(velox::bits::nwords(id + 1));
  }

  void ensureWords(int32_t size) {
    if (bits_.size() < size) {
      bits_.resize(size);
    }
  }

  std::vector<uint64_t, velox::StlAllocator<uint64_t>> bits_{stl<uint64_t>()};
};

struct Value {
  Value() = default;
  Value(const velox::Type* _type, float _cardinality)
      : type(_type), cardinality(_cardinality) {}

  float byteSize() const;

  const velox::Type* FOLLY_NONNULL type;
  velox::variant min;
  velox::variant max;
  const float cardinality;
  // 0 means no nulls, 0.5 means half are null.
  float nullFraction{0};
  bool nullable{true};
};

struct Relation;
using RelationPtr = Relation*;

struct Expr : public PlanObject {
  Expr(PlanType type, Value _value) : PlanObject(type), value(_value) {}

  bool isExpr() const override {
    return true;
  }

  // Returns the single base or derived table 'this' depends on, nullptr if
  // 'this' depends on none or multiple tables.
  PlanObjectPtr singleTable();

  PlanObjectSet allTables() const;

  PlanObjectSet equivTables() const;

  /// True if '&other == this' or is recursively equal with column
  /// leaves either same or in same equivalence.
  bool sameOrEqual(const Expr& other) const;

  PlanObjectSet columns;
  Value value;
};

/// If 'object' is an Expr, returns Expr::singleTable, else nullptr.
PlanObjectPtr singleTable(PlanObjectPtr object);

struct Equivalence;
using EquivalencePtr = Equivalence*;

struct Literal : public Expr {
  Literal(Value value, velox::variant _literal)
      : Expr(PlanType::kLiteral, value), literal(_literal) {}
  velox::variant literal;
};

struct Column : public Expr {
  Column(Name _name, PlanObjectPtr _relation, Value value);
  void equals(ColumnPtr other);

  Name name;
  PlanObjectPtr relation;
  EquivalencePtr equivalence{nullptr};

  // If this is a column of a BaseTable, points to the corresponding
  // column in the SchemaTable. Used for matching with
  // ordering/partitioning columns in the SchemaTable.
  Column* schemaColumn{nullptr};

  std::string toString() const override;
};

template <typename T>
inline folly::Range<T*> toRange(
    const std::vector<T, velox::StlAllocator<T>>& v) {
  return folly::Range<T*>(const_cast<T*>(v.data()), v.size());
}

template <typename T, typename U>
inline folly::Range<T*> toRangeCast(
    const std::vector<U, velox::StlAllocator<U>>& v) {
  return folly::Range<T*>(
      reinterpret_cast<T*>(const_cast<U*>(v.data())), v.size());
}

using ColumnVector = std::vector<ColumnPtr, velox::StlAllocator<ColumnPtr>>;

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
  Call(
      PlanType _type,
      Name _func,
      Value value,
      ExprVector _args,
      FunctionSet _functions)
      : Expr(_type, value),
        func(_func),
        args(std::move(_args)),
        functions(_functions) {
    for (auto arg : args) {
      columns.unionSet(arg->columns);
    }
  }

  Call(Name _func, Value value, ExprVector _args, FunctionSet _functions)
      : Call(PlanType::kCall, _func, value, _args, _functions) {}

  Name func;
  ExprVector args;

  // Set of functions used in 'this' and 'args'.
  FunctionSet functions;

  PtrSpan<PlanObject> children() const override {
    return toRangeCast<PlanObjectPtr>(args);
  }

  std::string toString() const override;
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

/// Represents a system that contains or produces data. nullptr means
/// a shared access distributed file system or data lake for schema
/// objects. nullptr means the system running the top level query for
/// a RelationOp. For cases of federation where data is only
/// accessible via a specific instance of a specific type of system,
/// the locus represents the instance and the subclass of Locus
/// represents the type of system for a schema object. For a
/// RelationOp, a non-null locus means that the op is pushed down into
/// the corresponding system. Distributions can be copartitioned only
/// if their locus is pointer equal to the other locus.
struct Locus {};

using LocusPtr = Locus*;

/// Method for determining a partition given an ordered list of partitioning
/// keys. Hive hash is an example, range partitioning is another. Add values
/// here for more types.
enum class ShuffleMode { kNone, kHive };

/// Distribution of data. 'numPartitions' is 1 if the data is not partitioned.
/// There is copartitioning if the DistributionType is the same on both sides
/// and both sides have an equal number of 1:1 type matched partitioning keys.
struct DistributionType {
  bool operator==(const DistributionType& other) const {
    return mode == other.mode && numPartitions == other.numPartitions &&
        locus == other.locus;
  }

  ShuffleMode mode{ShuffleMode::kNone};
  int32_t numPartitions{1};
  LocusPtr locus{nullptr};
};

// Describes output of relational operator. If base table, cardinality is
// after filtering, column value ranges are after filtering.
struct Distribution {
  Distribution() = default;
  Distribution(DistributionType type, ExprVector _partition)
      : distributionType(std::move(type)), partition(std::move(_partition)) {}

  float cardinality;

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
  float spacing{-1};

  // specifies that data in each partition is clustered by these columns, i.e.
  // all rows with any value k1=v1,...kn =vn are consecutive. Specifies that
  // once the value of any clustering column changes between consecutive rows,
  // the same combination of clustering columns will not repeat. means that a
  // final group by can be flushed when seeing a change in clustering columns.
  ColumnVector clustering{stl<ColumnPtr>()};

  // True if the data is replicated to 'numPartitions'.
  bool isBroadcast{false};

  bool isSamePartition(const Distribution& other) const;
  std::string toString() const;
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
  Expr* filter;

  // True if an unprobed right side row produces a result with right side
  // columns set and left side columns as null. Possible only be hash or
  // merge.
  bool leftOptional{false};

  // True if a right side miss produces a row with left side columns
  // and a null for right side columns (left outer join). A full outer
  // join has both left and right optional.
  bool rightOptional{false};

  // True if the right side is only checked for existence of a match. If
  // rightOptional is set, this can project out a null for misses.
  bool rightExists{false};

  // True if the join is a right semijoin. This is possible only by hash or
  // merge.
  bool leftExists{false};

  // True if produces a result for left if no match on the right.
  bool rightNotExists{false};

  void guessFanout();
};

using JoinPtr = Join*;

using JoinVector = std::vector<JoinPtr, velox::StlAllocator<JoinPtr>>;

/// Identifies a base table or the operator type producing the relation. Base
/// data as in Index has type kBase. The result of a table scan is kTableScan.
enum class RelType {
  kBase,
  kTableScan,
  kRepartition,
  kFilter,
  kProject,
  kJoin,
  kHashBuild,
  kAggregation,
  kOrderBy
};

struct Relation {
  Relation() = default;

  Relation(RelType type) : relType(type) {}

  Relation(
      RelType _relType,
      Distribution _distribution,
      const ColumnVector& _columns)
      : relType(_relType),
        distribution(std::move(_distribution)),
        columns(_columns) {}

  RelType relType;
  Distribution distribution;
  velox::RowTypePtr type;
  ColumnVector columns{stl<ColumnPtr>()};
};

struct SchemaTable;
using SchemaTablePtr = SchemaTable*;

struct BaseTable : public PlanObject {
  BaseTable() : PlanObject(PlanType::kTable) {}

  Name cname{nullptr};

  SchemaTablePtr schemaTable;

  ColumnVector columns{stl<ColumnPtr>()};
  ColumnVector schemaColumns{stl<ColumnPtr>()};

  JoinVector joinedBy{stl<JoinPtr>()};

  // Top level conjuncts on single columns and literals, column to the left.
  ExprVector columnFilters{stl<ExprPtr>()};

  // Multicolumn filters dependent on 'this' alone.
  ExprPtr filter{nullptr};

  // the fraction of base table rows selected by all filters involving this
  // table only.
  float filterSelectivity{1};

  // System specific representation of filter on columns, e.g. set of
  // common::Filter.
  void* nativeFilter;

  // Columns referenced from 'this' that do not participate in filters, joins,
  // grouping or ordering.
  PlanObjectSet payload;

  std::string toString() const override;
};

using BaseTablePtr = BaseTable*;

// Aggregate function. The aggregation and arguments are in the
// inherited Call. The Value pertains to the aggregation
// result or accumulator.
struct Aggregate : public Call {
  Aggregate(
      Name _func,
      Value value,
      ExprVector _args,
      FunctionSet _functions,
      bool _isDistinct,
      ExprPtr _condition,
      bool _isAccumulator)
      : Call(PlanType::kAggregate, _func, value, std::move(_args), _functions),
        isDistinct(_isDistinct),
        condition(_condition),
        isAccumulator(_isAccumulator) {
    if (condition) {
      columns.unionSet(condition->columns);
    }
  }

  bool isDistinct;
  ExprPtr condition;
  bool isAccumulator;
};

using AggregatePtr = Aggregate*;

struct Aggregation;
using AggregationPtr = Aggregation*;

struct OrderBy : public Relation {
  std::vector<ExprPtr> keys;

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to payload.
  PlanObjectSet dependentKeys;
};

using OrderByPtr = OrderBy*;

struct DerivedTable : public PlanObject {
  DerivedTable() : PlanObject(PlanType::kDerivedTable) {}

  Name cname{nullptr};

  // Columns projected out. Visible in the enclosing query.
  ColumnVector columns{stl<ColumnPtr>()};

  // Exprs projected out.1:1 to 'columns'.
  ExprVector exprs{stl<ExprPtr>()};

  JoinVector joinedBy{stl<JoinPtr>()};

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of BaseTable.
  std::vector<PlanObjectPtr, velox::StlAllocator<PlanObjectPtr>> tables{
      stl<PlanObjectPtr>()};

  std::vector<JoinPtr, velox::StlAllocator<JoinPtr>> joins{stl<JoinPtr>()};

  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  ExprVector conjuncts{stl<ExprPtr>()};

  AggregationPtr aggregation{nullptr};
  ExprPtr having{nullptr};
  OrderByPtr orderBy{nullptr};
  int32_t limit{-1};
  int32_t offset{0};

  // after 'joins' is filled in, links tables to their direct and
  // equivalence-implied joins.
  void expandJoins();

  /// Initializes 'this' to join 'tables' from 'super'. Adds the joins from
  /// 'existences' as semijoins to limit cardinality when making a hash join
  /// build side. Allows importing a reducing join from probe to build.
  void import(
      const DerivedTable& super,
      const PlanObjectSet& tables,
      const std::vector<PlanObjectSet>& existences);

  bool hasTable(PlanObjectPtr table) {
    return std::find(tables.begin(), tables.end(), table) != tables.end();
  }
};

using DerivedTablePtr = DerivedTable*;
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
struct RelationOp : public Relation {
  RelationOp(
      RelType type,
      boost::intrusive_ptr<RelationOp> input,
      Distribution _distribution)
      : Relation(relType, _distribution, ColumnVector{stl<ColumnPtr>()}),
        input(std::move(input)) {}

  virtual ~RelationOp() = default;

  void operator delete(void* ptr) {
    queryCtx()->allocator().free(velox::HashStringAllocator::headerOf(ptr));
  }

  // thread local reference count. PlanObjects are freed when the
  // QueryGraphContext arena is freed, candidate plans are freed when no longer
  // referenced.
  int32_t refCount{0};

  // Input of filter/project/group by etc., Left side of join, nullptr for a
  // leaf table scan.
  boost::intrusive_ptr<struct RelationOp> input;

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
  // by/distinct / repartition. The memory footprint may not be this if the operation is streaming or spills.
  float totalBytes{0};

  // Maximum memory occupancy. If the operation is blocking, e.g. group by, the amount of spill is 'totalBytes' - 'peakResidentBytes'.
  float peakResidentBytes{0};

  virtual void setCost(const PlanState& input);

  virtual std::string toString(bool recursive, bool detail) const {
    if (input && recursive) {
      return input->toString(true, detail);
    } 
   return "";
  }

  // adds a line of cost information to 'out'
  void printCost(bool detail, std::stringstream& out) const;
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
      IndexPtr _index)
      : RelationOp(RelType::kTableScan, input, _distribution),
        baseTable(table),
        index(_index) {}

  // The base table reference. May occur in multiple scans if the base
  // table decomposes into access via secondary index joined to pk or
  // if doing another pass for late materialization.
  BaseTablePtr baseTable;

  // Index (or other materialization of table) used for the physical data
  // access.
  IndexPtr index;

  // Lookup keys, empty if full table scan.
  ExprVector keys{stl<ExprPtr>()};

  // Columns read from 'baseTable'. Can be more than 'columns' if
  // there are filters that need columns that are not projected out to
  // next op.
  PlanObjectSet extractedColumns;

  // If this is a lookup, 'joinType' can  be inner, left or anti.
  velox::core::JoinType joinType{velox::core::JoinType::kInner};

  ExprPtr joinFilter{nullptr};

  /// Columns of base table available in 'index'.
  PlanObjectSet availableColumns();

  void setRelation(
      const ColumnVector& columns,
      const ColumnVector& schemaColumns);

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
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
  ExprVector exprs{stl<ExprPtr>()};
};

enum class JoinMethod { kHash, kMerge };

struct JoinOp : public RelationOp {
  JoinOp(
      JoinMethod _method,
      velox::core::JoinType _joinType,
      RelationOpPtr input,
      RelationOpPtr right,
      ColumnVector _columns)
      : RelationOp(RelType::kJoin, input, input->distribution),
        method(_method),
        joinType(_joinType),
        right(std::move(right)) {
    columns = std::move(_columns);
  }

  JoinMethod method;
  velox::core::JoinType joinType;
  RelationOpPtr right;
  ExprVector leftKeys{stl<ExprPtr>()};
  ExprVector rightKeys{stl<ExprPtr>()};
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
  ExprVector keys{stl<ExprPtr>()};

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

  ExprVector grouping{stl<ExprPtr>()};

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to any() aggregates.
  PlanObjectSet dependentKeys;

  std::vector<AggregatePtr, velox::StlAllocator<AggregatePtr>> aggregates{
      stl<AggregatePtr>()};

  velox::core::AggregationNode::Step step{
      velox::core::AggregationNode::Step::kSingle};

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
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

  /// Returns cost of next lookup when the hit is within 'range' rows
  /// of the previous hit. If lookups are not batched or not ordered,
  /// then 'range' should be the cardinality of the index.
  float lookupCost(float range);
};

// Describes the number of rows to look at and the number of expected matches
// given an arbitrary set of values for a set of columns.
struct IndexInfo {
  // Index chosen based on columns.
  IndexPtr index;

  // True if the column combination is unique. This can be true even if there is
  // no key order in 'index'.
  bool unique{false};

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

  /// Returns the schema column for the BaseTable column 'column' or nullptr if
  /// not in the index.
  ColumnPtr schemaColumn(ColumnPtr keyValue) const;
};

struct SchemaTable {
  SchemaTable(Name _name, const velox::RowTypePtr& _type)
      : name(_name), type(_type) {}

  void addIndex(
      Name name,
      float cardinality,
      int32_t numKeysUnique,
      int32_t numOrdering,
      const ColumnVector& keys,
      DistributionType distType,
      const ColumnVector& partition,
      const ColumnVector& columns);

  ColumnPtr column(const std::string& name, Value value);

  ColumnPtr findColumn(const std::string& name) const;
  bool isUnique(PtrSpan<Column> columns);

  IndexInfo indexInfo(IndexPtr index, PtrSpan<Column> columns);

  IndexInfo indexByColumns(PtrSpan<Column> columns);

  // private:
  std::vector<ColumnPtr> toColumns(const std::vector<std::string>& names);
  const std::string name;
  const velox::RowTypePtr type;
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

} // namespace facebook::verax

namespace std {
template <>
struct hash<::facebook::verax::PlanObjectSet> {
  size_t operator()(const ::facebook::verax::PlanObjectSet& set) const {
    return set.hash();
  }
};
} // namespace std
