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

#include "velox/experimental/query/Schema.h"

#include "velox/core/PlanNode.h"

/// Defines subclasses of PlanObject for describing the logical
/// structure of queries. These are the constraints that guide
/// generation of plan candidates. These are referenced from
/// candidates but stay immutable acrosss the candidate
/// generation. Sometimes new derived tables may be added for
/// representing constraints on partial plans but otherwise these stay
/// constant.
namespace facebook::verax {

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

/// Superclass for all expressions.
struct Expr : public PlanObject {
  Expr(PlanType type, const Value& _value) : PlanObject(type), value(_value) {}

  bool isExpr() const override {
    return true;
  }

  // Returns the single base or derived table 'this' depends on, nullptr if
  // 'this' depends on none or multiple tables.
  PlanObjectConstPtr singleTable() const;

  /// Returns all tables 'this' depends on.
  PlanObjectSet allTables() const;

  /// True if '&other == this' or is recursively equal with column
  /// leaves either same or in same equivalence.
  bool sameOrEqual(const Expr& other) const;

  // The columns this depends on.
  PlanObjectSet columns;

  // Type Constraints on the value of 'this'.
  Value value;
};

/// If 'object' is an Expr, returns Expr::singleTable, else nullptr.
PlanObjectConstPtr singleTable(PlanObjectConstPtr object);

struct Equivalence;
using EquivalencePtr = Equivalence*;

/// Represents a literal.
struct Literal : public Expr {
  Literal(const Value& value, velox::variant _literal)
      : Expr(PlanType::kLiteral, value), literal(_literal) {}
  velox::variant literal;
};

/// Represents a column. A column is always defined by a relation, whether table
/// or derived table.
struct Column : public Expr {
  Column(Name _name, PlanObjectPtr _relation, const Value& value);

  /// Asserts that 'this' and 'other' are joined on equality. This has a
  /// transitive effect, so if a and b are previously asserted equal and c is
  /// asserted equal to b, a and c are also equal.
  void equals(ColumnPtr other) const;

  Name name;
  PlanObjectPtr relation;

  // Equivalence class. Lists all columns directly or indirectly asserted equal
  // to 'this'.
  mutable EquivalencePtr equivalence{nullptr};

  // If this is a column of a BaseTable, points to the corresponding
  // column in the SchemaTable. Used for matching with
  // ordering/partitioning columns in the SchemaTable.
  ColumnPtr schemaColumn{nullptr};

  std::string toString() const override;
};

template <typename T>
inline folly::Range<T*> toRange(const std::vector<T, QGAllocator<T>>& v) {
  return folly::Range<T const*>(v.data(), v.size());
}

template <typename T, typename U>
inline PtrSpan<T> toRangeCast(U v) {
  return PtrSpan<T>(reinterpret_cast<const T* const*>(v.data()), v.size());
}

/// A bit set that qualifies a function call. Represents which functions/kinds
/// of functions are found inside the children of a function call.
class FunctionSet {
 public:
  FunctionSet() : set_(0) {}
  FunctionSet(uint32_t set) : set_(set) {}

  /// True if 'item' is in 'this'.
  bool contains(int32_t item) {
    return 0 != (set_ & (1UL << item));
  }

  /// Unions 'this' and 'other' and returns the result.
  FunctionSet operator|(const FunctionSet& other) const {
    return FunctionSet(set_ | other.set_);
  }

 private:
  uint64_t set_;
};

/// Represents a function call or a special form, any expression with
/// subexpressions.
struct Call : public Expr {
  Call(
      PlanType type,
      Name _func,
      const Value& value,
      ExprVector _args,
      FunctionSet functions)
      : Expr(type, value),
        func(_func),
        args(std::move(_args)),
        functions(functions) {
    for (auto arg : args) {
      columns.unionSet(arg->columns);
    }
  }

  Call(Name func, Value value, ExprVector args, FunctionSet functions)
      : Call(PlanType::kCall, func, value, args, functions) {}

  // name of function.
  Name func;

  // Arguments.
  ExprVector args;

  // Set of functions used in 'this' and 'args'.
  FunctionSet functions;

  PtrSpan<PlanObject> children() const override {
    return folly::Range<const PlanObject* const*>(
        reinterpret_cast<const PlanObject* const*>(args.data()), args.size());
  }

  std::string toString() const override;
};

using CallPtr = const Call*;

/// Represens a set of transitively equal columns.
struct Equivalence {
  // Each element has a direct or implied equality edge to every other.
  ColumnVector columns;
};

/// Represents one side of a join. See Join below for the meaning of the
/// members.
struct JoinSide {
  PlanObjectConstPtr table;
  const ExprVector& keys;
  const bool isOptional;
  const bool isExists;
  const bool isNotExists;

  /// Returns the join type to use if 'this' is the right side.
  velox::core::JoinType leftJoinType() const {
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

/// Represents a possibly directional equality join edge.
/// 'rightTable' is always set. 'leftTable' is nullptr if 'leftKeys' come from
/// different tables. If so, 'this' must be non-inner and not full outer.
struct JoinEdge {
  // Leading left side join keys.
  ExprVector leftKeys;
  // Leading right side join keys, compared equals to 1:1 to 'leftKeys'.
  ExprVector rightKeys;

  PlanObjectConstPtr leftTable{nullptr};
  PlanObjectConstPtr rightTable{nullptr};

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

  // True if produces a result for left if no match on the right.
  bool rightNotExists{false};

  //// Fills in 'lrFanout' and 'rlFanout', 'leftUnique', 'rightUnique'.
  void guessFanout();

  /// True if inner join.
  bool isInner() const {
    return !leftOptional && !rightOptional && !rightExists && !rightNotExists;
  }

  // True if all tables referenced from 'leftKeys' must be placed before placing
  // this.
  bool isNonCommutative() const {
    // Inner and full outer joins are commutative.
    return !leftTable || (rightOptional && !leftOptional) || rightExists ||
        rightNotExists;
  }

  // Returns the join side info for 'table'. If 'other' is set, returns the
  // other side.
  const JoinSide sideOf(PlanObjectConstPtr side, bool other = false) const;

  std::string toString() const;
};

using JoinEdgePtr = JoinEdge*;

using JoinEdgeVector = std::vector<JoinEdgePtr, QGAllocator<JoinEdgePtr>>;

/// Represents a reference to a table from a query. The There is one of these
/// for each occurrence of the schema table. A TableScan references one
/// baseTable but the same BaseTable can be referenced from many TableScans, for
/// example if accessing different indices in a secondary to primary key lookup.
struct BaseTable : public PlanObject {
  BaseTable() : PlanObject(PlanType::kTable) {}

  // Correlation name, distinguishes between uses of the same schema table.
  Name cname{nullptr};

  SchemaTablePtr schemaTable;

  /// All columns referenced from 'schemaTable' under this correlation name.
  /// Different indices may have to be combined in different TableScans to cover
  /// 'columns'.
  ColumnVector columns;

  // All joins where 'this' is an end point.
  JoinEdgeVector joinedBy;

  // Top level conjuncts on single columns and literals, column to the left.
  ExprVector columnFilters;

  // Multicolumn filters dependent on 'this' alone.
  ExprVector filter;

  // the fraction of base table rows selected by all filters involving this
  // table only.
  float filterSelectivity{1};

  // System specific representation of filter on columns, e.g. set of
  // common::Filter.
  void* nativeFilter;

  std::string toString() const override;
};

using BaseTablePtr = const BaseTable*;

// Aggregate function. The aggregation and arguments are in the
// inherited Call. The Value pertains to the aggregation
// result or accumulator.
struct Aggregate : public Call {
  Aggregate(
      Name func,
      const Value& value,
      ExprVector args,
      FunctionSet functions,
      bool isDistinct,
      ExprPtr condition,
      bool isAccumulator)
      : Call(PlanType::kAggregate, func, value, std::move(args), functions),
        isDistinct(isDistinct),
        condition(condition),
        isAccumulator(isAccumulator) {
    if (condition) {
      columns.unionSet(condition->columns);
    }
  }

  bool isDistinct;
  ExprPtr condition;
  bool isAccumulator;
};

using AggregatePtr = const Aggregate*;

struct Aggregation;
using AggregationPtr = Aggregation*;

/// Represents an order by for a derived table.
struct OrderBy : public Relation {
  ExprVector keys;
  OrderTypeVector orderTypes;

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to payload.
  PlanObjectSet dependentKeys;
};

using OrderByPtr = OrderBy*;

/// Represents a derived table, i.e. a select in a from clause. This is the
/// basic unit of planning. Derived tables can be merged and split apart from
/// other ones. Join types and orders are decided within each derived table. A
/// derived table is likewise a reorderable unit inside its parent derived
/// table. Joins can move between derived tables within limits, considering the
/// semantics of e.g. group by.
struct DerivedTable : public PlanObject {
  DerivedTable() : PlanObject(PlanType::kDerivedTable) {}

  // Correlation name.
  Name cname{nullptr};

  // Columns projected out. Visible in the enclosing query.
  ColumnVector columns;

  // Exprs projected out.1:1 to 'columns'.
  ExprVector exprs;

  // References all joins where 'this' is an end point.
  JoinEdgeVector joinedBy;

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of BaseTable.
  std::vector<PlanObjectConstPtr, QGAllocator<PlanObjectConstPtr>> tables;

  // Repeats the contents of 'tables'. Used for membership check. A DerivedTable
  // can be a subset of another, for example when planning a join for a build
  // side. In this case joins that refer to tables not in 'tableSet' are not
  // considered.
  PlanObjectSet tableSet;

  // Tables that are not to the right sides of non-commutative joins.
  PlanObjectSet startTables;

  // Joins between 'tables'.
  JoinEdgeVector joins;

  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  ExprVector conjuncts;

  // Set of reducing joined tables imported to reduce build size. Set if 'this'
  // represents a build side join.
  PlanObjectSet importedExistences;

  // Postprocessing clauses, group by, having, order by, limit, offset.
  AggregationPtr aggregation{nullptr};
  ExprPtr having{nullptr};
  OrderByPtr orderBy{nullptr};
  int32_t limit{-1};
  int32_t offset{0};

  // Guess of cardinality. The actual cardinality is calculated with a plan but.
  // This is only for deciding in which order to cost candidates.
  float baseCardinality{1};

  /// Adds an equijoin edge between 'left' and 'right'. The flags correspond to
  /// the like-named members in Join.
  void addJoinEquality(
      ExprPtr left,
      ExprPtr right,
      bool leftOptional,
      bool rightOptional,
      bool rightExists,
      bool rightNotExists);

  // after 'joins' is filled in, links tables to their direct and
  // equivalence-implied joins.
  void linkTablesToJoins();

  /// Completes 'joins' with edges implied by column equivalences.
  void addImpliedJoins();

  /// Initializes 'this' to join 'tables' from 'super'. Adds the joins from
  /// 'existences' as semijoins to limit cardinality when making a hash join
  /// build side. Allows importing a reducing join from probe to build.
  /// 'firstTable' is the joined table that is restricted by the other tables in
  /// 'tables' and 'existences'.
  void import(
      const DerivedTable& super,
      PlanObjectConstPtr firstTable,
      const PlanObjectSet& tables,
      const std::vector<PlanObjectSet>& existences);

  //// True if 'table' is of 'this'.
  bool hasTable(PlanObjectConstPtr table) {
    return std::find(tables.begin(), tables.end(), table) != tables.end();
  }

 private:
  void setStartTables();
  void guessBaseCardinality();
};

using DerivedTablePtr = DerivedTable*;

/// Returns all distinct tables 'exprs' depend on.
PlanObjectSet allTables(PtrSpan<Expr> exprs);

} // namespace facebook::verax
