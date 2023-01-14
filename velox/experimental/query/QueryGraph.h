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

namespace facebook::verax {

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
inline folly::Range<T*> toRange(const std::vector<T, QGAllocator<T>>& v) {
  return folly::Range<T*>(const_cast<T*>(v.data()), v.size());
}

template <typename T, typename U>
inline folly::Range<T*> toRangeCast(const std::vector<U, QGAllocator<U>>& v) {
  return folly::Range<T*>(
      reinterpret_cast<T*>(const_cast<U*>(v.data())), v.size());
}

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
  ColumnVector columns;
  ;
  // Corresponds pairwise to 'exprs'. True if the Expr comes from an
  // outer optional side key join and is therefore null or equal.
  std::vector<bool> nullable;
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
  ExprVector leftKeys;
  // Leading right side join keys, compared equals to 1:1 to 'leftKeys'.
  ExprVector rightKeys;

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

using JoinVector = std::vector<JoinPtr, QGAllocator<JoinPtr>>;

struct BaseTable : public PlanObject {
  BaseTable() : PlanObject(PlanType::kTable) {}

  Name cname{nullptr};

  SchemaTablePtr schemaTable;

  ColumnVector columns;
  ColumnVector schemaColumns;

  JoinVector joinedBy;

  // Top level conjuncts on single columns and literals, column to the left.
  ExprVector columnFilters;

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
  ColumnVector columns;

  // Exprs projected out.1:1 to 'columns'.
  ExprVector exprs;

  JoinVector joinedBy;

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of BaseTable.
  std::vector<PlanObjectPtr, QGAllocator<PlanObjectPtr>> tables;

  JoinVector joins;

  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  ExprVector conjuncts;

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
