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

#include "velox/experimental/query/PlanObject.h"

namespace facebook::verax {

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

enum class OrderType {
  kAscNullsFirst,
  kAscNullsLast,
  kDescNullsFirst,
  kDescNullsLast
};

using OrderTypeVector = std::vector<OrderType, QGAllocator<OrderType>>;

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
  Distribution(
      DistributionType type,
      float cardinality,
      ExprVector _partition,
      ExprVector _order = {},
      OrderTypeVector _orderType = {},
      int32_t uniquePrefix = 0,
      int _spacing = 0)
      : distributionType(std::move(type)),
	cardinality(cardinality),
        partition(std::move(_partition)),
        order(std::move(_order)),
        orderType(std::move(_orderType)),
        numKeysUnique(uniquePrefix),
        spacing(_spacing) {}

  static Distribution broadcast(DistributionType type, float cardinality) {
    Distribution result(type, cardinality, {});
    result.isBroadcast = true;
    return result;
  }

  bool isSamePartition(const Distribution& other) const;
  std::string toString() const;

  DistributionType distributionType;

  // Number of rows 'this' applies to. This is the size in rows if 'this' occurs in a table or index.
  float cardinality;

  // Partitioning columns. The values of these columns determine which of
  // 'numPartitions' contains any given row. This does not specify the
  // partition function (e.g. Hive bucket or range partition).
  ExprVector partition;

  // Ordering columns. Each partition is ordered by these. Specifies that
  // streaming group by or merge join are possible.
  ExprVector order;

  // Corresponds 1:1 to 'order'. The size of this gives the number of leading
  // columns of 'order' on which the data is sorted.
  OrderTypeVector orderType;

  // Number of leading elements of 'order' such that these uniquely
  // identify a row. 0 if there is no uniqueness. This can be non-0 also if data
  // is not sorted. This indicates a uniqueness for joining.
  int32_t numKeysUnique{0};

  // Specifies the selectivity between the source of the ordered data
  // and 'this'. For example, if orders join lineitem and both are
  // ordered on orderkey and there is a 1/1000 selection on orders,
  // the distribution after the filter would have a spacing of 1000,
  // meaning that lineitem is hit every 1000 orders, meaning that an
  // index join with lineitem would skip 4000 rows between hits
  // because lineitem has an average of 4 repeats of orderkey.
  float spacing{-1};

  // True if the data is replicated to 'numPartitions'.
  bool isBroadcast{false};
};

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
  ColumnVector columns;
};

struct SchemaTable;
using SchemaTablePtr = SchemaTable*;

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

using IndexPtr = Index*;

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

} // namespace facebook::verax
