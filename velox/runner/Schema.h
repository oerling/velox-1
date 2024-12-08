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

#include "velox/common/base/Fs.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/connectors/Connector.h"

/// Base classes for schema elements used in execution. A Schema is a collection
/// of Tables. A Table is a collection of Columns. Tables and Columns have
/// statistics and Tables can be sampled. Derived classes connect to different
/// metadata stores and provide different metadata, e.g. order, partitioning,
/// bucketing etc.
namespace facebook::velox::runner {

/// Represents statistics of a column. The statistics may represent the column
/// across the table or may be calculated over a sample of a layout of the
/// table. All fields are optional.
struct ColumnStatistics {
  /// Empty for top level  column. Struct member name or string of key for
  /// struct  or flat map subfield.
  std::string name;

  /// If true, the column cannot have nulls.
  bool nonNull{false};

  /// Observed percentage of nulls. 0 does not mean that there are no nulls.
  float nullPct{0};

  /// Minimum observed value for comparable scalar columns.
  std::optional<variant> min;

  /// Maximum observed value for a comparable scalar.
  std::optional<variant> max;

  /// For string, varbinary, array and map, the maximum observed number of
  /// characters/bytes/elements/key-value pairs.
  std::optional<int32_t> maxLength;

  /// Average count of characters/bytes/elements/key-value pairs.
  std::optional<int32_t> avgLength;

  /// Estimated number of distinct values. Not specified for complex types.
  std::optional<int64_t> numDistinct;

  /// For complex type columns, statistics of children. For array, contains one
  /// element describing the array elements. For struct, has one element for
  /// each member. For map, has an element for keys and one for values. For flat
  /// map, may have one element for each key. In all cases, stats may be
  /// missing.
  std::vector<ColumnStatistics> children;
};

/// Base class for column. The column's name and type are immutable but the
/// stats may be set multiple times.
class Column {
 public:
  virtual ~Column() = default;

  Column(const std::string& name, TypePtr type) : name_(name), type_(type) {}

  const ColumnStatistics* stats() const {
    return latestStats_;
  }

  ColumnStatistics* mutableStats() const {
    return latestStats_;
  }

  /// Sets statistics. May be called multipl times if table contents change.
  void setStats(std::unique_ptr<ColumnStatistics> stats) {
    std::lock_guard<std::mutex> l(mutex_);
    allStats_.push_back(std::move(stats));
    latestStats_ = allStats_.back().get();
  }

  const std::string& name() const {
    return name_;
  }

  const TypePtr& type() const {
    return type_;
  }

  int64_t approxNumDistinct(int64_t deflt = 1000) const {
    auto* s = stats();
    return s && s->numDistinct.has_value() ? s->numDistinct.value() : deflt;
  }

 protected:
  const std::string name_;
  const TypePtr type_;

  // The latest element added to 'allStats_'.
  tsan_atomic<ColumnStatistics*> latestStats_{nullptr};

  // All statistics recorded for this column. Old values can be purged when the
  // containing Schema is not in use.
  std::vector<std::unique_ptr<ColumnStatistics>> allStats_;

 private:
  // Serializes changes to statistics.
  std::mutex mutex_;
};

class Table;

/// Represents a physical manifestation of a table. There is at least
/// one layout but for tables that have multiple sort orders,
/// partitionings, indices, column groups, etc, there is a separate
/// layout for each. Scans, index lookups, stats, sampling and split
/// enumeration are done with respect to table layout.
class TableLayout {
 public:
  TableLayout(
      const std::string& name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitionColumns,
      std::vector<const Column*> orderColumns,
      std::vector<core::SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      bool supportsScan)
      : name_(name),
        table_(table),
        connector_(connector),
        columns_(std::move(columns)),
        partitionColumns_(std::move(partitionColumns)),
        orderColumns_(std::move(orderColumns)),
        sortOrder_(std::move(sortOrder)),
        lookupKeys_(lookupKeys),
        supportsScan_(supportsScan) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (auto& column : columns_) {
      names.push_back(column->name());
      types.push_back(column->type());
    }
    rowType_ = ROW(std::move(names), std::move(types));
  }

  virtual ~TableLayout() = default;

  /// Name for documentation. If there are multiple layouts, this is unique
  /// within the table.
  const std::string name() const {
    return name_;
  }

  const Table* table() const {
    return table_;
  }

  /// Returns a connector specific table and layout information that
  /// encapsulates details like synthetic columns that need to be known when
  /// making column and table handles.
  virtual std::unique_ptr<connector::LayoutMetadata> metadata() const = 0;

  /// List of columns present in this layout.
  const std::vector<const Column*>& columns() const;

  /// Set of partitioning columns. The values in partitioning columns determine
  /// the location of the row. Joins on equality of partitioning columns are
  /// co-located.
  const std::vector<const Column*>& partitioningColumns() const {
    return partitionColumns_;
  }

  /// Columns on which content is ordered within the range of rows covered by a
  /// Split.
  const std::vector<const Column*>& orderColumns() const {
    return orderColumns_;
  }

  const std::vector<core::SortOrder>& sortOrder() const {
    return sortOrder_;
  }

  /// Returns the key columns usable for index lookup. This is modeled
  /// separately from sortedness since some sorted files may not
  /// support lookup. An index lookup has 0 or more equalities
  /// followed by up to one range. The equalities need to be on
  /// contiguous, leading parts of the column list and the range must
  /// be on the next. This coresponds to a multipart key.
  const std::vector<const Column*>& lookupKeys() const {
    return lookupKeys_;
  }

  /// True if a full table scan is supported. Some lookup sources prohibit this.
  /// At the same time the dataset may be available in a scannable form in
  /// another layout.
  bool supportsScan() const {
    return supportsScan_;
  }

  /// Returns the Connector to use for generating ColumnHandles and TableHandles
  /// for operations against this layout.
  connector::Connector* connector() const {
    return connector_;
  }

  const RowTypePtr& rowType() const {
    return rowType_;
  }

  /// Samples 'pct' percent of rows. Applies filters in 'handle'
  /// before sampling. Returns {count of sampled, count matching
  /// filters}. 'extraFilters' is a list of conjuncts to evaluate in
  /// addition to the filters in 'handle'.  If 'statistics' is
  /// non-nullptr, fills it with post-filter statistics for the
  /// subfields in 'fields'. When sampling on demand, it is usually sufficient
  /// to look at a subset of all accessed columns, so we specify these instead
  /// of defaulting to the columns in 'handle'.  'allocator' is used for
  /// temporary memory in gathering statistics.
  virtual std::pair<int64_t, int64_t> sample(
      const connector::ConnectorTableHandlePtr& handle,
      float pct,
      std::vector<core::TypedExprPtr> extraFilters,
      const std::vector<common::Subfield>& fields = {},
      HashStringAllocator* allocator = nullptr,
      std::vector<ColumnStatistics>* statistics = nullptr) {
    VELOX_UNSUPPORTED("Table class does not support sampling.");
  }

 private:
  const std::string name_;
  const Table* table_;
  connector::Connector* connector_;
  std::vector<const Column*> columns_;
  const std::vector<const Column*> partitionColumns_;
  const std::vector<const Column*> orderColumns_;
  const std::vector<core::SortOrder> sortOrder_;
  const std::vector<const Column*> lookupKeys_;
  const bool supportsScan_;
  RowTypePtr rowType_;
};

class Schema;

/// Base class for table. This is used for name resolution. A TableLayout is
/// used     for Split generation, statistics, sampling etc.
class Table {
 public:
  virtual ~Table() = default;

  Table(const std::string& name, const Schema* schema)
      : schema_(schema), name_(name) {}

  const std::string& name() const {
    return name_;
  }

  const RowTypePtr& rowType() const {
    return type_;
  }

  const Schema* schema() const {
    return schema_;
  }

  /// Returns the set of columns as abstract, non-owned
  /// columns. Implementations may hav different Column
  /// implementations with different options, so we do not return the
  /// implementation's columns but an abstract form.
  virtual const std::unordered_map<std::string, const Column*>& columnMap()
      const = 0;

  const Column* findColumn(const std::string& name) {
    auto& map = columnMap();
    auto it = map.find(name);
    return it == map.end() ? nullptr : it->second;
  }

  virtual const std::vector<const TableLayout*>& layouts() const = 0;

  virtual uint64_t numRows() const = 0;

 protected:
  const Schema* const schema_;
  const std::string name_;

  // Discovered from data. In the event of different types, we take the
  // latest (i.e. widest) table type.
  RowTypePtr type_;
};

/// Base class for collection of tables. A query executes against a
/// Schema and its tables and columns are resolved against the
/// Schema. The schema is mutable and may acquire tables and the
/// tables may acquire stats during their lifetime.
class Schema {
 public:
  virtual ~Schema() = default;

  Schema(const std::string& name, memory::MemoryPool* pool)
      : name_(name), pool_(std::move(pool)) {}

  Table* findTable(const std::string& name) {
    auto it = tables_.find(name);
    VELOX_CHECK(it != tables_.end(), "Table {} not found", name);
    return it->second.get();
  }

  virtual connector::Connector* connector() const = 0;

  virtual const std::shared_ptr<connector::ConnectorQueryCtx>&
  connectorQueryCtx() const = 0;

 protected:
  const std::string name_;

  memory::MemoryPool* const pool_;

  std::unordered_map<std::string, std::unique_ptr<Table>> tables_;
};

} // namespace facebook::velox::runner
