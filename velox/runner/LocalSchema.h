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
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"
#include "velox/runner/Schema.h"

namespace facebook::velox::runner {

class LocalColumn : public Column {
 public:
  LocalColumn(const std::string& name, TypePtr type) : Column(name, type) {}

 private:
  std::optional<uint64_t> approxNumDistinct_;

  friend class LocalSchema;
};

/// Describes a Hive table layout. Adds a file format and a list of
/// Hive partitioning columns to the base TableLayout. The
/// partitioning in TableLayout does not differentiate between
/// bucketing and Hive partitioning columns. The bucketing columns
/// are the 'partitioning' columns minus the
/// 'hivePartitioningColumns'
class HiveTableLayout : public TableLayout {
 public:
  HiveTableLayout(
      const std::string& name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitioning,
      std::vector<const Column*> orderColumns,
      std::vector<core::SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      std::vector<const Column*> hivePartitionColumns,
      dwio::common::FileFormat fileFormat)
      : TableLayout(
            name,
	    table,
            connector,
            columns,
            partitioning,
            orderColumns,
            sortOrder,
            lookupKeys,
            true),
        fileFormat_(fileFormat),
        hivePartitionColumns_(hivePartitionColumns) {}


  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  std::unique_ptr<connector::LayoutMetadata> metadata() const override;

  const std::vector<const Column*>& hivePartitionColumns() const {
    return hivePartitionColumns_;
  }
  
protected:
  const dwio::common::FileFormat fileFormat_;
  const std::vector<const Column*> hivePartitionColumns_;
};

/// A HiveTableLayout backed by local files. Implements sampling by reading
/// local files and stores the file list inside 'this'.
class LocalHiveTableLayout : public HiveTableLayout {
 public:
  LocalHiveTableLayout(
      const std::string& name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitioning,
      std::vector<const Column*> orderColumns,
      std::vector<core::SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      std::vector<const Column*> hivePartitionColumns,
      dwio::common::FileFormat fileFormat)
      : HiveTableLayout(
            name,
	    table,
            connector,
            columns,
            partitioning,
            orderColumns,
            sortOrder,
            lookupKeys,
            hivePartitionColumns,
            fileFormat) {}

  std::pair<int64_t, int64_t> sample(
      const connector::ConnectorTableHandlePtr& handle,
      float pct,
      std::vector<core::TypedExprPtr> extraFilters,
      const std::vector<common::Subfield>& fields,
      HashStringAllocator* allocator = nullptr,
      std::vector<ColumnStatistics>* statistics = nullptr) override;

  const std::vector<std::string>& files() const {
    return files_;
  }

  void setFiles(std::vector<std::string> files) {
    files_ = std::move(files);
  }
  
  /// Like sample() above, but fills 'builders' with the data.
  std::pair<int64_t, int64_t> sample(
      const connector::ConnectorTableHandlePtr& handle,
      float pct,
      const std::vector<common::Subfield>& fields,
      HashStringAllocator* allocator,
      std::vector<std::unique_ptr<dwrf::StatisticsBuilder>>* statsBuilders);

   private:

  std::vector<std::string> files_;
};

class LocalSchema;

class LocalTable : public Table {
 public:
  LocalTable(
      const std::string& name,
      dwio::common::FileFormat format,
      Schema* schema)
      : Table(name, schema) {}

  std::unordered_map<std::string, std::unique_ptr<LocalColumn>>& columns() {
    return columns_;
  }
  const std::vector<const TableLayout*>& layouts() const override {
    return exportedLayouts_;
  }

  const std::unordered_map<std::string, const Column*>& columnMap()
      const override;

  void setType(const RowTypePtr& type) {
    type_ = type;
  }

  void makeDefaultLayout(std::vector<std::string> files);
  
  uint64_t numRows() const override {
    return numRows_;
  }

  /// Samples  'samplePct' % rows of the table and sets the num distincts
  /// estimate for the columns. uses 'pool' for temporary data.
  void sampleNumDistincts(float samplePct, memory::MemoryPool* pool);

  
 private:
  // Serializes initialization, e.g. exportedColumns_.
  mutable std::mutex mutex_;

  // All columns. Filled by loadTable().
  std::unordered_map<std::string, std::unique_ptr<LocalColumn>> columns_;

  // Non-owning columns map used for exporting the column set as abstract
  // columns.
  mutable std::unordered_map<std::string, const Column*> exportedColumns_;

  ///  Table layouts. For a Hive table this is normally one layout with all
  ///  columns included.
  std::vector<std::unique_ptr<TableLayout>> layouts_;

  // Copy of 'llayouts_' for use in layouts().
  std::vector<const TableLayout*> exportedLayouts_;

  int64_t numRows_{0};
  int64_t numSampledRows_{0};

  friend class LocalSchema;
};

class LocalSchema : public Schema {
 public:
  /// 'path' is the directory containing a subdirectory per table.
  LocalSchema(
      const std::string& path,
      dwio::common::FileFormat format,
      connector::hive::HiveConnector* hiveConector,
      std::shared_ptr<connector::ConnectorQueryCtx> ctx);

  const std::shared_ptr<connector::ConnectorQueryCtx>& connectorQueryCtx()
      const {
    return connectorQueryCtx_;
  }

  const std::unordered_map<std::string, std::unique_ptr<Table>>& tables()
      const {
    return tables_;
  }

  connector::Connector* connector() const override {
    return hiveConnector_;
  }

  dwio::common::FileFormat fileFormat() const {
    return format_;
  }
  
 private:
  void initialize(const std::string& path);

  void loadTable(const std::string& tableName, const fs::path& tablePath);

  connector::hive::HiveConnector* const hiveConnector_;
  const std::string connectorId_;
  const std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  const dwio::common::FileFormat format_;
};

} // namespace facebook::velox::runner
