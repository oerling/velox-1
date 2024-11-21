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

  uint64_t approxNumDistinct(uint64_t deflt = 0) const override {
    return approxNumDistinct_.has_value() ? approxNumDistinct_.value() : deflt;
  }

private:  
  std::optional<uint64_t> approxNumDistinct_; 

  friend class LocalSchema;
};

class LocalSchema;

class LocalTable : public Table {
 public:
  LocalTable(
      const std::string& name,
      dwio::common::FileFormat format,
      Schema* schema)
      : Table(name, format, schema) {}

  std::unordered_map<std::string, std::unique_ptr<LocalColumn>>& columns() {
    return columns_;
  }

  const std::unordered_map<std::string, const Column*>& columnMap() const override;
  
  LocalSchema* schema() const {
    return reinterpret_cast<LocalSchema*>(schema_);
  }

  void setType(const RowTypePtr& type) {
    type_ = type;
  }

  std::pair<int64_t, int64_t> sample(
      float pct,
      const std::vector<common::Subfield>& columns,
      connector::hive::SubfieldFilters filters,
      const core::TypedExprPtr& remainingFilter,
      HashStringAllocator* allocator = nullptr,
      std::vector<std::unique_ptr<dwrf::StatisticsBuilder>>* statsBuilders =
          nullptr) override;

  /// Samples  'samplePct' % rows of the table and sets the num distincts
  /// estimate for the columns. uses 'pool' for temporary data.
  void sampleNumDistincts(float samplePct, memory::MemoryPool* pool);

  const std::vector<std::string>& files() const {
    return files_;
  }


  uint64_t numRows() const override {
    return numRows_;
  }

  
 private:
  // Serializes initialization, e.g. exportedColumns_.
  mutable std::mutex mutex_;

  // All columns. Filled by loadTable().
  std::unordered_map<std::string, std::unique_ptr<LocalColumn>> columns_;

  // Non-owning columns map used for exporting the column set as abstract columns.
  mutable std::unordered_map<std::string, const Column*> exportedColumns_;

  std::vector<std::string> files_;
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

  connector::Connector* connector() const override {
    return hiveConnector_;
  }

  const std::shared_ptr<connector::ConnectorQueryCtx>& connectorQueryCtx()
      const {
    return connectorQueryCtx_;
  }

  const std::unordered_map<std::string, std::unique_ptr<Table>>& tables() const {
    return tables_;
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
