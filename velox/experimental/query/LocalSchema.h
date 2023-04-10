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
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/experimental/query/Schema.h"

namespace facebook::verax {

struct LocalColumn {
  LocalColumn(const std::string& name, velox::TypePtr type)
      : name(name), type(type) {}

  void addStats(std::unique_ptr<velox::dwio::common::ColumnStatistics> stats);

  std::string name;
  velox::TypePtr type;
  std::unique_ptr<velox::dwio::common::ColumnStatistics> stats;
};

struct LocalTable {
  LocalTable(const std::string& name, velox::dwio::common::FileFormat format)
      : name(name), format(format) {}

  const velox::RowTypePtr& rowType() const {
    return type;
  }

  std::string name;
  velox::dwio::common::FileFormat format;
  velox::RowTypePtr type;
  std::vector<std::string> files;
  std::unordered_map<std::string, std::unique_ptr<LocalColumn>> columns;
  int64_t numRows{0};
};

class LocalSchema : public SchemaSource {
 public:
  LocalSchema(
      const std::string& path,
      velox::dwio::common::FileFormat format,
      const std::string& connectorId);

  void fetchSchemaTable(std::string_view name, const Schema* schema) override;

  const std::unordered_map<std::string, std::unique_ptr<LocalTable>>& tables() {
    return tables_;
  }

  LocalTable* findTable(const std::string& name) {
    auto it = tables_.find(name);
    VELOX_CHECK(it != tables_.end(), "Table {} not found", name);
    return it->second.get();
  }

 private:
  void initialize(const std::string& path);

  void readTable(const std::string& tableName, const fs::path& tablePath);
  velox::dwio::common::FileFormat format_;
  std::string connectorId_;
  std::unique_ptr<Locus> locus_;
  std::unordered_map<std::string, std::unique_ptr<LocalTable>> tables_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

} // namespace facebook::verax
