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
 *0123 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/common/base/Fs.h"
#include "velox/experimental/query/SchemaSource.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Statistics.h"


namespace facebook::verax {

class LocalSchema : public SchemaSource {
 public:
  LocalSchema(const std::string& path, velox::dwio::common::FileFormat format);

  void fetchSchemaTable(std::string_view name, SchemaPtr schema) override;

 private:
  struct LocalColumn {
    std::string name;
    velox::TypePtr type;
    std::unique_ptr<velox::dwio::common::ColumnStatistics> stats;
  };

  struct LocalTable {
    velox::dwio::common::FileFormat format;
    velox::TypePtr type;
    std::vector<std::string> files;
    std::unordered_map<std::string, std::unique_ptr<LocalColumn>> columns;
  };

  void initialize(const std::string& path);

  void readTable(const std::string& tableName, const fs::path& tablePath);  
  velox::dwio::common::FileFormat format_;

  std::unordered_map<std::string, std::unique_ptr<LocalTable>> tables_;
  std::unique_ptr<velox::memory::MemoryPool> pool_;
};


} // namespace facebook::verax
