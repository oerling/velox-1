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

#include "velox/experimental/query/LocalSchema.h"
#include "velox/experimental/query/Schema.h"

#include "velox/common/base/Fs.h"

namespace facebook::verax {
  using namespace facebook::velox;
  
LocalSchema::LocalSchema(
    const std::string& path,
    velox::dwio::common::FileFormat fmt) {
  format_ = fmt;
  initialize(path);
}

void LocalSchema::initialize(const std::string& path) {
  for (auto const& dirEntry : fs::directory_iterator{path}) {
    if (!dirEntry.is_directory() || dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    readTable(dirEntry.path().filename(), dirEntry.path());
  }
}

void LocalSchema::readTable(
    const std::string& tableName,
    const fs::path& tablePath) {
  for (auto const& dirEntry : fs::directory_iterator{tablePath}) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    if (tables_.find(tableName) == tables_.end()) {
      tables_[tableName] = std::make_unique<LocalTable>(tableName, format_);
      auto& table = *tables_[tableName];
      dwio::common::ReaderOptions readerOptions{pool_.get()};
      readerOptions.setFileFormat(format_);
      auto input = std::make_unique<dwio::common::BufferedInput>(
          std::make_shared<LocalReadFile>(dirEntry.path().string()),
          readerOptions.getMemoryPool());
      std::unique_ptr<dwio::common::Reader> reader =
          dwio::common::getReaderFactory(readerOptions.getFileFormat())
              ->createReader(std::move(input), readerOptions);
      const auto fileType = reader->rowType();
      for (auto i = 0; i < fileType.size(); ++i) {
        auto name = tableType->nameOf(i);
        auto column =
            std::make_unique<LocalColumn>(name, tableType->childat(i), nullptr);
        table.columns[name] = std::move(column);
      }
    }
    table.dataFiles.push_back(dirEntry.path());
  }
}
void LocalSchema::fetchSchemaTable(std::string_view name, SchemaPtr schema) {}

} // namespace facebook::verax
