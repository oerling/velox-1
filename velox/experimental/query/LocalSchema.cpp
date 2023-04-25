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
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/experimental/query/QueryGraph.h"

#include "velox/common/base/Fs.h"

namespace facebook::verax {
using namespace facebook::velox;

LocalSchema::LocalSchema(
    const std::string& path,
    velox::dwio::common::FileFormat fmt,
    const std::string& connectorId,
    std::shared_ptr<velox::memory::MemoryPool> pool)
    : connectorId_(connectorId), pool_(pool) {
  format_ = fmt;
  initialize(path);
  locus_ = std::make_unique<Locus>(connectorId_.c_str());
}

void LocalSchema::initialize(const std::string& path) {
  for (auto const& dirEntry : fs::directory_iterator{path}) {
    if (!dirEntry.is_directory() ||
        dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    readTable(dirEntry.path().filename(), dirEntry.path());
  }
}

void LocalSchema::readTable(
    const std::string& tableName,
    const fs::path& tablePath) {
  RowTypePtr tableType;
  LocalTable* table = nullptr;

  for (auto const& dirEntry : fs::directory_iterator{tablePath}) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    auto it = tables_.find(tableName);
    if (it != tables_.end()) {
      table = it->second.get();
    } else {
      tables_[tableName] = std::make_unique<LocalTable>(tableName, format_);
      table = tables_[tableName].get();
    }
    dwio::common::ReaderOptions readerOptions{pool_.get()};
    readerOptions.setFileFormat(format_);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(dirEntry.path().string()),
        readerOptions.getMemoryPool());
    std::unique_ptr<dwio::common::Reader> reader =
        dwio::common::getReaderFactory(readerOptions.getFileFormat())
            ->createReader(std::move(input), readerOptions);
    const auto fileType = reader->rowType();
    if (!tableType) {
      tableType = fileType;
    }
    auto rows = reader->numberOfRows();
    if (rows.has_value()) {
      table->numRows += rows.value();
    }
    for (auto i = 0; i < fileType->size(); ++i) {
      auto name = fileType->nameOf(i);
      LocalColumn* column;
      auto columnIt = table->columns.find(name);
      if (columnIt != table->columns.end()) {
        column = columnIt->second.get();
      } else {
        table->columns[name] =
            std::make_unique<LocalColumn>(name, fileType->childAt(i));
        column = table->columns[name].get();
      }
      column->addStats(reader->columnStatistics(i));
    }
    table->files.push_back(dirEntry.path());
  }
  if (table) {
    table->type = tableType;
  }
}

void LocalColumn::addStats(
    std::unique_ptr<dwio::common::ColumnStatistics> stats) {
  if (!stats) {
    stats = std::move(stats);
  }
}

void LocalSchema::fetchSchemaTable(
    std::string_view name,
    const Schema* schema) {
  auto str = std::string(name);
  auto it = tables_.find(str);
  if (it == tables_.end()) {
    return;
  }
  auto table = it->second.get();
  Declare(SchemaTable, schemaTable, toName(str), table->rowType());
  ColumnVector columns;
  for (auto& pair : table->columns) {
    float cardinality = table->numRows;
    auto original = pair.second.get();
    Value value(original->type.get(), cardinality);
    auto columnName = toName(pair.first);
    Declare(Column, column, columnName, nullptr, value);
    schemaTable->columns[columnName] = column;
    columns.push_back(column);
  }
  DistributionType defaultDist;
  defaultDist.locus = locus_.get();
  schemaTable->addIndex(
      toName("pk"), table->numRows, 0, 0, {}, defaultDist, {}, columns);
  schema->addTable(schemaTable);
}

} // namespace facebook::verax
