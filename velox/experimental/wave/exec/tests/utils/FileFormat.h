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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/type/StringView.h"
#include "velox/vector/ComplexVector.h"

/// Sample set of composable encodings. Bit packing, direct and dictionary.
namespace facebook::velox::wave::test {

enum Encoding { kFlat, kDict };

struct Column {
  TypeKind kind;
  Encoding encoding;
  // Number of encoded values.
  int32_t numValues{0};

  // Distinct values in kDict.
  std::shared_ptr<Column> alphabet;

  // Encoded lengths for strings in 'data' if type is varchar and encoding is
  // kFlat.
  std::unique_ptr<Column> lengths;

  // Width of bit packed encoding.
  int8_t bitWidth;

  BufferPtr values;
};

    struct Stripe {
  Stripe(
      std::vector<std::unique_ptr<Column>>&& in,
      std::shared_ptr<const common::TypeWithId> type)
      : typeWithId(std::move(type)), columns(std::move(in)) {}

  Column* findColumn(TypeWithId& child);

  // Unique name assigned when associating with a Table.
  std::string name;

  dwio::common::TypeWithId typeWithId;

  // Top level columns.
  std::vector<std::unique_ptr<Column>> columns;
};

class StringSet {
 public:
  StringSet(memory::MemoryPool* pool) : pool_(pool) {}

  StringView add(StringView data);
  std::unique_ptr<Column> toColumn();

 private:
  int64_t totalSize_{0};
  int32_t maxLength_{0};
  std::vector<int32_t> lengths_;
  std::vector<BufferPtr> buffers_;
  memory::MemoryPool* pool_;
};

class Encoder {
 public:
  Encoder(memory::MemoryPool* pool)
      : pool_(pool), dictStrings_(pool), allStrings_(pool) {}
  // Adds data.
  void append(VectorPtr data);

  // Retrieves the data added so far as an encoded column.
  std::unique_ptr<Column> toColumn();

 private:
  template <TypeKind kind>
  void appendTyped(VectorPtr data);

  template <typename T>
  void add(T data);

  int64_t flatSize();
  int64_t dictSize();

  memory::MemoryPool* pool_;
  TypeKind kind_{TypeKind::UNKNOWN};
  int32_t count_{0};
  // Distincts for either int64_t or double.
  folly::F14FastMap<uint64_t, int32_t> ints_;
  // Distincts for strings.
  folly::F14FastMap<StringView, int32_t> strings_;
  // Values as indices into dicts.
  std::vector<int32_t> indices_;

  std::vector<uint64_t> dictInts_;
  // The fixed width values as direct.
  std::vector<uint64_t> direct_;
  // True if too many distinct values for dict.
  bool abandonDict_{false};
  uint64_t max_{0};
  // longest string, if string type.
  int32_t maxLength_{0};
  // Total bytes in distinct strings.
  int64_t dictBytes_{0};
  // Total string bytes without dict.
  int32_t totalStringBytes_{0};

  StringSet dictStrings_;
  StringSet allStrings_;
};

class Writer {
 public:
  Writer(int32_t stripeSize)
      : stripeSize_(stripeSize),
        pool_(memory::MemoryManager::getInstance()->addLeafPool()) {}

  /// Appends a batch of data.
  void append(RowVectorPtr data);

  // Finishes encoding data, makes the table ready to read.
  void finalize(std::string tableName);

 private:
  TypePtr type_;
  void finishStripe();
  const int32_t stripeSize_;
  std::vector<std::unique_ptr<Stripe>> stripes_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::vector<std::unique_ptr<Encoder>> encoders_;
};

using SplitVector = std::vector<std::shared_ptr<connector::ConnectorSplit>>;

class Table {
 public:
  Table(const std::string name) : name_(name) {}

  static const Table* defineTable(
      const std::string& name,
      const std::vector<RowVectorPtr>& data);

  static Table* getTable(const std::string& name, bool makeNew = false) {
    std::lock_guard<std::mutex> l(mutex_);
    auto it = allTables_.find(name);
    if (it == allTables_.end()) {
      if (makeNew) {
        auto table = std::make_unique<Table>(name);
        auto ptr = table.get();
        allTables_[name] = std::move(table);
        return ptr;
      }
      return nullptr;
    }
    return it->second.get();
  }

  static void dropTable(const std::string& name);

  static Stripe* getStripe(const std::string& path) {
    std::lock_guard<std::mutex> l(mutex_);
    auto it = allStripes_.find(path);
    VELOX_CHECK(it != allStripes_.end());
    return it->second;
  }

  void addStripes(
      std::vector<std::unique_ptr<Stripe>>&& stripes,
      std::shared_ptr<memory::MemoryPool> pool) {
    std::lock_guard<std::mutex> l(mutex_);
    for (auto& s : stripes) {
      s->name = fmt::format("wavemock://{}/{}", name_, stripes_.size());
      stripes_.push_back(std::move(s));
    }
    pools_.push_back(pool);
  }

  int32_t numStripes() const {
    return stripes_.size();
  }

  Stripe* stripeAt(int32_t index) {
    std::lock_guard<std::mutex> l(mutex_);
    if (index >= stripes_.size()) {
      return nullptr;
    }
    return stripes_[index].get();
  }

  SplitVector splits() const {
    SplitVector result;
    std::lock_guard<std::mutex> l(mutex_);
    for (auto& stripe : stripes_) {
      result.push_back(std::make_shared<connector::hive::HiveConnectorSplit>(
          "wavemock", stripe->name, dwio::common::FileFormat::UNKNOWN));
    }
    return result;
  }

 private:
  static std::mutex mutex_;
  static std::unordered_map<std::string, std::unique_ptr<Table>> allTables_;
  static std::unordered_map<std::string, Stripe*> allStripes_;
  std::vector<std::shared_ptr<memory::MemoryPool>> pools_;
  std::string name_;
  std::vector<std::unique_ptr<Stripe>> stripes_;
};

} // namespace facebook::velox::wave::test
