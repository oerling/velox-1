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
#include "velox/type/StringView.h"
#include "velox/vector/ComplexVector.h"

/// Sample set of composable encodings. Bit packing, direct and dictionary.
namespace facebook::velox::wave {

enum Encoding { kFlat, kBits, kDict };

struct Column {
  Encoding encoding;
  // Number of encoded values.
  int32_t numValues;
  // Distinct values in kDict.
  std::shared_ptr<Column> alphabet;
  // Encoded lengths for strings in 'data' if type is varchar and encoding is
  // kFlat.
  std::unique_ptr<Column> lengths;
  // Width of bit packed encoding.
  int8_t bitWidth;
  // Byte size of encoded data in 'data'.
  int32_t numBytes;

  BufferPtr data;
};

struct Stripe {
  Stripe(std::vector<std::unique_ptr<Column>>&& in) : columns(std::move(in)) {}

  std::vector<std::unique_ptr<Column>> columns;
};

class StringSet {
 public:
  StringSet(memory::MemoryPool* pool) : pool_(pool) {}

  StringView add(StringView data);
  std::unique_ptr<Column> column();

 private:
  std::vector<int32_t> lengths_;
  std::vector<BufferPtr> buffers_;
  memory::MemoryPool* pool_;
};

class Encoder {
 public:
  Encoder(memory::MemoryPool* pool) : dictStrings_(pool), allStrings_(pool) {}
  // Adds data.
  void append(VectorPtr data);

  // Retrieves the data added so far as an encoded column.
  std::unique_ptr<Column> column();

 private:
  template <TypeKind kind>
  void appendTyped(VectorPtr data);

  template <typename T>
  void add(T data);

  // Distincts for either int64_t or double.
  folly::F14FastMap<uint64_t, int32_t> ints_;
  // Distincts for strings.
  folly::F14FastMap<StringView, int32_t> strings_;
  // Values as indices into dicts.
  std::vector<int32_t> indices_;
  // The fixed width values as direct.
  std::vector<uint64_t> direct_;
  // True if too many distinct values for dict.
  bool abandonDict_{false};
  // longest string, if string type.
  int32_t maxLength{0};

  StringSet dictStrings_;
  StringSet allStrings_;
};

class Writer {
 public:
  Writer(int32_t stripeSize)
      : 
        stripeSize_(stripeSize),
        pool_(memory::addDefaultLeafMemoryPool()) {}

  /// Appends a batch of data.
  void append(RowVectorPtr data);
  // Finishes encoding data, makes the table ready to read.
  void finalize(std::string tableName);

 private:
  void finishStripe();
  const int32_t stripeSize_;
  std::vector<std::unique_ptr<Stripe>> stripes_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::vector<std::unique_ptr<Encoder>> encoders_;
};

struct WaveTestConnectorSplit : public connector::ConnectorSplit {
  Stripe* stripe;
};

class Table {
 public:
  Table(const std::string name) : name_(name) {}

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
  void addStripes(
      std::vector<std::unique_ptr<Stripe>>&& stripes,
      std::shared_ptr<memory::MemoryPool> pool) {
    std::lock_guard<std::mutex> l(mutex_);
    for (auto& s : stripes) {
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

 private:
  static std::mutex mutex_;
  static std::unordered_map<std::string, std::unique_ptr<Table>> allTables_;

  std::vector<std::shared_ptr<memory::MemoryPool>> pools_;
  std::string name_;
  std::vector<std::unique_ptr<Stripe>> stripes_;
};

} // namespace facebook::velox::wave
