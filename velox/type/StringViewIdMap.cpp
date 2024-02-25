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

#include "velox/type/StringViewIdMap.h"

namespace facebook::velox {

StringViewIdMap::StringViewIdMap(int32_t capacity)
    : capacity_(bits::nextPowerOfTwo(capacity)) {
  sizeMask_ = capacity_ - 1;
  table_.resize(3 * capacity_);
  memset(table_.data(), 0, table_.size() * sizeof(int64_t));
  maxEntries_ = capacity_ / 4 * 2;
}

void StringViewIdMap::resize(int32_t newSize) {
  raw_vector<uint64_t> oldTable = std::move(table_);
  capacity_ = newSize;
  sizeMask_ = newSize - 1;
  table_.resize(newSize * 3);
  memset(table_.data(), 0, table_.size() * sizeof(int64_t));
  auto limit = oldTable.size();
  auto oldData = oldTable.data();
  auto data = table_.data();
  for (int32_t offset = 0; offset < limit; offset += 3) {
    if (oldData[offset] == 0) {
      continue;
    }
    auto* view = reinterpret_cast<const StringView*>(oldData + offset);
    auto hash = hash1(*view) & sizeMask_;
    for (;;) {
      if (data[hash * 3] == 0) {
        addEntry(data + hash * 3, *view, oldTable[offset + 2]);
        break;
      }
      hash = (hash + 1) & sizeMask_;
    }
  }
  maxEntries_ = capacity_ / 4 * 2;
}

void StringViewIdMap::clear() {
  numEntries_ = 0;
  memset(table_.data(), 0, table_.size() * sizeof(int64_t));
}

} // namespace facebook::velox
