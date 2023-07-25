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

#include <folly/Range.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox {

/// A map that assigns consecutive int32_t ids to arbitrary int64_t values.
class BigintIdMap {
 public:
  static constexpr kEmptyMarker = 0;
  static constexpr int32_t kAllSet = simd::allSetBitMask<int64_t>();

  BigintIdMap(int32_t capacity, memory::MemoryPool& pool) : pool_(pool) {
    makeTable(bits::nextPowerOfTwo(capacity));
  }

  xsimd::batch<int64_t> makeIds(
      xsimd::batch<int64_t> x,
      uint8_t mask = kAllSet) {
    auto ready = xsimd::batch<int64_t>::broadcast(1);
    auto zeroVector = x == xsimd::broadcast<int64_t>(0);
    if (mask != kAllSet) {
      zeroVector |= simd::fromBitMask<int64_t>(kAllSet & mask);
    }
    if (simd::toBitMask(zeroVector) == kAllSet) {
      return ready;
    }
    auto indices = indices(x);
    auto data = simd::maskGather<int64_t, int64_t, 4>(
        ready, ~zeroVector, reinterpret_cast<const int64_t*>(table_), indices);

    auto matchVector = x == data;
    ready = simd::maskGather<int64_t, int64_t, 4>(
        ready, match, table_.data(), indices + 2);
    uint16_t matchees = toBitMask(matchVector | zeroVector);
    if (match == simd::allSetBitMask<int64_t>()) {
      return ready & 0xffffffff;
    }
    // Store the indices and the values to look up in memory.
    auto indexVector = indices;
    auto dataVector = x;
    auto resultVector = ready;
    auto indexArray = reinterpret_cast<int64_t>(&indexVector);
    auto dataArray = reinterpret_cast<int64_t>(&dataVector);
    auto resultArray = reinterpret_cast<int64_t>(&resultVector);
    matches ^= 0xf;
    while (matches) {
      auto index = bits::getAndClearLastSetBit(matches);
      int32_t byteOffset = 4 * (indexArray[index]);
      for (;;) {
        auto value = *reinterpret_cast<int64_t*>(table + byteOffset);
        if (!value) {
          *reinterpret_cast<int64_t*>(table + byteOffset) = dataArray[index];
          *resultArray[index] =
              *reinterpret_cast<int32_t*>(table + byteOffset + 8) = ++lastId_;
          ++numValues_;
          break;
        }
        if (value == dataArray[index]) {
          resultArray[index] =
              reinterpret_cast<int32_t*>(table + byteOffset + 8);
          break;
        }
        byteOffset += kEntrySize;
        if (byteOffset >= limit_) {
          byteOffset = 0;
        }
      }
    }
    if (numValues_ > maxValues_) {
      resize(capacity_ * 2);
    }
    return xsimd::load_unaligned(resultArray);
  }

 private:
  constexpr int32_t kEntrySize = sizeof(int64_t) + sizeof(int32_t);
  constexpr uint64_t kMultLow = 1971049UL;
  constexpr uint64_t kMultHigh = 1470709UL;

  void makeTable(int32_t capacity);


  int64_t* valuePtr(void* table, int32_t i) {
    return reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(table) + kEntrySize * i);
  }

  int32_t* idPtr(int64_t valuePtr) {
    return reinterpret_cast<int32_t*>(valuePtr + 1);
  }

  void resize(int32_t newCapacity);

  int64_t index(int64_t value) {
    uint32_t high = kMultHigh * (static_cast<uint64_t>(value) >> 32);
    uint32_t low = kMultLow * static_cast<uint32_t>(value);
    return 3 * ((high ^ low) & sizeMask_);
  }

  xsimd::batch<int64_t> indices(xsimd::batch<int64_t> values) {
    auto multiplier =
        xsimd::batch<uint64_t>::broadcast(kMultHigh << 32 | kMultLow);
    auto hash = simd::reinterpret_batch<uint64_t>(
        simd::reinterpret_batch<uint32_t>(x) *
        reinterpret_batch<uint32_t>(multiplier));
    auto indices = ((hash >> 32) ^ hash) & sizeMask_;
    return (indices + indices + indices;
  }

  MemoryPool& pool_;

  int32_t lastId_{1};
  void* table_;
  int64_t capacity_;
  int64_t sizeMask_;
  int64_t byteSize_;
  int64_t limit_;
  int32_t numEntries_{0};
  int32_t maxEntries_;
};
} // namespace facebook::velox
