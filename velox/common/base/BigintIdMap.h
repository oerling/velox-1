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
  static constexpr int64_t kEmptyMarker = 0;
  static constexpr int32_t kAllSet = bits::lowMask(xsimd::batch<int64_t>::size);

  BigintIdMap(int32_t capacity, memory::MemoryPool& pool) : pool_(pool) {
    makeTable(bits::nextPowerOfTwo(capacity));
  }

  ~BigintIdMap() {
    if (table_) {
      pool_.free(table_, byteSize_);
    }
  }

  /// Returns a batch of unique ids for a batch of arbitrary int64_t
  /// values. Each value is given an int32_t id when first seen and
  /// this same id will be given on subsequent occurrences. 'mask'
  /// specifies the active lanes of 'x'. The id for a non-active lane
  /// of x is always zero. Ids for values start at 1.
  xsimd::batch<int64_t> makeIds(
      xsimd::batch<int64_t> x,
      uint8_t mask = kAllSet) {
    // 0 is the id for a non-active lane.
    xsimd::batch_bool<int64_t> activeLanes;
    if (FOLLY_UNLIKELY(mask != kAllSet)) {
      if (FOLLY_UNLIKELY(!mask)) {
        return xsimd::broadcast<int64_t>(0);
      }
      activeLanes = simd::fromBitMask<int64_t, int64_t>(mask);
    } else {
      activeLanes = x == x; // All true.
    }
    auto ready = xsimd::batch<int64_t>::broadcast(0);
    xsimd::batch_bool<int64_t> emptyMarkerVector =
        x == xsimd::broadcast<int64_t>(kEmptyMarker);

    auto emptyMarkerMask = simd::toBitMask(emptyMarkerVector);
    if (FOLLY_UNLIKELY(emptyMarkerMask)) {
      // The zeros on active lanes get zeroId_.
      if (!emptyId_ && (emptyMarkerMask & mask)) {
        // Assign an id to kEmptyMarker when it first occurs on an active lane.
        emptyId_ = ++lastId_;
        emptyBatch_ = xsimd::broadcast(static_cast<int64_t>(emptyId_));
      }
      // 'ready' is all 0. Now, we OR emptyId_ to the lanes that are
      // both active and zero. We rely on batch_bool having the same
      // layout as a regular batch with all bits set for true. We
      // store and reload with interpret_cast because xsimd has no
      // conversion betwee batch_bool and batch.
      xsimd::batch_bool<int64_t> temp = emptyMarkerVector & activeLanes;
      ready = ready |
          (xsimd::load_unaligned(reinterpret_cast<int64_t*>(&temp)) &
           emptyBatch_);
      activeLanes = activeLanes & ~emptyMarkerVector;
    }
    if (FOLLY_UNLIKELY(!simd::toBitMask(activeLanes))) {
      return ready;
    }

    auto indices = makeIndices(x);
    auto data = simd::maskGather<int64_t, int64_t, 4>(
        ready, activeLanes, reinterpret_cast<const int64_t*>(table_), indices);

    auto matchVector = x == data;
    ready = simd::maskGather<int64_t, int64_t, 4>(
        ready,
        matchVector,
        reinterpret_cast<const int64_t*>(table_) + 1,
        indices);
    uint16_t matches = simd::toBitMask(matchVector | ~activeLanes);
    if (matches == kAllSet) {
      return ready & kLow32;
    }
    // Store the indices and the values to look up in memory.
    auto indexVector = indices;
    auto dataVector = x;
    auto resultVector = ready;
    auto indexArray = reinterpret_cast<int64_t*>(&indexVector);
    auto dataArray = reinterpret_cast<int64_t*>(&dataVector);
    auto resultArray = reinterpret_cast<int64_t*>(&resultVector);
    uint16_t misses = matches ^ kAllSet;
    while (misses) {
      auto index = bits::getAndClearLastSetBit(misses);
      int32_t byteOffset = 4 * (indexArray[index]);
      for (;;) {
        auto value = *valuePtr(table_ + byteOffset);
        if (value == kEmptyMarker) {
          *reinterpret_cast<int64_t*>(table_ + byteOffset) = dataArray[index];
          resultArray[index] =
              *reinterpret_cast<int32_t*>(table_ + byteOffset + 8) = ++lastId_;
          ++numEntries_;
          break;
        }
        if (value == dataArray[index]) {
          resultArray[index] = *reinterpret_cast<int32_t*>(
              table_ + byteOffset + sizeof(int64_t));
          break;
        }
        byteOffset += kEntrySize;
        if (byteOffset >= limit_) {
          byteOffset = 0;
        }
      }
    }
    if (numEntries_ > maxEntries_) {
      resize(capacity_ * 2);
    }
    return xsimd::load_unaligned(resultArray) & kLow32;
  }

 private:
  static constexpr int32_t kEntrySize = sizeof(int64_t) + sizeof(int32_t);
  static constexpr int64_t kLow32 = (1L << 32) - 1;

  // Constants for hash calculation.
  static constexpr uint64_t kMultLow = 1971049UL;
  static constexpr uint64_t kMultHigh = 1470709UL;

  // Allocates a new table.
  void makeTable(int32_t capacity);

  // Returns the pointer to the value of the 'i'th entry in 'table'.
  int64_t* valuePtr(void* table, int32_t i) {
    return reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(table) + kEntrySize * i);
  }

  // Returns the pointer of the int32_t id for an entry.
  int32_t* idPtr(int64_t* valuePtr) {
    return reinterpret_cast<int32_t*>(valuePtr + 1);
  }

  // Rehashes 'this' to a size of 'newCapacity'.
  void resize(int32_t newCapacity);

  // Returns the hashed position of 'value' as a
  // an index into an array of 12 byte entries. The function  is the same as indices()  for a single value. The difference is that indices returns distances in 4 byte words and this returns them i 
      .int64_t indexOfEntry(int64_t value) {
    uint32_t high = kMultHigh * (static_cast<uint64_t>(value) >> 32);
    uint32_t low = kMultLow * static_cast<uint32_t>(value);
    auto entry = ((high ^ low) & sizeMask_);
    return entry;
  }

  xsimd::batch<int64_t> makeIndices(xsimd::batch<int64_t> values) {
    auto multiplier =
        xsimd::batch<uint64_t>::broadcast(kMultHigh << 32 | kMultLow);
    auto hash = simd::reinterpretBatch<uint64_t>(
        simd::reinterpretBatch<uint32_t>(values) *
        simd::reinterpretBatch<uint32_t>(multiplier));
    auto indices =
        simd::reinterpretBatch<int64_t>(((hash >> 32) ^ hash) & sizeMask_);
    return indices + indices + indices;
  }

  memory::MemoryPool& pool_;

  // Counter for assigning ids to values.
  int32_t lastId_{0};

  // Id for value == kEmptyMarker
  int32_t emptyId_{0};

  //  emptyId_ in all lanes.
  xsimd::batch<int64_t> emptyBatch_;

  // Entries, 12 bytes per entry, 8 first are the value, the next 4 are its
  // assigned id.
  char* table_{nullptr};

  // Number of 12 byte entries in 'table_'.
  int64_t capacity_;

  // Mask, one less than 'capacity_'.
  int64_t sizeMask_;

  // Allocation byte size of 'table_', including padding.
  int64_t byteSize_;

  // Byte offset of first byte after last byte of 'table_'.
  int64_t limit_;

  // Count of non-empty entries in 'table_'.
  int32_t numEntries_{0};

  // Count of entries after which a resize() should be done.
  int32_t maxEntries_;
};

} // namespace facebook::velox
