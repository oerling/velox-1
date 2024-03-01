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
    : capacity_(std::max<int32_t>(64, bits::nextPowerOfTwo(capacity))) {
  sizeMask_ = capacity_ - 1;
  table_.resize(capacity_ * kEntrySize);
  memset(table_.data(), 0, table_.size());
  maxEntries_ = capacity_ / 4 * 2;
  lastEntryOffset_ = (capacity_ - 1) * kEntrySize;
}

void StringViewIdMap::resize(int32_t newSize) {
  raw_vector<uint8_t> oldTable = std::move(table_);
  auto limit = lastEntryOffset_;
  capacity_ = newSize;
  sizeMask_ = newSize - 1;
  lastEntryOffset_ = (capacity_ - 1) * kEntrySize;
  maxEntries_ = capacity_ / 4 * 2;
  table_.resize(newSize * kEntrySize);
  memset(table_.data(), 0, table_.size());
  auto oldData = oldTable.data();
  auto data = table_.data();
  for (int32_t offset = 0; offset <= limit; offset += kEntrySize) {
    if (*reinterpret_cast<int64_t*>(oldData + offset) == kEmpty) {
      continue;
    }
    auto* view = reinterpret_cast<const StringView*>(oldData + offset);
    auto newOffset = (hash1(*view) & sizeMask_) * kEntrySize;
    for (;;) {
      if (reinterpret_cast<int64_t*>(data + newOffset) == 0) {
        addEntry(
            reinterpret_cast<int64_t*>(data + newOffset),
            *view,
            oldTable[offset + 2]);
        break;
      }
      newOffset = nextOffset(newOffset);
    }
  }
}

void StringViewIdMap::clear() {
  numEntries_ = 0;
  memset(table_.data(), 0, table_.size());
}

static inline void hash(
    const int64_t* headArray,
    const int64_t* tailArray,
    int32_t i,
    int32_t* hashArray) {
  hashArray[i] = tailArray[i] == 0
      ? simd::crc32U64(1, headArray[1])
      : simd::crc32U64(headArray[i] >> 32, tailArray[i]);
}

  
void StringViewIdMap::findIds8(
    const StringView* views,
    const int32_t* indices,
    bool makeIds,
    int32_t* ids,
    char** tails) {
  xsimd::broadcast<int64_t>(0).store_unaligned(
      reinterpret_cast<int64_t*>(tails));
  xsimd::broadcast<int64_t>(0).store_unaligned(
      reinterpret_cast<int64_t*>(tails) + 4);
  if (makeIds) {
    if (UNLIKELY(numEntries_ >= maxEntries_)) {
      resize(capacity_ * 2);
    }
  }
#define LOAD_HEAD(n)                                           \
  auto gatherIndices##n =                                      \
      simd::loadGatherIndices<int64_t, int32_t>(               \
          reinterpret_cast<const int32_t*>(indices + (n * 4))) \
      << 4;                                                    \
  auto heads##n = simd::gather<int64_t, int32_t, 1>(           \
      reinterpret_cast<const int64_t*>(views), gatherIndices##n);

  LOAD_HEAD(0);
  LOAD_HEAD(1);

#define LOAD_TAIL(n)                                        \
  auto lengths##n = heads##n & bits::lowMask(32);           \
  auto haveTail##n = lengths##n > 4;                        \
  auto tails##n = simd::maskGather<int64_t, int32_t, 1>(    \
      xsimd::broadcast<int64_t>(0),                         \
      haveTail##n,                                          \
      reinterpret_cast<const int64_t*>(views) + 1,          \
      gatherIndices##n);                                    \
  auto viewTails##n = tails##n;                             \
  auto longTails##n = lengths##n > StringView::kInlineSize; \
  tails##n = simd::maskGather<int64_t, int64_t, 1>(         \
      tails##n, longTails##n, nullptr, tails##n + lengths##n - 8);

  LOAD_TAIL(0);
  LOAD_TAIL(1);

  // Store heads and tails in memory to read for CRC
  int64_t headArray[8];
  int64_t tailArray[8];
  heads0.store_unaligned(&headArray[0]);
  tails0.store_unaligned(&tailArray[0]);
  heads1.store_unaligned(&headArray[4]);
  tails1.store_unaligned(&tailArray[4]);

  // Compute 8 hashes
  int32_t hashArray[8];
  for (auto i = 0; i < 8; ++i) {
    hash(headArray, tailArray, i, hashArray);
  }

  // Scale the hashes to table size, all 8 at a time.
  ((xsimd::load_unaligned(&hashArray[0]) & sizeMask_) * kEntrySize)
      .store_unaligned(&hashArray[0]);

  #define COMPARE_TABLE_HEAD(n)						\
  auto offsetBytes##n =                                                 \
  simd::loadGatherIndices<int64_t, int32_t>(&hashArray[n * 4]);		\
  auto words##n = simd::gather<int64_t, int32_t, 1>(                    \
      reinterpret_cast<const int64_t*>(table_.data()), offsetBytes##n); \
  auto hits##n = heads##n == words##n;                                  \
  auto empties ##n = words##n == kEmpty;



  COMPARE_TABLE_HEAD(0);
  COMPARE_TABLE_HEAD(1);

  uint16_t hitBits = simd::toBitMask(hits0) | (simd::toBitMask(hits1) << 4);
  uint16_t emptyBits = simd::toBitMask(empties0) | simd::toBitMask(empties1) << 4;
  uint16_t tailBits = simd::toBitMask(haveTail0) | simd::toBitMask(haveTail1) << 4;
  uint16_t longTailBits =
      simd::toBitMask(longTails0) | simd::toBitMask(longTails1) << 4;


  auto tailCheck = hitBits & tailBits & ~longTailBits;
  uint16_t longTailCheck = hitBits & longTailBits;
#define CHECK_SHORT_TAIL(n)                                            \
  if ((tailCheck & (0xf << (n * 4))) != 0) {                           \
    auto tableTails = simd::maskGather<int64_t, int32_t, 1>(	       \
        xsimd::broadcast<int64_t>(0),                                  \
        haveTail##n,                                                   \
        reinterpret_cast<int64_t*>(table_.data()) + 1,                 \
        offsetBytes##n);                                               \
    uint16_t tailMiss = simd::toBitMask(tableTails != viewTails##n) & (tailCheck >> (n * 4)); \
    hitBits ^= tailMiss << (n * 4);                                   \
  }
  CHECK_SHORT_TAIL(0);
  CHECK_SHORT_TAIL(1);

  while (longTailCheck) {
    auto lane = bits::getAndClearLastSetBit(longTailCheck);
    hitBits ^=
        (0 !=
         memcmp(
             tails[lane],
             *reinterpret_cast<char**>(table_.data() + hashArray[lane] + 8),
             headArray[lane] & bits::lowMask(32)))
        << lane;
  }

  if (hitBits == 0xff) {
    simd::gather<int32_t, int32_t, 1>(
        reinterpret_cast<const int32_t*>(table_.data() + sizeof(StringView)),
        simd::loadGatherIndices<int32_t, int32_t>(&hashArray[0]))
        .store_unaligned(&ids[0]);
    return;
  }
  if (!makeIds) {
    simd::maskGather<int32_t, int32_t, 1>(
        xsimd::broadcast<int32_t>(kNotFound),
        simd::fromBitMask<int32_t>(hitBits),
        reinterpret_cast<const int32_t*>(table_.data() + sizeof(StringView)),
        simd::loadGatherIndices<int32_t, int32_t>(&hashArray[0]))
        .store_unaligned(ids);
    uint16_t remaining = (~hitBits & ~emptyBits) & 0xff;
    while (remaining) {
      auto lane = bits::getAndClearLastSetBit(remaining);
      ids[lane] = findEntry<false, false>(
					  nextOffset(hashArray[lane]), views[indices[lane]], headArray[lane], nullptr);
    }
  }
  simd::gather<int32_t, int32_t, 1>(
      reinterpret_cast<const int32_t*>(table_.data() + sizeof(StringView)),
      simd::loadGatherIndices<int32_t, int32_t>(&hashArray[0]))
      .store_unaligned(&ids[0]);
  uint16_t remaining = ~hitBits & 0xff;
  while (remaining) {
    auto lane = bits::getAndClearLastSetBit(remaining);
    auto offset = hashArray[lane];
    if ((emptyBits & (1 << lane)) == 0) {
      offset = nextOffset(offset);
    }
    ids[lane] = findEntry<false, false>(
				      offset, views[indices[lane]], headArray[lane], &tails[lane]);
  }
}

} // namespace facebook::velox
