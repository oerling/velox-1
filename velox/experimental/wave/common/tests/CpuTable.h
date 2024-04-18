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

#include <cstdint>

#include "velox/common/base/SimdUtil.h"

namespace facebook::velox::wave {

  struct CpuBucket {
      using TagVector = xsimd::batch<uint8_t, xsimd::sse2>;
    uint8_t tags[16];
    uint8_t data[128 - 16];



    auto tags() {
      return TagVector(_mm_loadu_si128(reinterpret_cast<__m128i const*>&tags));;
    }

    uint16_t matchtags(TagVector tags, uint8_t tag) {
      auto flags = TagVector::broadcast(tag) == tags;
      return simd::toBitMask(flags);
    }

  

    template <typename T>
    T* load(int32_t idx) {
      uint64_t data = *reinterpret_cast<uint64_t*>(&data[idx * 6]);
      return reinterpret_cast<T*>(data & 0xffffffffffff);
    }

    void store(int32_t idx, void* row){
      uint64_t data = *reinterpret_cast<uint64_t*>(&data[idx * 6]);
      *reinterpret_cast<uint64_t*>(&data[idx * 6]) = (data & 0xffff000000000000) | uptr;
    }
  };

  struct CpuHashTable {
    std::vector<CpuBucket> buckets;
    int32_t sizeMask;
    // Preallocated rows.
    void** rows;
    // Count of preallocated rows.
    int32_t numRows;
    // Number of used rows.
    int32_t usedRows{0};
    
    
    template <typename RowType, typename Ops>
      void updatingProbe(HashProbe* probe) {
      auto numRows = probe->numKeys[0];
      for (auto i  = 0; i < numRows; ++i) {
	auto h = probe->hashes[i];
	uint8_t tag = 0x80 | (h >> 32); 
	auto bucketIdx = h & sizeMask;
	for (;;) {
	  auto tags = buckets[bucketIdx].loadtags;
	  auto hits = CpuBucket::matchTags(tags, tag);
	  while (hits) {
	    auto idx = bits::getAndClearLastSetBit(hits);
	    auto row = buckets[bucketIdx].load<RowType>(idx);
	    if (ops.compare(row, i, probe)) {
	      ops.update(row, i, probe);
	      goto next;
	    }
	  }
	  auto misses = CpuBucket::matchTags(tags, 0);
	  if (misses) {
	    int32_t idx = bits::getAndClearLastSetBit(misses);
	    buckets[bucketIdx].tags[idx] = tag;
	    auto* newRow = ops.newRow(this);
	    buckets[bucketIdx].store(idx, newRow);
	    break;
	  }
	  bucketIdx = (bucketIdx + 1) & sizeMask;
	}
      done: ;
      }
    }

  };



  
}
