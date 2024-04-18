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

#include "velox/experimental/wave/common/HashTable.h"
#include "velox/experimental/wave/common/Hash.h"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include <cub/util_ptx.cuh>

namespace facebook::velox::wave {

struct RowAllocator {
  int32_t numUsed{0};
  int32_t numFreed{0};
  int32_t capacity{0};
  // Array of pointers to starts of preallocated rows.
  void** items{nullptr};
  void** freedItems{nullptr};
  char* strings{nullptr};
  uint32_t stringsFill{0};
  uint32_t stringsCapacity{0};
  
    template <typename T>
    T* __device__ allocateRow() {
      auto idx = atomicAdd(&numUsed, 1);
      if (idx < capacity) {
	return reinterpret_cast<T*>(items[idx]);
      }
      return nullptr;
    }

  void __device__ free(void* row) {
    auto idx = atomicAdd(&numFreed, 1);
    if (idx < capacity) {
      freedItems[idx] = row;
    }
  }

  template <typename T>
  T* __device__ allocate(int32_t cnt) {
    uint32_t size = sizeof(T) * cnt;
    if (!strings) {
      return nullptr;
    }
    auto offset = atomicAdd(&stringsFill, size);
    if (offset + size > stringsCapacity) {
      stringsFill = stringsCapacity;
      return nullptr;
    }
    return reinterpret_cast<T*>(strings + offset);
  }
};

  
  inline uint8_t __device__ hashTag(uint64_t h) {
    return 0x80 | (h >> 32);
  }

  struct GpuBucket : public GpuBucketMembers {
  
  template <typename RowType>
  inline RowType* __device__ load(int32_t idx) const {
    uint32_t low = reinterpret_cast<const uint32_t*>(&data)[idx];
    if (low == 0) {
      return nullptr;
    }
    uint32_t uptr = low;
    uptr |= static_cast<uint64_t>(data[idx + 8]);
    return reinterpret_cast<RowType*>(uptr);
  }

    inline void __device__ store(int32_t idx, void* ptr) {
          auto uptr = reinterpret_cast<uint64_t>(ptr);
      data[8 + idx] = uptr >> 32;
      reinterpret_cast<uint32_t*>(&data)[idx] = uptr;
      // The two writes must be seen together on other threads.
      __threadfence();
    }

    bool __device__ addNewTag(uint8_t tag, uint32_t oldTags, uint8_t tagShift) {
    uint32_t newTags = oldTags | ((static_cast<uint32_t>(tag) << tagShift));
    return (oldTags == atomicCAS(&tags, oldTags, newTags));
  }
  };
  
    
  /// Shared memory state for an updating probe.
  struct ProbeShared {
    uint32_t numKernelRetries;
    uint32_t numHostRetries;
    int32_t blockBase;
  };

class GpuHashTable : public GpuHashTableBase {
  public:
  static constexpr int32_t kExclusive = 1;
  
  static int32_t updatingProbeSharedSize() {
    return sizeof(ProbeShared);
  }

template <typename RowType, typename Ops>
void __device__ readOnlyProbe(HashProbe* probe, Ops ops) {
  int32_t blockBase = ops.blockBase(probe);
  int32_t end = ops.numRowsInBlock(probe) + blockBase;
  for (auto i = blockBase + threadIdx.x; i < end; i += blockDim.x) {
    auto h = probe->hashes[i];
        uint32_t   tagWord = hashTag(h);
	tagWord |= tagWord << 8;
	    tagWord = tagWord | tagWord << 16;
    auto bucketIdx = h & sizeMask;
    RowType* hit;
    for (;;) {
      auto bucket = buckets + bucketIdx;
      auto tags = bucket->tags;
      auto hits = __vcmpeq4(tags, tagWord) & 0x01010101;
      while (hits) {
	auto hitIdx = (__ffs(hits) - 1) / 8;
	hit = bucket->load<RowType>(hitIdx);
	if (ops.compare(this, i, probe, hit)) {
	  goto done;
	}
	hits = hits & (hits - 1);
      }
      if (__vcmpeq4(tags, 0)) {
	hit = nullptr;
	break;
      }
      bucketIdx = (bucketIdx + 1) & sizeMask;
    }
  done: ;
    
  }
}

template <typename RowType, typename Ops>
void __device__ updatingProbe(HashProbe* probe, Ops ops) {
  extern __shared__ __align__(16) char smem[];
  auto* sharedState = reinterpret_cast<ProbeShared*>(smem);
  if (threadIdx.x == 0) {
    sharedState->numKernelRetries = 0;
    sharedState->numHostRetries = 0;
    sharedState->blockBase = ops.blockBase(probe);
  }
  __syncthreads();
  auto lane = cub::LaneId();
  constexpr int32_t kWarpThreads = 1 << CUB_LOG_WARP_THREADS(0);
  auto warp = threadIdx.x / kWarpThreads;
  bool doingRetries = false;
  int32_t numKeys = ops.numRowsInBlock(probe);
  for (;;) {
    for (auto count = 0; count < numKeys; count += blockDim.x) {
      auto i = count + threadIdx.x;
      int32_t numInBlock = numKeys - count;
    if (i >= numInBlock) {
      break;
    }
    uint32_t laneMask = warp  * kWarpThreads + kWarpThreads > numInBlock ? lowMask<uint32_t>(numInBlock - warp * kWarpThreads) : 0xffffffff;
    i += sharedState->blockBase;
    if (doingRetries) {
      i = probe->kernelRetries[i];
    }
    auto h = probe->hashes[i];
    uint32_t tagWord = hashTag(h);
    tagWord |= tagWord << 8;
    tagWord = tagWord | tagWord << 16;
    auto bucketIdx = h & sizeMask;
    uint32_t misses = 0;
    RowType* hit = nullptr;
    int32_t hitIdx;
    GpuBucket* bucket; 
    uint32_t tags;
    for (;;) {
      bucket = buckets + bucketIdx;
      tags = bucket->tags;
      auto hits = __vcmpeq4(tags, tagWord) & 0x01010101;
      while (hits) {
	hitIdx = (__ffs(hits) - 1) / 8;
	do {
	  // It could be somebody inserted the tag but did not fill in the pointer. The pointer is coming in a few clocks.
	  hit = bucket->load<RowType>(hitIdx);
	  // The thread expects to see writes from other threads.
	  __threadfence();
	} while (hit);
	if (ops.compare(this, probe, i, hit)) {
	  break;
	}
	hits = hits & (hits - 1);
      }
      if (hit) {
	break;
      }
      misses = __vcmpeq4(tags, 0);
      if (misses) {
	hit = nullptr;
	break;
      }
      bucketIdx = (bucketIdx + 1) & sizeMask;
    }
    // Every lane has a hit or a miss.
    uint32_t peers = __match_any_sync(laneMask, reinterpret_cast<int64_t>(hit));
      RowType* writable = nullptr;
      if (!hit) {
	// The missers divide by bucket.
	uint32_t sameBucket = __match_any_sync(peers, reinterpret_cast<int64_t>(bucket));
	// The leader is given by the lowest bit. This is processed first in sequence, so its tags etc are already in the lane's registers.
	int32_t leader = __ffs(sameBucket) - 1;
	auto insertBits = sameBucket;
	auto idxToInsert = i;
	ProbeState success = ProbeState::kDone;
	while (insertBits) {
	if (lane == leader) {
	  if (success == ProbeState::kDone) {
	    // The leader tries to insert the next from its cohort.
	    success = ops.insert(this, bucket, misses, tags, tagWord, idxToInsert, probe);
	  }
	  if (success != ProbeState::kDone) {
	    if (success == ProbeState::kNeedSpace) {
	      addHostRetry(sharedState, idxToInsert, probe);
	    } else {
	      // The lane goes into retries with the rest of cohort.
	      addKernelRetry(sharedState, idxToInsert, probe);
	    }
	  }
	}
	insertBits &= insertBits - 1;
	if (insertBits) {
	  // The other threads hitting the same bucket get retried. They could be inserts or hits of each other.
	  success = ProbeState::kRetry;
	}
	}
      } else {
	int32_t leader = (kWarpThreads - 1) - __clz(peers);
	RowType* writable = nullptr;
	if (lane == leader) {
	writable = ops.getExclusive(this, bucket, hit, hitIdx, warp);
	}
	auto toUpdate = peers;
	ProbeState success = ProbeState::kDone;
	while (toUpdate) {
	  auto peer = __ffs(toUpdate) - 1;
	  auto idxToUpdate = __shfl_sync(peers, i, peer);
	  if (lane == leader) {
	    if (success == ProbeState::kDone) {
	      success = ops.update(this, bucket, writable, idxToUpdate, probe);
	    }
	    if (success != ProbeState::kDone) {
	      if (success == ProbeState::kNeedSpace) {
		addHostRetry(sharedState, idxToUpdate, probe);
	      } else {
		addKernelRetry(sharedState, idxToUpdate, probe);
	      }
	    }
	  }
	  toUpdate &= toUpdate - 1;
	}
      }
    }
    __syncthreads();
    // All probes have had one try.
    if (sharedState->numKernelRetries == 0) {
      return;
    }
    numKeys = sharedState->numKernelRetries;
    doingRetries = true;
    if (threadIdx.x == 0) {
      sharedState->numKernelRetries = 0;
    }
    __syncthreads();
  }
}

  int32_t __device__ partitionIdx(int32_t bucketIdx) const {
    return (bucketIdx & partitionMask) >> partitionShift;
  }
  

private:
  static void __device__ addKernelRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->kernelRetries[shared->blockBase + atomicAdd(&shared->numKernelRetries, 1)] = i;
  }

  static void __device__ addHostRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->hostRetries[shared->blockBase + atomicAdd(&shared->numHostRetries, 1)] = i;
  }
};
}
