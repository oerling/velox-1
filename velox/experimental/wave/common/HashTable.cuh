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
#include <cub/util_ptx.cuh>

namespace facebook::velox::wave {

struct RowAllocator {
  int32_t numUsed{0};
  int32_t numFreed{0};
  int32_t capacity{0};
  // Array of pointers to starts of preallocated rows.
  void** items{nullptr};
  void** freedItems{nullptr};

    template <typename T>
    T* __device__ allocateRow() {
      auto idx = atomicAdd(&numUsed);
      if (idx < capacity) {
	return items[idx];
      }
      return nullptr;
    }

  void __device__ free(void* row) {
    auto idx = atomicAdd(&numFreed, 1);
    if (idx < capacity) {
      freedItems[idx] = row;
    }
  }
};

  
  inline uint8_t __device__ hashTag(uint64_t h) {
    return h >> 32;
  }

  struct GpuBucket : public GpuBucketMembers {
  
  template <typename RowType>
  inline RowType* __device__ load(int32_t idx) const {
    uint32_t low = reinterpret_cast<uint32_t*>(&data)[idx];
    if (low == 0) {
      return nullptr;
    }
    uint32_t uptr = low;
    uptr |= static_cast<uint64_t>(data[idx + 8]);
    return reinterpret_cast<RowType*>(uptr);
  }

    inline void __device__ store(BucketPtr* bucket, int32_t idx, void* row) {
      auto uptr = reinterpret_cast<uint64_t>(ptr);
      data[8 + idx] = uptr >> 32;
      // If the low 32 are seen then the high 16 must also be seen.
      __threadfence();
      reinterpret_cast<uint32_t*>(&data)[idx] = uptr;
    }

    bool __device__ addNewTag(uint8_t tag, uint32_t oldTags, uint8_t tagShift) {
    uint32_t nextTagWord = oldTags | ((static_cast<uint32_t>(tag) << tagShift));
    return (oldTags == atomicCAS(&tags, oldTags, newtags));
  }
  };
  
    
  /// Shared memory state for an updating probe.
  struct ProbeShared {
    uint32_t numKernelRetries;
    uint32_t numHostRetries;
  };

class GpuHashTable : public GpuHashTableMembers {
  public:
  
  static int32_t updatingProbeSharedSize() {
    return sizeof(ProbeShared);
  }

template <typename RowType, typename Ops>
void __device__ readOnlyProbe(hashProbe* probe, Ops ops) {
  int32_t numKeys = probe->numKeys;
  for (auto i = threadIdx.x; i < numKeys; i += blockDim.x) {
    auto h = probe->hashes[i];
    uint    32_t   tagWord = hashTag(h);
    tagword 	 |= tagWord << 8;
    uint32_t tagWord = tagWord | tagWord << 16;
    auto bucketIdx = h & sizeMask;
    for (;;) {
      auto bucket = buckets + bucketIdx;
      tags = bucket->tags;
      auto hits = __vcmpeq4(tags, tagWord) & 0x01010101;
      while (hits) {
	auto hitIdx = (__ffs(hits) - 1) / 8;
	auto hit = bucket->load<RowType>(hitIdx);
	if (ops.compare(this, i, probe, hit)) {
	  goto done;
	}
	hits = hits & (hits - 1);
      }
      misses = __vcmpeq4(tags, 0);
      if (misses) {
	break;
      }
      bucketIdx = (bucketIdx + 1) & sizeMask;
    }
  done: ;
  }
}

template <typename RowType, typename Ops>
void __device__ updatingProbe(hashProbe* probe, Ops ops) {
  extern __shared__ __align__(16) char smem[];
  ProbeShared* sharedState = reinterpret_cast<ProbeShared>(smem);
  if (threadIdx.x == 0) {
    sharedState->numKernelRetries = 0;
    sharedState->numHostRetries = 0;
  }
  __syncthreads();
  auto lane = cub::LaneId();
  constexpr int32_t kWarpThreads = 1 << CUB_LOG_WARP_THREADS(0);
  auto warp = threadIdx.x / kWarpThreads;
  bool doingRetries = false;
  int32_t numKeys = probe->numKeys;
  for (;;) {
    for (auto count = 0; count < numKeys; count += blockDim.x) {
      auto i = count + threadIdx.x;
      int32_t numInBlock = numKeys - count;
    if (i >= numinBlock) {
      break;
    }
    uint32_t laneMask = warp  * kWarpThreads + kWarpThreads > numInBlock ? lowMask<uint32_t>(numInBlock - warp * kWarpThreads) : 0xffffffff;
    if (doingRetries) {
      i = probe->kernelRetries[i];
    }
    auto h = probe->hashes[i];
    uint32_t tagWord = hashTag(h);
    tagword |= tagWord << 8;
    uint32_t tagWord = tagWord | tagWord << 16;
    auto bucketIdx = h & sizeMask;
    uint32_t misses = 0;
    RowType* hit = nullptr;
    for (;;) {
      auto bucket = buckets + bucketIdx;
      tags = bucket->tags;
      auto hits = __vcmpeq4(tags, tagWord) & 0x01010101;
      while (hits) {
	auto hitIdx = (__ffs(hits) - 1) / 8;
	do {
	  // It could be somebody inserted the tag but did not fill in the pointer. The pointer is coming in a few clocks.
	  hit = loadFromBucket<RowType>(bucket, hitIdx);
	  __threadfence();
	} while (hit);
	if (ops.compare(table, probe, i, hit)) {
	  break;
	}
	hits = hits & (hits - 1);
      }
      if (hit) {
	break;
      }
      misses = __vcmpeq(tags, 0);
      if (misses) {
	break;
      }
      bucketIdx = (bucketIdx + 1) & sizeMask;
    }
    // Every lane has a hit or a miss.
    uint32_t peers = __match_any_sync(laneMask, hit);
      RowType* writable = nullptr;
      if (!hit) {
	// The missers divide by bucket.
	uint32_t sameBucket = __match_any_sync(peers, bucket);
	// The leader is given by the lowest bit. This is processed first in sequence, so its tags etc are already in the lane's registers.
	int32_t leader = __ffs(sameBucket) - 1;
	auto insertBits = sameBucket;
	auto idxToInsert = i;
	ProbeState success = ProbeState::kDone;
	while (insertBits) {
	if (lane == leader) {
	  if (success == ProbeState::kDone) {
	    // The leader tries to insert the next from its cohort.
	    success = ops.insert(this, bucket, misses, oldTag, tagWord, idxToInsert, probe);
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
	  // There is another insert for the same bucket. It could be the same key as just was inserted.
	  peerLane = __ffs(insertBits) - 1;
  	  idxToInsert = __shfl_sync(sameBucket, i, peerLane);
	  tagWord = __shfl_sync(sameBucket, tagWord, peerlane);
	  hits = __vcmpeq(bucket->tags, tagWord) & 0x01010101; 
	  while (hits
	}
	}
      } else {
	int32_t leader = (kWarpThreads - 1) - __clz(peers);
	RowType* writable = nullptr;
	if (lane == leader) {
	writable = ops.getExclusive(this, bucket, currentHit, hitIdx, warp);
	}
	auto toUpdate = peers;
	probeState success = ProbeState::kDone;
	while (toUpdate) {
	  auto peer = __ffs(toUpdate) - 1;
	  auto idxToUpdate = __shfl_sync(peers, i, peer);
	  if (lane == leader) {
	    if (success == ProbeState::kDone) {
	      success = ops.update(writable, idxToUpdate, probe);
	    }
	    if (success != ProbeState::kDone) {
	      if (success == probeState::kNeedSpace) {
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
private:
  static void __device__ addKernelRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->retries[atomicAdd(&shared->numKernelRetries, 1)] = i;
  }

  static void __device__ addHostRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->hostRetries[atomicAdd(&shared->numHostRetries, 1)] = i;
  }

  
  

};
}
