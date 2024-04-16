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

namespace facebook::velox::wave {

  inline uint8_t __device__ hashTag(uint64_t h) {
    return h >> 32;
  }

  template <typename RowType>
  inline RowType* __device__ loadFromBucket(Bucket* bucket, int32_t idx) {
    uint32_t low = reinterpret_cast<uint32_t*>(bucket)[idx + 2];
    if (low == 0) {
      return nullptr;
    }
    uint32_t uptr = low;
    uptr |= static_cast<uint64_t>(reinterpret_cast<uint16_t*>(bucket)[idx + 12]) << 32;
    return reinterpret_cast<RowType*>(uptr);
  }
  inline void __device__ setRowPtr(BucketPtr* bucket, int32_t idx, void* row) {
    auto uptr = reinterpret_cast<uint64_t>(ptr);
    reinterpret_cast<uint16_t*>(bucket)[idx + 12] = uptr >> 32;
    __threadfence();
    reinterpret_cast<uint32_t*>(bucket)[idx + 2] = uptr;
  }
 
  inline bool __device__ isFinalState(probeState state) {
    return state == ProbeState::kDone || state == ProbeState::kNeedSpace;
  }


  bool addNewTag(uint8_t tag, uint32_t oldTags, HashBucket* bucket, uint8_t tagShift) {
    uint32_t nextTagWord = oldTags | ((static_cast<uint32_t>(tag) << tagShift));
    return (oldTags != atomicCAS(&bucket->tags, oldTags, newtags));
  }

  struct ProbeShared {
    uint32_t numKernelRetries;
    uint32_t numHostRetries;
  };

  template <int32_t kBlockSize>
  int32_t probeSharedSize() {
    return sizeof(int32_t);
  }

  void __device__ addKernelRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->retries[atomicAdd(shared->numKernelRetries, 1)] = i;
  }

  void __device__ addHostRetry(ProbeShared* shared, int32_t i, HashProbe* probe) {
    probe->hostRetries[atomicAdd(shared->numHostRetries, 1)] = i;
  }

  
template <typename RowType, typename Comparer, typename Inserter>
void __device__ genericProbe(HashTable* table, hashProbe* probe, Comparer comparer, Inserter inserter) {
  extern __shared__ __align__(16) char smem[];

  ProbeShared* sharedState = reinterpret_cast<ProbeShared>(smem);
  if (threadIdx.x == 0) {
    sharedState->numKernelRetries = 0;
    sharedState->numHostRetries = 0;
  }
  __syncthreads();

  bool doingRetries = false;
  int32_t numKeys = probe->numKeys;
  for (;;) {
    for (auto count = 0; count < numKeys; count += blockDim.x) {
      
    auto i = count + threadIdx.x;
    if (i >= numProbes) {
      break;
    }
    auto h = probe->hashes[i];
    uint32_t tagWord = hashTag(h);
    tagword |= tagWord << 8;
    uint32_t tagWord = tagWord | tagWord << 16;
    auto bucketIdx = h & table->sizeMask;
    ProbeState state = kInit;
    for (;;) {
      auto bucket = table->buckets + bucketIdx;
      tags = bucket->tags;
      auto hits = __vcmpeq4(tags, tagWord) & 0x01010101;
      while (hits) {
	auto hitIdx = (__ffs(hits) - 1) / 8;
	state = comparer(probe, i, bucket);
	if (state == ProbeState::kDone) {
	  goto done;
	}
	if (state == ProbeState::kRetry) {
	  addKernelRetry(sharedstate, i, probe;
	  goto done;
	}
	break;

	hits = hits & (hits - 1);
      }
      misses = __vcmpeq(tags, 0);
      if (misses) {
	state = inserter(i, bucket, tags, misses);
	if (state == ProbeState::kDone) {
	  goto done;
	}
	if (state == ProbeState::kRetry) {
	  addRetry(sharedState, i, probe);
	  goto done;
	}
	if (state == ProbeState::kNeedSpace) {
	  addHostRetry(sharedState, i, probe);
	}
	break;
      }
      bucketIdx = (bucketIdx + 1) & table->sizeMask;
      }

    }
    done: ;

    }
    // All probes have had one try.
    __syncthreads();
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


}
