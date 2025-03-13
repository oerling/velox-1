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

#include "velox/experimental/wave/common/HashTable.cuh"
#include "velox/experimental/wave/exec/WaveCore.cuh"

namespace facebook::velox::wave {

struct JoinShared {
  HashJoinExpandGridStatus* gridStatus;
  HashJoinExpandBlockStatus* blockStatus;
  int32_t anyNext;
  int32_t temp[kBlockSize / 32];
};

inline __device__ JoinShared* joinShared(WaveShared* shared) {
  return reinterpret_cast<JoinShared*>(&shared->data);
}

template <int32_t gridStatusSize, int32_t blockStatusOffset>
int64_t __device__ loadJoinNext(WaveShared* shared) {
  auto* status = blockStatus<HashJoinExpandBlockStatus>(
      shared, gridStatusSize, blockStatusOffset);
  return reinterpret_cast<int64_t>(status->next[threadIdx.x]);
}

template <
    typename RowType,
    int32_t indicesIdx,
    int32_t gridstatusOffset,
    int32_t gridStatusSize,
    int32_t blockStatusOffset>
bool __device__ __forceinline__ joinResult(
    int64_t& hitAsInt,
    bool filterResult,
    bool joinContinue,
    ErrorCode laneStatus,
    WaveShared* shared,
    bool hasDuplicates,
    int64_t* hitsAsInt) {
  RowType* hit = reinterpret_cast<RowType*>(hitAsInt);
  auto hits = reinterpret_cast<RowType**>(hitsAsInt);
  if (threadIdx.x == 0) {
    auto* j = joinShared(shared);
    if (hasDuplicates) {
      j->gridStatus =
          gridStatus<HashJoinExpandGridStatus>(shared, gridstatusOffset);
      j->blockStatus = blockStatus<HashJoinExpandBlockStatus>(
          shared, gridStatusSize, blockStatusOffset);
      j->anyNext = 0;
    } else {
      j->gridStatus = nullptr;
      j->blockStatus = nullptr;
    }
  }
  if (!joinContinue) {
    auto nth = exclusiveSum<int32_t, kBlockSize>(
        filterResult, &shared->numRows, joinShared(shared)->temp);
    if (filterResult) {
      hits[shared->blockBase + nth] = hit;
      auto* indices =
          reinterpret_cast<int32_t*>(shared->operands[indicesIdx]->base);
      indices[shared->blockBase + nth] = shared->blockBase + threadIdx.x;
    }
    if (hasDuplicates) {
      RowType* next = nullptr;
      if (hit) {
        next = *hit->nextPtr();
        joinShared(shared)->blockStatus->next[threadIdx.x] = next;
        hit = next;
      }
      uint32_t flags = __ballot_sync(0xffffffff, next != nullptr);
      if ((threadIdx.x & 31) == 0) {
        if (flags) {
          atomicOr(&joinShared(shared)->anyNext, flags);
        }
        // the grid-wide flag is set by one thread per warp if not already set.
        if (!asDeviceAtomic<int32_t>(
                 &joinShared(shared)->gridStatus->anyContinuable)
                 ->load(cuda::memory_order_relaxed)) {
          // The write goes to L2 as write through without any memory order.
          joinShared(shared)->gridStatus->anyContinuable = true;
        }
      }
      __syncthreads();
      return shared->numRows < kBlockSize - 32 &&
          joinShared(shared)->anyNext != 0;
    }
    return false;
  }
  // We come here when there are  places to fill above shared->numRows.
  bool laneFull = false;
  if (hit && filterResult) {
    auto row = atomicAdd(&shared->numRows, 1);
    if (row < kBlockSize) {
      auto* indices =
          reinterpret_cast<int32_t*>(shared->operands[indicesIdx]->base);
      indices[shared->blockBase + row] = shared->blockBase + threadIdx.x;
      hits[shared->blockBase + row] = hit;
    } else {
      laneFull = true;
    }
    if (!laneFull) {
      auto* next = *hit->nextPtr();
      joinShared(shared)->blockStatus->next[threadIdx.x] = next;
      hit = next;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && shared->numRows > kBlockSize) {
    shared->numRows = kBlockSize;
  }
  uint32_t flags = __ballot_sync(
      0xffffffff,
      joinShared(shared)->blockStatus->next[threadIdx.x] != nullptr);
  if ((threadIdx.x & 31) == 0) {
    if (flags) {
      atomicOr(&joinShared(shared)->anyNext, flags);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && joinShared(shared)->anyNext &&
      !joinShared(shared)->gridStatus->anyContinuable) {
    joinShared(shared)->gridStatus->anyContinuable = 1;
  }
  return shared->numRows < kBlockSize - 32 && joinShared(shared)->anyNext;
}

template <typename RowType, typename CopyRow>
void __device__ __forceinline__ joinRow(
    int64_t* hitsAsInt,
    ErrorCode laneStatus,
    WaveShared* shared,
    CopyRow copy) {
  if (laneStatus == ErrorCode::kOk) {
    auto* hits = reinterpret_cast<RowType**>(hitsAsInt);
    copy(hits[shared->blockBase + threadIdx.x], shared->blockBase + threadIdx.x);
  }
}

} // namespace facebook::velox::wave
