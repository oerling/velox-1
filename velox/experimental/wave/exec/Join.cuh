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

#include <velox/experimental/wave/exec/WaveCore.cuh"

namespace facebook::velox::wave {

struct JoinShared {
  HashJoinExpandGridStatus* gridStatus;
  HashJoinExpandBlockStatus* blockstatus;
  int32_t anyNext;
  int32_t temp[kBlockSize / 32];
}

Joinshared* inline __device__
joinShared(Waveshared* shared) {
  return reinterpret_cast<JoinShared*>(&shared->data);
}

template <
    typename RowType,
    typename copyRow,
    int32_t indicesIdx,
    int32_t gridstateOffset,
    int32_t gridStateSize,
    int32_t blockStateOffset>
bool __device__ __forceinline__ joinResult(
    RowType*& hit,
    bool filterResult,
    bool joinContinue,
    WaveShared* shared,
    bool hasDuplicates,
    CopyRow copyRow) {
  if (threadIdx.x == 0) {
    auto* j = joinShared(shared);
    if (hasDuplicates) {
      j->gridstatus =
          gridstatus<hashJoinExpandGridStatus>(shared, gridstatusOffset);
      j->blockstatus = blockstatus(shared, gridstatusSize, blockStatusOffset);
      j->anyNext = 0;
    } else {
      j->gridStatus = nullptr;
      j->blockstatus = nullptr;
    }
  }
  if (!joinContinue) {
    auto nth = exclusiveSum<int32_t, kBlockSize>(
        filterResult, &shared->numRows, joinTemp(shared));
    if (filterResult) {
      copyRow(hit, nth);
      auto* indices =
          reinterpret_cast<int32_t*>(shared->operands[indicesIdx]->base);
      indices[shared->blockBase + nth] = shared->blockBase + threadIdx.x;
    }
    if (hasDuplicates) {
      RowType* next = nullptr;
      if (hit) {
        next = *hit->nextPtr();
        joinShared(shared)->next[threadIdx.x] = next;
        hit = next;
      }
      uint32_t flags = ballot_sync(0xffffffff, next != nullptr);
      if ((threadIdx.x & 31) == 0) {
        if (flags) {
          atomicOr(&joinShared(shared)->anyNext, flags);
        }
        // the grid-wide flag is set by one thread per warp if not already set.
        if (!asDeviceAtomic<int32_t>(
                 &joinShared(shared)->gridState->hasContinue)
                 .load(cuda::memory_order_relaxed)) {
          // The write goes to L2 as write through without any memory order.
          joinShared(shared)->gridState->hasContinue = true;
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
      copyRow(hit, row);
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
      !joinShared(shared)->gridStatus->hasContinuable) {
    joinShared(shared)->gridStatus->hasContinue = 1;
  }
  return shared->numRows < kBlockSize - 32 && joinShared(shared)->anyNext;
}

} // namespace facebook::velox::wave
