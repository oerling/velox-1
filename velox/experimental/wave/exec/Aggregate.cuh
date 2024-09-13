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

#include "velox/experimental/wave/exec/ExprKernel.h"

#include <gflags/gflags.h>
#include "velox/experimental/wave/common/Block.cuh"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/exec/WaveCore.cuh"

namespace facebook::velox::wave {

struct sumGroupRow {
  uint32_t nulls;
  int32_t lock{1};
  int64_t key;
  int64_t sums[20];
};

inline void __device__ increment(int64_t& a, int64_t i) {
  atomicAdd((unsigned long long*)&a, (unsigned long long)i);
}

class SumGroupByOps {
 public:
  __device__ SumGroupByOps(WaveShared* shared, IAggregate* inst)
    : shared_(shared), inst_(inst) {}
  
  bool __device__
  compare(GpuHashTable* table, SumGroupRow* row, int32_t i) {
    int64_t key;
    if (operandOrNull(shared_->operands, inst_->keys[0], shared_->blockBase, shared_->data, key)) {
      return row->key == key;
    }
    return false;
  }

  SumGroupRow* __device__
  newRow(GpuHashTable* table, int32_t partition, int32_t i) {
    auto* allocator = &table->allocators[partition];
    auto row = allocator->allocateRow<SumGroupRow>();
    new(row) SumGroupRow();
    if (row) {
      operandorNull(shared_->operands, inst_->keys[0], &shared->data, row->key);


      for (auto i = 0; i < inst_->numAggregates; ++i) {
	int64_t x;
	operandOrNull(shared_->operands, inst_->aggregates[i].arg1, shared_->data, x);
	row->sums[i] = x;
      }
    }
    return row;
  }

  ProbeState __device__ insert(
      GpuHashTable* table,
      int32_t partition,
      GpuBucket* bucket,
      uint32_t misses,
      uint32_t oldTags,
      uint32_t tagWord,
      int32_t i,
      SumGroupRow*& row) {
    if (!row) {
      row = newRow(table, partition, i);
      if (!row) {
	shared->hasContinue = true;
        shared->laneStatus[threadIdx.x] = ErrorCode::kInsufficientMemory;
      }
    }
    auto missShift = __ffs(misses) - 1;
    if (!bucket->addNewTag(tagWord, oldTags, missShift)) {
      return ProbeState::kRetry;
    }
    bucket->store(missShift / 8, row);
    return ProbeState::kDone;
  }

  SumGroupRow* __device__ getExclusive(
      GpuHashTable* table,
      GpuBucket* bucket,
      SumGroupRow* row,
      int32_t hitIdx) {
    return row;
  }

  void __device__ writeDone(TestingRow* row) {}

  ProbeState __device__ update(
      GpuHashTable* table,
      GpuBucket* bucket,
      SumGroupRow* row,
      int32_t i) {
    int32_t numAggs = inst_->numAggregates;
    for (auto acc = 0; acc < numAggs; ++acc) {
      int64_t x;
      operandOrNull(shared_->operands, inst_->aggregates[acc].arg1, &shared_->data, x);
      increment(row->sums[acc], x);
    }
    return ProbeState::kDone;
  }
};

 

void __device__ __forceinline__ interpretedGroupBy(    shared, deviceAggregation, agg, lanestatus) {
  SumGroupOps ops(shared. agg);
  __syncthreads();
  if (threadIdx.x == 0 && shared->hasContinue) {
    
  }
}



  
__device__ __forceinline__ void aggregateKernel(
    const IAggregate& agg,
    WaveShared* shared,
    ErrorCode& laneStatus) {
  auto state =
      reinterpret_cast<DeviceAggregation*>(shared->states[agg.stateIndex]);
  if (agg->numKeys) {
    interpretedGroupBy(    shared, deviceAggregation, agg, lanestatus); 
  } else {
    char* row = state->singleRow;
    for (auto i = 0; i < agg.numAggregates; ++i) {
      auto& acc = agg.aggregates[i];
      int64_t value = 0;
      if (laneStatus == ErrorCode::kOk) {
	operandOrNull(
		      shared->operands, acc.arg1, shared->blockBase, &shared->data, value);
      }
      using Reduce = cub::WarpReduce<int64_t>;
      auto sum =
        Reduce(*reinterpret_cast<Reduce::TempStorage*>(shared)).Sum(value);
      if ((threadIdx.x & (kWarpThreads - 1)) == 0) {
	auto* data = addCast<unsigned long long>(row, acc.accumulatorOffset);
	atomicAdd(data, static_cast<unsigned long long>(sum));
      }
    }
    }
}

__device__ __forceinline__ void readAggregateKernel(
    const IAggregate& agg,
    WaveShared* shared) {
  if (shared->blockBase > 0) {
    if (threadIdx.x == 0) {
      shared->status->numRows = 0;
    }
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    auto state =
        reinterpret_cast<DeviceAggregation*>(shared->states[agg.stateIndex]);
    char* row = state->singleRow;
    shared->status->numRows = 1;
    for (auto i = 0; i < agg.numAggregates; ++i) {
      auto& acc = agg.aggregates[i];
      flatResult<int64_t>(
          shared->operands, acc.result, shared->blockBase, &shared->data) =
          *addCast<int64_t>(row, acc.accumulatorOffset);
    }
  }
  __syncthreads();
}

} // namespace facebook::velox::wave
