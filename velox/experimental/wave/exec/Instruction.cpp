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

#include "velox/experimental/wave/exec/Wave.h"

namespace facebook::velox::wave {

std::string rowTypeString(const Type& type) {
  return "";
}

void AbstractAggregation::reserveState(InstructionStatus& reservedState) {
  instructionStatus = reservedState;
  // A group by produces 8 bytes of grid level state and uses the main main
  // BlockStatus for lane status.
  reservedState.gridState += sizeof(AggregateReturn);
}

  int32_t countErrors(BlockStatus* status, int32_t numBlocks, ErrorCode error) {
    for (auto i = 0; i < numBlocks; ++i) {
      for (auto j = 0; j < status[i].numRows; ++j) {
	count += status[i].errors[j] == error;
      }
    }
    return count;
  }

  void restockAllocator(AggregateOperatorState* state, GpuArena& arena, int32_t size, HashPartitionAllocator* allocator) {
    if (allocator->ranges[0].fixedFull) {
      state->ranges.push_back(allocator->ranges[0]);
      allocator->ranges[0] = std::move(allocator->ranges[1]);
    }
    auto buffer = arena.allocate<char>(size);
    state.buffers.push_back(buffer);
    AllocationRange newRange = AllocationRange(buffer->as<char>, size, size);
    if (allocator->ranges[0].empty()) {
      allocator.ranges[0] = std::move(newRange);
    } else {
      allocator->ranges[1] = std::move(newRange);
    }
  }

  
  void resupplyHashTable(WaveStream& stream, AbstractInstruction& inst) {
    auto* agg = inst->as<AbstractAggregation>();
    auto* deviceStream = WaveStream::streamFrommReserve();
    auto stateId = agg->stateId).value();
  auto* state = stream.operatorState(operatorState.stateId)->as<AggregateOperatorState>();
  auto* head = state->buffers.front()->as<DeviceAggregation>();
  auto* hashTable = reinterpret_cast<GpuHashTableBase*>(head + 1);
  auto* gridState = stream.gridState<AggregateStatus>(agg->status);
  BlockStatus* status = stream.hostBlockStatus();
  int32_t numBlocks = bits::roundUp(stream.numRows(), kBlockSize);
  int32_t numFailed = countErrors(blockStatus, numBlocks, ErrorCode::kInsufficientMemory);
  int32_t rowSize = agg->rowSize();
  int64_t newSize = bits::nextPowerOfTwo(numFailed + hashTable->numDistinct * 2);
  int64_t increment = rowSize * (newSize - hashTable->numDistinct) / numPartitions;
  int32_t numPartitions = hashTable->partitionMask + 1;
  for (auto i = 0; i < numPartitions; ++i) {
    auto* allocator = &reinterpret_cast<HashPartitionAllocator*>(hashTable + 1)[i];
    if (allocator->availableFixed() < increment) {
      restockAllocator(state, stream.arena(), increment, allocator); 
    }
  }
  bool rehash = false;
  WaveBufferPtr oldBuckets;
  int32_t numOldBuckets;
  if (gridState->numEntries > hashTable->maxEntries) {
    // Would need rehash.
    oldBuckets = state.buffers[1];
    numOldBuckets = hashTable->sizeMask + 1;
    state.buffers[1] = stream.arena().allocate<GpuBucketMembers>(newSize / GpuBucketMembers::kNumSlots);
    hashTable->sizeMask = (newSize / GpuBucketMembers::kNumSlots) - 1;
    hashTable->buckets = state.buffers[1]->as<GpuBucket>();
    rehash = true;
  }
  stream->prefetch(getDevice(), state.alignedHead, state.headSize);
  if (rehash) {
    AggregationControl control;
    control.oldBuckets = 
  }
}

AdvanceResult AbstractAggregation::canAdvance(
    WaveStream& stream,
    LaunchControl* control,
    OperatorState* state,
    int32_t instructionIdx) const {
  if (keys.empty()) {
    return {};
  }
  auto gridState = stream.gridStatus<AggregateReturn>(instructionStatus);
  if (gridState->numDistinct) {
    // The hash table needs memory or rehash. Request a Task-wide break to
    // resupply the device side hash table.
    return {
        .instructionIdx = instructionIdx,
        .isRetry = true,
        .syncDrivers =
            true, .updateStatus = resupplyHashTable, .reason = state};
  }
  return {};
}

AdvanceResult AbstractReadAggregation::canAdvance(
    WaveStream& stream,
    LaunchControl* control,
    OperatorState* state,
    int32_t programIdx) const {
  auto* aggState = reinterpret_cast<AggregateOperatorState*>(state);
  if (aggState->isNew) {
    aggState->isNew = false;
    return {.numRows = 1};
  }
  return {};
}

} // namespace facebook::velox::wave
