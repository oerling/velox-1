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

#include "velox/experimental/wave/exec/AggregateGen.h"

namespace facebook::velox::exec {

struct SumGroupRow {
  uint32_t lock;
  uint32_t nulls;
  int64_t key;
  int32_t accNulls;
  int64_t sums[20];
};

inline void __device__ increment(int64_t& a, int64_t i) {
  atomicAdd((unsigned long long*)&a, (unsigned long long)i);
}

class SumGroupByOps {
 public:
  __device__ SumGroupByOps(WaveShared* shared, const IAggregate* inst)
      : shared_(shared), inst_(inst) {}

  uint64_t __device__ hash(int32_t i) {
    int64_t key;
    if (operandOrNull(
            shared_->operands,
            *reinterpret_cast<int16_t*>(
                &inst_->aggregates[inst_->numAggregates]),
            shared_->blockBase,
            key)) {
      constexpr uint64_t kMul = 0x9ddfea08eb382d69ULL;
      return kMul * key;
    }
    return 1;
  }

  uint64_t __device__ hashRow(SumGroupRow* row) {
    constexpr uint64_t kMul = 0x9ddfea08eb382d69ULL;
    return kMul * row->key;
  }

  bool __device__ compare(GpuHashTable* table, SumGroupRow* row, int32_t i) {
    int64_t key;
    auto k =
        asDeviceAtomic<int64_t>(&row->key)->load(cuda::memory_order_consume);
    if (operandOrNull(
            shared_->operands,
            *reinterpret_cast<int16_t*>(
                &inst_->aggregates[inst_->numAggregates]),
            shared_->blockBase,
            key)) {
      return k == key;
    }
    return false;
  }

  SumGroupRow* __device__
  newRow(GpuHashTable* table, int32_t partition, int32_t i) {
    auto* allocator = &table->allocators[partition];
    auto row = allocator->allocateRow<SumGroupRow>();
    if (row) {
      for (auto i = 0; i < inst_->numAggregates; ++i) {
        row->sums[i] = 0;
      }
      int64_t k;
      operandOrNull(
          shared_->operands,
          *reinterpret_cast<int16_t*>(&inst_->aggregates[inst_->numAggregates]),
          shared_->blockBase,
          k);
      asDeviceAtomic<int64_t>(&row->key)->store(k, cuda::memory_order_release);
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
        return ProbeState::kNeedSpace;
      }
    }
    auto missShift = __ffs(misses) - 1;
    if (!bucket->addNewTag(tagWord, oldTags, missShift)) {
      return ProbeState::kRetry;
    }
    bucket->store(missShift / 8, row);
    increment(table->numDistinct, 1);
    return ProbeState::kDone;
  }

  void __device__ addHostRetry(int32_t i) {
    shared_->hasContinue = true;
    shared_->status[i / kBlockSize].errors[i & (kBlockSize - 1)] =
        ErrorCode::kInsufficientMemory;
  }

  void __device__
  freeInsertable(GpuHashTable* table, SumGroupRow* row, uint64_t h) {
    int32_t partition = table->partitionIdx(h);
    auto* allocator = &table->allocators[partition];
    allocator->markRowFree(row);
  }

  SumGroupRow* __device__ getExclusive(
      GpuHashTable* table,
      GpuBucket* bucket,
      SumGroupRow* row,
      int32_t hitIdx) {
    return row;
  }

  void __device__ writeDone(SumGroupRow* row) {}

  ProbeState __device__
  update(GpuHashTable* table, GpuBucket* bucket, SumGroupRow* row, int32_t i) {
    int32_t numAggs = inst_->numAggregates;
    for (auto acc = 0; acc < numAggs; ++acc) {
      int64_t x;
      operandOrNull(
          shared_->operands,
          inst_->aggregates[acc].arg1,
          shared_->blockBase,
          x);
      increment(row->sums[acc], x);
    }
    return ProbeState::kDone;
  }

  WaveShared* shared_;
  const IAggregate* inst_;
};

 
  std::string makeAggregateRow(const AggregatProbe& probe) {
    std::stringstream out;
    out << "struct AggregateRow {\n"
	"  int32_t flags;\n" << std::endl;
    
    makeKeyMembers(probe.keys, out);
    int32_t numNullable = probe.keys.size() + probe.updates.size();
    auto numFlagWords = bits::roundUp(numNullable, 32) / 32;
    out << fmt::format("  nullFlags[{}];\n", numFlagWords);
    for (auto i = 0; i < probe.updates.size(); ++i) {
      out << probe.updates[i]->generateMember(state, probe, probe.updates[i]) << std::endl;  
    }
    out << "};\n\n";
    return out.str();
  }
    
  void makeAggregateClass(CompileState& state, const AggregateProbe* probe, bool forRead) {
    auto& out = state.inlines();
    out << makeAggregateRow(state, probe);

    out << "class AggregateFuncs {\n";
    
    out << "};\n\n";
  }

  
}








