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
#include "velox/experimental/wave/exec/Aggregate.cuh"

DEFINE_bool(kernel_gdb, false, "Run kernels sequentially for debugging");

namespace facebook::velox::wave {

template <typename T>
__device__ inline T opFunc_kPlus(T left, T right) {
  return left + right;
}

template <typename T, typename OpFunc>
__device__ __forceinline__ void binaryOpKernel(
    OpFunc func,
    IBinary& instr,
    Operand** operands,
    int32_t blockBase,
    char* shared,
    ErrorCode& laneStatus) {
  if (!laneActive(laneStatus)) {
    return;
  }
  T left;
  T right;
  if (operandOrNull(operands, instr.left, blockBase, shared, left) &&
      operandOrNull(operands, instr.right, blockBase, shared, right)) {
    flatResult<decltype(func(left, right))>(
        operands, instr.result, blockBase, shared) = func(left, right);
  } else {
    resultNull(operands, instr.result, blockBase, shared);
  }
}

__device__ void filterKernel(
    const IFilter& filter,
    Operand** operands,
    int32_t blockBase,
        WaveShared* shared,
    ErrorCode& laneStatus) {
  bool isPassed = laneActive(laneStatus);
  if (isPassed) {
    if (!operandOrNull(operands, filter.flags, blockBase, &shared->data, isPassed)) {
      isPassed = false;
    }
  }
  uint32_t bits = __ballot_sync(0xffffffff, isPassed);
  if (threadIdx.x == 0) {
    reinterpret_cast<int32_t*>(&shared->data)[threadIdx.x / kWarpThreads] = __popc(bits);
  }
  __syncthreads();
  if (threadIdx.x < kWarpThreads) {
    constexpr int32_t kNumWarps = kBlockSize / kWarpThreads;
    int32_t cnt = threadIdx.x < kNumWarps ? reinterpret_cast<int32_t*>(&shared->data)[threadIdx.x] : 0;
    int32_t sum;
    using Scan = cub::WarpScan<int32_t, kBlockSize / kWarpThreads>;
    Scan(*reinterpret_cast<Scan::TempStorage*>(shared)).ExclusiveSum(cnt, sum);
    if (threadIdx.x < kNumWarps) {
      if (threadIdx.x == kNumWarps - 1) {
	shared->numRows = cnt + sum;
      }
      reinterpret_cast<int32_t*>(&shared->data)[threadIdx.x] = sum;
      }
  }
  __syncthreads();
  if (bits & (1 << threadIdx.x & (kWarpThreads-1))) {
    auto* indices = reinterpret_cast<int32_t*>(operands[filter.indices]->base);
    auto start = reinterpret_cast<int32_t*>(&shared->data)[threadIdx.x / kWarpThreads];
    auto bit = start + __popc(bits & lowMask<uint32_t>(threadIdx.x & (kWarpThreads - 1)));
    indices[bit] = blockBase + threadIdx.x;
  }
  laneStatus = threadIdx.x < shared->numRows ? ErrorCode::kOk : ErrorCode::kInactive;
}

__device__ void wrapKernel(
    const IWrap& wrap,
    Operand** operands,
    int32_t blockBase,
    int32_t numRows,
    void* shared) {
  Operand* op = operands[wrap.indices];
  auto* filterIndices = reinterpret_cast<int32_t*>(op->base);
  if (filterIndices[blockBase + numRows - 1] == numRows + blockBase - 1) {
    // There is no cardinality change.
    return;
  }

  struct WrapState {
    int32_t* indices;
  };

  auto* state = reinterpret_cast<WrapState*>(shared);
  bool rowActive = threadIdx.x < numRows;
  for (auto column = 0; column < wrap.numColumns; ++column) {
    if (threadIdx.x == 0) {
      auto opIndex = wrap.columns[column];
      auto* op = operands[opIndex];
      int32_t** opIndices = &op->indices[blockBase / kBlockSize];
      if (!*opIndices) {
        *opIndices = filterIndices + blockBase;
        state->indices = nullptr;
      } else {
        state->indices = *opIndices;
      }
    }
    __syncthreads();
    // Every thread sees the decision on thred 0 above.
    if (!state->indices) {
      continue;
    }
    int32_t newIndex;
    if (rowActive) {
      newIndex =
          state->indices[filterIndices[blockBase + threadIdx.x] - blockBase];
    }
    // All threads hit this.
    __syncthreads();
    if (rowActive) {
      state->indices[threadIdx.x] = newIndex;
    }
  }
  __syncthreads();
}

#define BINARY_TYPES(opCode, OP)                             \
  case OP_MIX(opCode, WaveTypeKind::BIGINT):                 \
    binaryOpKernel<int64_t>(                                 \
        [](auto left, auto right) { return left OP right; }, \
        instruction->_.binary,                               \
        operands,                                            \
        blockBase,                                           \
        &shared->data,                                              \
        laneStatus);                                             \
    break;

  
__global__ void oneAggregate(
			       KernelParams params, int32_t pc, int32_t base) {
  PROGRAM_PREAMBLE(base);
  aggregateKernel(instruction[pc]._.aggregate, shared, laneStatus);
      PROGRAM_EPILOGUE();
}

__global__ void oneReadAggregate(
				 KernelParams params, int32_t pc, int32_t base);

  template <typename T>
__global__ void onePlus(
			       KernelParams params, int32_t pc, int32_t base) {
  PROGRAM_PREAMBLE(base);
    binaryOpKernel<T>(                                 
        [](auto left, auto right) { return left + right; }, 
        instruction->_.binary,                               
        operands,                                            
        blockBase,                                           
        &shared->data,                                              
        laneStatus);                                             
    PROGRAM_EPILOGUE();
}

    template <typename T>
__global__ void oneLt(
			       KernelParams params, int32_t pc, int32_t base) {
  PROGRAM_PREAMBLE(base);
    binaryOpKernel<T>(                                 
        [](auto left, auto right) { return left < right; }, 
        instruction->_.binary,                               
        operands,                                            
        blockBase,                                           
        &shared->data,                                              
        laneStatus);                                             
    PROGRAM_EPILOGUE();
}

  
  __global__ void oneFilter(
			       KernelParams params, int32_t pc, int32_t base) {
  PROGRAM_PREAMBLE(base);
  filterKernel(instruction[pc]._.filter, operands, blockBase, shared, laneStatus);
  wrapKernel(instruction[pc + 1]._.wrap, operands, blockBase, shared->numRows, &shared->data);
      PROGRAM_EPILOGUE();
}

  
  
__global__ void waveBaseKernel(
			       KernelParams params) {
  PROGRAM_PREAMBLE(0);
  for (;;) {
#if 0
    switch (instruction->opCode) {
      case OpCode::kReturn:
	PROGRAM_EPILOGUE();
        return;

    case OpCode::kFilter:
        filterKernel(
            instruction->_.filter,
            operands,
            blockBase,
	    shared,
	    laneStatus);
        break;

      case OpCode::kWrap:
        wrapKernel(
            instruction->_.wrap, operands, blockBase, shared->numRows, &shared->data);
        break;
    case OpCode::kAggregate:
      aggregateKernel(instruction->_.aggregate, shared);
      break;
    case OpCode::kReadAggregate:
      readAggregateKernel(instruction->_.aggregate, shared);
      break;
      BINARY_TYPES(OpCode::kPlus, +);
        BINARY_TYPES(OpCode::kLT, <);
    }
    #endif
    ++instruction;
  }
}

int32_t instructionSharedMemory(const Instruction& instruction) {
  switch (instruction.opCode) {
    case OpCode::kFilter:
      return sizeof(WaveShared) + (2 + (kBlockSize / kWarpThreads)) * sizeof(int32_t);
  default:
    return sizeof(WaveShared);
  }
}

#define CALL_ONE(k, params, pc, base) \
  k<<< \
      blocksPerExe,\
      kBlockSize,\
      sharedSize,\
      alias ? alias->stream()->stream : stream()->stream>>>(\
							    params, pc, base);


  void WaveKernelStream::callOne(
    Stream* alias,
    int32_t numBlocks,
    int32_t sharedSize,
    KernelParams& params) {
  int32_t blocksPerExe = 0;
  auto first = params.programIdx[0];
  for (; blocksPerExe < numBlocks; ++blocksPerExe) {
    if (params.programIdx[blocksPerExe] != first) {
      break;
    }
  }
  std::vector<std::vector<OpCode>> programs;
  for (auto i = 0; i < numBlocks; i += blocksPerExe) {
    auto programIdx = programs.size();
    programs.emplace_back();
    auto* instructions = params.programs[programIdx]->instructions;
    for (auto pc = 0; instructions[pc].opCode != OpCode::kReturn; ++pc) {
      programs.back().push_back(instructions[pc].opCode);
    }
  }
  auto initialStartPC = params.startPC;
  for (auto programIdx = 0; programIdx < programs.size(); ++programIdx) {
    auto& program = programs[programIdx];
    int32_t base = programIdx * blocksPerExe;
    params.startPC = initialStartPC;
    int32_t start = 0;
    if (params.startPC) {
      start = params.startPC[programIdx];
    }
    for (auto pc = start; pc < program.size(); ++pc) {
      switch (program[pc]) {
      case OpCode::kFilter:
	CALL_ONE(oneFilter, params, pc, base)
	  ++pc;
	break;
      case OpCode::kAggregate:
	CALL_ONE(oneAggregate, params, pc, base)
	  break;
      case OpCode::kReadAggregate:
	CALL_ONE(oneReadAggregate, params, pc, base)
	break;
      case OP_MIX(OpCode::kPlus, WaveTypeKind::BIGINT):
	CALL_ONE(onePlus<int64_t>, params, pc, base);
	break;
      case OP_MIX(OpCode::kLT, WaveTypeKind::BIGINT):
	CALL_ONE(oneLt<int64_t>, params, pc, base);
	break;
      }
    }
    params.startPC = nullptr;
  }
}


  
void WaveKernelStream::call(
    Stream* alias,
    int32_t numBlocks,
    int32_t sharedSize,
    KernelParams& params) {
  if (FLAGS_kernel_gdb) {
    callOne(alias, numBlocks, sharedSize, params);
    (alias ? alias : this)->wait();
    return;
  }

  waveBaseKernel<<<
      numBlocks,
      kBlockSize,
      sharedSize,
      alias ? alias->stream()->stream : stream()->stream>>>(
							    params);
}
  
REGISTER_KERNEL("expr", waveBaseKernel);

  void __global__ setupAggregationKernel(AggregationControl op) {
  //    assert(op.maxTableEntries == 0);
  auto* data = new(op.head) DeviceAggregation();
  data->rowSize = op.rowSize;
  data->singleRow = reinterpret_cast<char*>(data + 1);
  memset(data->singleRow, 0, op.rowSize);
}
  
  void WaveKernelStream::setupAggregation(AggregationControl& op) {
    setupAggregationKernel<<<1, 1, 0, stream_->stream>>>(op);
    wait();
  }

  
} // namespace facebook::velox::wave
