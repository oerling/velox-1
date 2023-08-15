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

#include "velox/experimental/wave/exec/WaveCore.cuh"

namespace facebook::velox::wave {


  template typename<T>
  func_kPlus(T left, T right) {
  return left + right;
}

  
  template <typename T>

  __device__ void filterKernel(const Filter& filter, int32_t blockBase, char* shared, int32_t& numRows) {
    auto* flagop = filter.flags;
  }
  

  
  __global__ void waveBaseKernel(ThreadBlockProgram** programs, int32_t* baseIndices, BlockStatus* blockStatusArray) {
  extern __shared__ __align__(alignof(ScanAlgorithm::TempStorage)) char smem[];
  auto* program = programs[blockIdx.x];
  auto* status = blockStatusArray[blockIdx.x];
  int32_t blockBase = blockIdx.x - baseIndices[blockIdx.x];
  for (auto i = 0; i < program->numInstructions; ++i) {
    auto instruction = program->instructions[i];
    switch (instruction->opCode) {
    case OpCode::kFilter:
      filterKernel(instruction->_.filter, blockBase, shared, control->numRows);
    }
  }
  }
  
  } // namespace facebook::velox::wave
