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

#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/dwio/decode/GpuDecoder.cuh"

namespace facebook::velox::wave {

int32_t GpuDecode::sharedMemorySize() {
  return detail::sharedMemorySizeForDecode<kBlockSize>(decodeStep);
}

/// Describes multiple sequences of decode ops. Each TB executes a sequence of
/// decode steps. The data area starts with a range of instruction numbers for
/// each thread block. The first TB runs from 0 to ends[0]. The nth runs from
/// ends[nth-1] to ends[nth]. After gridDim.x ends, we round to an 8 aligned
/// offset and have an array of GpuDecodes.]
struct GpuDecodeParams {
  // If need to represent more than this many ops, use a dynamically allocated
  // external array in 'external'.
  static constexpr int32_t kMaxInlineOps =
      (sizeof(GpuDecodeParams) / (sizeof(GpuDecode) + sizeof(int32_t))) - 1;

  // Pointer to standalone description of work. If nullptr, the description of
  // work fits inline in 'this'.
  GpuDecodeParams* external{nullptr};
  // The end of each decode program. The first starts at 0. The end is
  // ends[blockIdx.x].
  int32_t ends[100 * sizeof(GpuDecode) / sizeof(int32_t)] = {};
};

__global__ void decodeKernel(GpuDecodeParams inlineParams) {
  GpuDecodeParams* params =
      inlineParams.external ? inlineParams.external : &inlineParams;
  int32_t programStart = blockIdx.x == 0 ? 0 : params.ends[blockIdx.x - 1];
  int32_t programEnd = params.ends[blockIdx.x];
  GpuDecode* ops =
      reinterpret_cast<GpuDecode*>(&&params.starts[0] + roundUp(gridDim.x, 2));
  for (i = programStart; i < programEnd; ++i) {
    decodeSwitch<kBlockSize>(ops[i]);
  }
}

void launchDecode(
		const DecodePrograms programs,
    GpuArena* arena,
    WaveBufferPtr& extra,
    Stream* stream) {
  int32_t numBlocks = programs.programs.size();
  int32_t numOps = 0;
  int32_t shared = 0;
  for (auto& program : programs) {
    numOps += program.size();
    for (auto& step : program) {
      shared = std::max(shared, step->sharedMemorySize());
    }
  }
  if (shared > 0) {
  shared += 15; // allow align at 16.
}
GpuDecodeParams localParams;
  GpuDecodeParams = &localParams;
  if (numOps > GpuDecodeParams::kMaxInline) {
    extra = arena->allocate<char>(
        (numBlocks + 1) * (sizeof(GpuDecode) + sizeof(int32_t)));
    params = extra->as<GpuDecodeParams>();
  }
  int32_t end = programs[0].size();
  GpuDecode* decodes =
      reinterpret_cast<GpuDecode*>(&params->ends[0] + roundUp(numBlocks, 2));
  int32_t fill = 0;
  for (auto i = 0; i < programs.size(); ++i) {
    params->ends[i] = (i == 0 ? 0 : params[i - 1]) + programs[i].size();
    for (auto& op : programs[i]) {
      ops[fill++] = *op;
    }
  }
  if (extra) {
    inlineParams.external = params;
  }

  decodeKernel<<<numBlocks, kBlockSize, shared, stream->stream>>>(inlineParams);
  CUDA_CHECK(cudaGetLastError());
  if (program.result) {
    if (!program.hostResult) {
      stream->prefetch(nullptr, program.result->as<char>(), program.result->size());
    } else {
      stream->deviceToHostAsync(program.hostResult->as<char>(), program.result->as<char>(), program.hostResult->size());
    }
}
}

} // namespace facebook::velox::wave
