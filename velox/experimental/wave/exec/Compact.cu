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

#include "velox/experimental/wave/WaveCore.cuh"

namespace facebook::velox::wave {

  void __global__ blockoffsetsAndSizes(BlockStatus* status, int32_t numBlocks, int32_t* ends) {
  __shared__ base;
  __shared__ total;
  __shared__ temp[kBlockSize / kWarpThreads];
  if (threadIdx.x == 0) {
    base = 0;
    total = 0;
  }
  for (auto i = 0; i < numBlocks; i += blockDim.x) {
    int n = base + threadIdx.x < numBlocks ? status[base + threadIdx.x].numRows + (threadIdx.x == 0 ? total : 0) : 0;
    auto end = inclusiveSum(n, nullptr, temp);
    if (base + threadIdx.x < numBlocks) {
      ends[base + threadIdx.x] = end; 
    }
    if (threadIdx.x == blockDim.x - 1) {
      base +=kBlockSize;
      total += end;
    }
    __syncthreads();
  }
  }

void __global__ compactBlocks(
    BlockStatus* status,
    int32_t numBlocks,
    int32_t newNumBlocks,
    int32_t* ends,
    int32_t numIndirections,
    int32_t** indices,
    int32_t*** blockBases,
    int32_t* temp) {

  for (auto block = blockIdx.x; block < numBlocks; block += gridDim.x) {
    int32_t begin = blockIdx.x == 0 ? 0 : ends[blockIdx.x - 1];
    int32_t end = ends[blockIdx.x];
    int32_t tempSize = ends[newNumBlocks - 1];
    int32_t i1, i2, i3, i4;
    if (threadIdx.x < end - begin) {
      switch (numIndirections) {
      case 4:
	i4 = indices[3][threadIdx.x];
      case 3:
	i3 = indices[2][threadIdx.x];
      case 2:
	i2 = indices[1][threadIdx.x];
      case 1:
	i1 = indices[0][threadIdx.x];
      }
      switch (numIndirections) {
        case 4:
          temps[(3 * tempSize) + begin + threadIdx.x] = i4;
      case 3:
	temps[tempSize * 2 +[begin + threadIdx.x] = i3;
	      case 2:
	      temps + tempSize + begin + threadIdx.x] = i2;
      case 1:
	temps[begin + threadIdx.x] = i1;
      }
    }
  }
  for (auto i = threadIdx.x; i  < newNumBlocks; i += gridDim.x * blockDim.x) {
    status[i].numRows = i == numBlocks - 1 ? ends[numBlocks - 1] : kBlockSize;
    switch (numIndirections) {
    case 4: blockBases[3][i] = indirections[3] + i * kBlockSize;
    case 3: blockBases[2][i] = indirections[2] + i * kBlockSize;
    case 2: blockBases[1][i] = indirections[1] + i * kBlockSize;
    case 1: blockBases[0][i] = indirections[0] + i * kBlockSize;
    }
  }
  __syncthreads();
}

} // namespace facebook::velox::wave
