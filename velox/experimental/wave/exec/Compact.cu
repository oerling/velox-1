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

void __global__ blockoffsetsAndSizes(BlockStatus* status, int32_t* endOffset) {}

void __global__ compactBlocks(
    BlockStatus* status,
    int32_t* ends,
    int32_t numIndirections,
    int32_t** indicesArrays) {
  int32_t begin = blockIdx.x == 0 ? 0 : ends[blockIdx.x - 1];
  int32_t end = ends[blockIdx.x];
  for (auto start = 0; start < numIndirections; start += 4) {
    int32_t toGo = numIndirections - start;
    int32_t i0, i1, i2, i4;
    if (threadIdx.x < end - begin) {
      if (toGo > 4) {
        toGo = 4;
      }
      switch (toGo) {
        case 4:
          i4 = indicesArray[start + 3][threadIdx.x];
        case 3:
          i3 = indicesArray[start + 2][threadIdx.x];
        case 2:
          i2 = indicesArray[start + 1][threadIdx.x];
        case 1:
          i1 = indicesArray[start][threadIdx.x];
      }
    }
    __syncthreads();
    if (threadIdx.x < end - begin) {
      switch (toGo) {
        case 4:
          indicesArray[start + 3][begin + threadIdx.x] = i4;
        case 3:
          indicesArray[start + 2][begin + threadIdx.x] = i3;
        case 2:
          indicesArray[start + 1][begin + threadIdx.x] = i2;
        case 1:
          indicesArray[start][begin + threadIdx.x] = i1;
      }
    }
  }
  __syncthreads();
}

} // namespace facebook::velox::wave
