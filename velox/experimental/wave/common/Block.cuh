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

#include <cub/block/block_scan.cuh>

/// Utilities for  booleans and indices and thread blocks.

namespace facebook::velox::wave {

  template <int32_t blockSize, cub::BlockScanAlgorithm Algorithm = cub::BLOCK_SCAN_RAKING>
__device__ inline void boolBlockToIndices(
    int32_t start,
    uint8_t* flags,
    int32_t* indices,
    int32_t* size,
    void* shmem) {
    typedef cub::BlockScan<int, blockSize, Algorithm> BlockScanT;

  auto* temp = reinterpret_cast<typename BlockScanT::TempStorage*>(shmem);
  int data[1];
  data[0] = flags[threadIdx.x];
  __syncthreads();
  int aggregate;
  BlockScanT(*temp).ExclusiveSum(data, data, aggregate);
  __syncthreads();
  if (flags[threadIdx.x]) {
    indices[data[0]] = threadIdx.x + start;
  }
  if (threadIdx.x == 0) {
    *size = aggregate;
  }
}



} // namespace facebook::velox::wave
