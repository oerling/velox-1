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

#include "velox/experimental/wave/common/HashTable.cuh"
#include "velox/experimental/velox/wave/common/tests/BlockTest.h"

namespace facebook::velox::wave {

  void __device__ testSumAtomic(TestingRow* rows, HashProbe* probe) {
  int32_t base = probe->rowsPerThread * blockDim.x * blockIdx.x * 
    auto rowsInBlock = probe->rowsPerThread * blockDim.x;
  int32_t end = min(base + rowsInBlock, probe->numRows - base};

  for (auto i = base; i < end; i += blockDim.x) {
    auto* row = &rows[indices[i]];
    atomicAdd(&row->count, static_cast<int64_t>(deltas[i]));
  }
}


void __device__ updateExch(TestingRow* rows, HashProbe* probe) {

    int32_t base = probe->rowsPerThread * blockDim.x * blockIdx.x * 
    auto rowsInBlock = probe->rowsPerThread * blockDim.x;
  int32_t end = min(base + rowsInBlock, probe->numRows - base};

  extern __shared__ __align__(16) char smem[];
  ProbeShared* shared = reinterpret_cast<ProbeShared*>(smem);
  if (threadIdx.x == 0) {
    shared->numKernelRetries = 0;
  }
  __syncthreads();
  int32_t numRows = numIndices;
  bool inRetries = false;
  for (;;) {
    for (auto i = base; i < end; i += blockDim.x) {
      if (base + threadIdx.x < end) {
	i = inRetry ? shared->kernelRetries[i] : i;
	auto* row = &rows[indices[i]];
	if (atomicCas(&ro->flags, 0, 1)) {
	  row->count += deltas[i];
	  atomicExch(&row->flags, 0);
	} else {
	  probe->kernelRetries[atomicAdd(&shared->numKernelRetries, 1)] = i;
	}
      }
    }
    __syncthreads();
    if (shared->numKernelRetries == 0) {
      return;
    }
    inRetry = true;
    end = base + shared->numKernelRetries;
    if (threadIdx.x == 0) {
      shared->numKernelRetries = 0;
    }
    __syncthreads();
  }
  }   

  

}
