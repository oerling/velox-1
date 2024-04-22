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
#include "velox/experimental/wave/common/tests/BlockTest.h"

namespace facebook::velox::wave {

  void __device__ testSumAtomic(TestingRow* rows, HashProbe* probe) {
    auto keys = reinterpret_cast<int64_t**>(probe->keys);
    auto indices = keys[0];
    auto deltas = keys[1];
    int32_t base = probe->numRowsPerThread * blockDim.x * blockIdx.x;
    auto rowsInBlock = probe->numRowsPerThread * blockDim.x;
  int32_t end = probe->numRows[blockIdx.x];

  for (auto i = base + threadIdx.x; i < end; i += blockDim.x) {
    auto* row = &rows[indices[i]];
    atomicAdd((unsigned long long*)&row->count, (unsigned long long)deltas[i]);
  }
}


void __device__ updateExch(TestingRow* rows, HashProbe* probe) {

  int32_t base = probe->numRowsPerThread * blockDim.x * blockIdx.x;
    auto rowsInBlock = probe->numRowsPerThread * blockDim.x;
    int32_t end = base + probe->numRows[blockIdx.x];
    auto keys = reinterpret_cast<int64_t**>(probe->keys);
    auto indices = keys[0];
    auto deltas = keys[1];

  extern __shared__ __align__(16) char smem[];
  ProbeShared* shared = reinterpret_cast<ProbeShared*>(smem);
  if (threadIdx.x == 0) {
    shared->numKernelRetries = 0;
  }
  __syncthreads();
  bool inRetries = false;
  for (;;) {
    for (auto counter = base; counter < end; counter += blockDim.x) {
      auto i = counter + threadIdx.x;
      if (i < end) {
	i = inRetries ? probe->kernelRetries[i] : i;
	auto* row = &rows[indices[i]];
	if (atomicCAS(&row->flags, 0, 1) == 0) {
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
    inRetries = true;
    end = base + shared->numKernelRetries;
    if (threadIdx.x == 0) {
      shared->numKernelRetries = 0;
    }
    __syncthreads();
  }
  }   

  

}
