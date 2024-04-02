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

#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/tests/CudaTest.h"

namespace facebook::velox::wave {

__global__ void
addOneKernel(int32_t* numbers, int32_t size, int32_t stride, int32_t repeats) {
  auto index = blockDim.x * blockIdx.x + threadIdx.x;
  for (auto counter = 0; counter < repeats; ++counter) {
    for (; index < size; index += stride) {
      ++numbers[index];
    }
    __syncthreads();
  }
}

void TestStream::addOne(int32_t* numbers, int32_t size, int32_t repeats) {
  constexpr int32_t kWidth = 10240;
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > kWidth / kBlockSize) {
    stride = kWidth;
    numBlocks = kWidth / kBlockSize;
  }
  addOneKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void addOneWideKernel(WideParams params) {
  auto index = blockDim.x * blockIdx.x + threadIdx.x;
  auto numbers = params.numbers;
  auto size = params.size;
  auto repeat = params.repeat;
  auto stride = params.stride;
  for (auto counter = 0; counter < repeat; ++counter) {
    for (; index < size; index += stride) {
      ++numbers[index];
    }
  }
}

void TestStream::addOneWide(int32_t* numbers, int32_t size, int32_t repeat) {
  constexpr int32_t kWidth = 10240;
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > kWidth / kBlockSize) {
    stride = kWidth;
    numBlocks = kWidth / kBlockSize;
  }
  WideParams params;
  params.numbers = numbers;
  params.size = size;
  params.stride = stride;
  params.repeat = repeat;
  addOneWideKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

  __device__ uint32_t scale32(uint32_t n, uint32_t scale) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(n)) *
                  scale) >> 32;
  }
  
__global__ void addOneRandomKernel(
    int32_t* numbers,
    const int32_t* lookup,
    uint32_t size,
    int32_t stride,
    int32_t repeats) {
  auto index = blockDim.x * blockIdx.x + threadIdx.x;
  for (uint32_t counter = 0; counter < repeats; ++counter) {
    for (; index < size; index += stride) {
      auto rnd = scale32(index * (counter + 1) * 1367836089, size);
      
      numbers[index] += lookup[rnd];
    }
    __syncthreads();
  }
}

void TestStream::addOneRandom(
    int32_t* numbers,
    const int32_t* lookup,
    int32_t size,
    int32_t repeats) {
  constexpr int32_t kWidth = 10240;
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > kWidth / kBlockSize) {
    stride = kWidth;
    numBlocks = kWidth / kBlockSize;
  }
  addOneRandomKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, lookup, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}
__device__ inline uint64_t hashMix(const uint64_t upper, const uint64_t lower) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (lower ^ upper) * kMul;
  a ^= (a >> 47);
  uint64_t b = (upper ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}
  void   __global__   __launch_bounds__(1024) makeInput(int32_t keyRange, int32_t startCount, uint8_t numColumns, int64_t** columns) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    columns[idx] = scale32(idx * 2017, keyrange);
    for (auto i = 1; i < numColumns; ++i) {
      columns[i][idx] = i;
    }
  }

  void TestStream::makeInput(int32_t numBlocks, int32_t keyRange, int32_t startCount, uint8_t numColumns, int64_t** columns) {
    makeInput<<<1024, numBlocks, 0, stream_->stream>>>(keyRange, startCount, numColumns, columns);
  }

  
  void __device__	updateAggs(int64_t* entry, uint16_t row, uint8_t numColumns, int64_t**args) {
    
  }
  
  void __global__ __launch_bounds__(1024) hashAndPartition8K(int64_t* keys, uint64_t* hashes, uint16_t partitions, uint16_t* rows) {
    auto base = blockIdx.x * 8192;
    for (auto stride = 0; stride < 8192; stride += 1024) {
      auto idx = base + stride + threadIdx.x;
      hashes[idx] = ashMix(1, keys[idx]); 
      rows[idx] = idx;
      partitions[idx] = hashes[idx] >> 40;
    }
    syncthreads();
    extern __shared__ __align__(16) char smem[];
    blockSort<1024, 8>([&](auto i) {
		return partitions[base + i];
	      },
    [&](auto i) { return rows[base + i];},
      partitions + base, rows + base, smem);
  }
  
  void TestStream::hashAndPartition8K(int32_t num8KBlocks, int64_t* keys, uint64_t* hashes, uint16_t partitions, uint16_t* rows) {
    hashAndPartition8K<<<1024, numBlocks, tempBytes, stream_->stream>>>(keys, hashes, partitions, rows);
  }
  void   __global__ __launch_bounds__(1024) update1K(int64_t* key, uint64_t* hash, uint16_t* partitions, uint16_t* rowNumbers, uint8_t numAggs, int64_t** args, MockProbe* probe, MockTable* table) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Indirection to access the keys, hashes, args.
    auto row = rows[idx];
    auto part = partition[idx];
    bool isLeader = idx == 0 || partition[idx - 1] != part;
    start = hash[row] & table->sizeMask;
    partEnd = (start & table->partitionMask) + table->partitionSize; 
    for (;;) {
      auto entry = table->rows[start];
      if (!entry) {
	// The test is supposed to only look for existing keys.
	assert(false);
	missed = true;
	break;
      }
      if (keys[row] == entry[0]) {
	hit = true;
	break;
      }
      start = (start + 1) + table->sizeMask;
    }
    probe.start[threadIdx.x] = start;
    probe.isHit[threadIdx.x]= isHit;
    __syncthreads();
    if (isLeader) {
      int32_t lane = threadIdx.x;
      for (;;) {
	updateAggs(table->rows[start], row, numColumns, args); 
      }
      if (lane >= blockDim.x - 1 || partition[lane + 1] != partition) {
	break;
      }
      start = probe.start[lane];
      row = rows[lane];
    }
  }

  void TestStream::update8K(int32_t num8KBlocks, int64_t* key, uint64_t* hash, uint16_t* partitions, uint16_t* rowNumbers, uint8_t numAggs, int64_t** args) {
  }

  
REGISTER_KERNEL("addOne", addOneKernel);
REGISTER_KERNEL("addOneWide", addOneWideKernel);
REGISTER_KERNEL("addOneRandom", addOneRandomKernel);  
} // namespace facebook::velox::wave
