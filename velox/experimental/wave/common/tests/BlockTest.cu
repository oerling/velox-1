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

#include "velox/experimental/wave/common/Block.cuh"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/tests/BlockTest.h"

namespace facebook::velox::wave {

using ScanAlgorithm = cub::BlockScan<int, 256, cub::BLOCK_SCAN_RAKING>;

__global__ void boolToIndices(
    uint8_t** bools,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times) {
  extern __shared__ __align__(alignof(ScanAlgorithm::TempStorage)) char smem[];
  int32_t idx = blockIdx.x;
  // Start cycle timer
  clock_t start = clock();
  uint8_t* blockBools = bools[idx];
  boolBlockToIndices<256>(
      [&]() { return blockBools[threadIdx.x]; },
      idx * 256,
      indices[idx],
      smem,
      sizes[idx]);
  clock_t stop = clock();
  if (threadIdx.x == 0) {
    times[idx] = (start > stop) ? start - stop : stop - start;
  }
}

void BlockTestStream::testBoolToIndices(
    int32_t numBlocks,
    uint8_t** flags,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times) {
  CUDA_CHECK(cudaGetLastError());
  auto tempBytes = sizeof(typename ScanAlgorithm::TempStorage);
  boolToIndices<<<numBlocks, 256, tempBytes, stream_->stream>>>(
      flags, indices, sizes, times);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void boolToIndicesNoShared(
    uint8_t** bools,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times,
    void* temp) {
  int32_t idx = blockIdx.x;

  // Start cycle timer
  clock_t start = clock();
  uint8_t* blockBools = bools[idx];
  char* smem = reinterpret_cast<char*>(temp) +
      blockIdx.x * sizeof(typename ScanAlgorithm::TempStorage);
  boolBlockToIndices<256>(
      [&]() { return blockBools[threadIdx.x]; },
      idx * 256,
      indices[idx],
      smem,
      sizes[idx]);
  clock_t stop = clock();
  if (threadIdx.x == 0) {
    times[idx] = (start > stop) ? start - stop : stop - start;
  }
}

void BlockTestStream::testBoolToIndicesNoShared(
    int32_t numBlocks,
    uint8_t** flags,
    int32_t** indices,
    int32_t* sizes,
    int64_t* times,
    void* temp) {
  CUDA_CHECK(cudaGetLastError());
  boolToIndicesNoShared<<<numBlocks, 256, 0, stream_->stream>>>(
      flags, indices, sizes, times, temp);
  CUDA_CHECK(cudaGetLastError());
}

int32_t BlockTestStream::boolToIndicesSize() {
  return sizeof(typename ScanAlgorithm::TempStorage);
}

__global__ void sum64(int64_t* numbers, int64_t* results) {
  extern __shared__ __align__(
      alignof(cub::BlockReduce<int64_t, 256>::TempStorage)) char smem[];
  int32_t idx = blockIdx.x;
  blockSum<256>(
      [&]() { return numbers[idx * 256 + threadIdx.x]; }, smem, results);
}

void BlockTestStream::testSum64(
    int32_t numBlocks,
    int64_t* numbers,
    int64_t* results) {
  auto tempBytes = sizeof(typename cub::BlockReduce<int64_t, 256>::TempStorage);
  sum64<<<numBlocks, 256, tempBytes, stream_->stream>>>(numbers, results);
  CUDA_CHECK(cudaGetLastError());
}

/// Keys and values are n sections of 8K items. The items in each section get
/// sorted on the key.
void __global__ __launch_bounds__(1024)
    testSort(uint16_t** keys, uint16_t** values) {
  extern __shared__ __align__(16) char smem[];
  auto keyBase = keys[blockIdx.x];
  auto valueBase = values[blockIdx.x];
  blockSort<256, 32>(
      [&](auto i) { return keyBase[i]; },
      [&](auto i) { return valueBase[i]; },
      keys[blockIdx.x],
      values[blockIdx.x],
      smem);
}

void BlockTestStream::testSort16(
    int32_t numBlocks,
    uint16_t** keys,
    uint16_t** values) {
  auto tempBytes = sizeof(
      typename cub::BlockRadixSort<uint16_t, 256, 32, uint16_t>::TempStorage);

  testSort<<<numBlocks, 256, tempBytes, stream_->stream>>>(keys, values);
}

/// Calls partitionRows on each thread block of 256 threads. The parameters
/// correspond to 'partitionRows'. Each is an array subscripted by blockIdx.x.
void __global__ partitionShortsKernel(
    uint16_t** keys,
    int32_t* numKeys,
    int32_t numPartitions,
    int32_t** ranks,
    int32_t** partitionStarts,
    int32_t** partitionedRows) {
  partitionRows<256>(
      [&](auto i) { return keys[blockIdx.x][i]; },
      numKeys[blockIdx.x],
      numPartitions,
      ranks[blockIdx.x],
      partitionStarts[blockIdx.x],
      partitionedRows[blockIdx.x]);
}

void BlockTestStream::partitionShorts(
    int32_t numBlocks,
    uint16_t** keys,
    int32_t* numKeys,
    int32_t numPartitions,
    int32_t** ranks,
    int32_t** partitionStarts,
    int32_t** partitionedRows) {
  constexpr int32_t kBlockSize = 256;
  auto shared = partitionRowsSharedSize<kBlockSize>(numPartitions);
  partitionShortsKernel<<<numBlocks, kBlockSize, shared, stream_->stream>>>(
      keys, numKeys, numPartitions, ranks, partitionStarts, partitionedRows);
  CUDA_CHECK(cudaGetLastError());
}

/// An Ops parameter class to do group by.
class MockGroupByOps {
public:
  bool __device__ compare(HashTable* table, HashProbe* probe, int32_t i, TestingRow* row) {
    return row->key == reinterpret_cast<int64_t*>(probe->keys)[i];
  }

  TestingRow* __device__  newRow(Hashtable* table, bucketIdx) {
    int32_t part = (bucketIdx & table->partitionMask) >> table->partitionShift;
    auto row = allocateRow<TestingRow>(table->allocators[partition]);
  }
  
  ProbeState __device__ insert(hashTable* table GpuBucket* bucket, uint32_t& misses, uint32_t& oldTag, uint32_t tagWord, int32_t i, HashProbe* probe) {
    int32_t bucketIdx = table->buckets - bucket;
    auto row = newRow(table, bucketIdx);
    if (!row) {
      return ProbeState::kNeedSpace;
    }
    row->key = reinterpret_cast<int64_t*>(probe->keys)[i];
    probe->flags = testingRow::kExclusive;
    auto missShift = __ffs(misses) - 1;
    if (!bucket->addNewTag(tagWord, oldTags, missShift)) {
      allocator->free(row);
      return ProbeState::kRetry;
    }
    bucket->store(missShift / 8, row);
    oldTags = bucket->tags;
    misses = __vcmpeq(oldTags, 0);

    return ProbeState::kDone;
  }
};


void __global__ hashTestKernel(HashTable* table HashProbe* probe, BlockTestStream::hashCase mode) {
  switch (mode) {
  case HashCase::kBuild:
    genericProbe<TestingRow>(int32_t i, TestingRow* row) {
      if (!row) {
	return ProbeState::kRetry;
      }
      if (probe->keys[i] == row->key) {
	return ProbeState::kDone;
      }
      return probestate::kInit;
    },
      [&](int32_t i, HashBucket* bucket, int32_t tagIdx, uint32_t oldTags) {
	if (!addNewTag(hashtag(probe->hashes[i]), oldTags, bucket, tagIdx)) {
	  return ProbeState::kRetry;
	}
	auto* row = getNewRow(table, probe->hash[i]);
	if (!row) {
	  *(long*)0 = 0;
	} else {
	  setBucketPtr(bucket, tagIdx, row);
	}
      });
     [&](hashProbe* probe, int32_t i, void* row) -> ProbeState {
       if (probe->keys[i] == row->key) {
	 return ProbeState::kDone;
       }
       return ProbeState::kMiss;
     });,
   break;
    case HashCase::kGroup:
 case HashCase::kProbe:

    }
  }
  

  void BlockTestStream::hash(HashTable* table, HashProbe* probe, HashCase mode) {
    hashTestKernel<<<numBlocks, 256, shared, stream_->stream>>>(table, probe, mode);
    CUDA_CHECK(cudaGetLastError());
  }


  
REGISTER_KERNEL("testSort", testSort);
REGISTER_KERNEL("boolToIndices", boolToIndices);
REGISTER_KERNEL("sum64", sum64);
REGISTER_KERNEL("partitionShorts", partitionShortsKernel);

} // namespace facebook::velox::wave
