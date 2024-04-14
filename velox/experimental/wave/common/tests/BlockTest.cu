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


  template <int kBlockSize>
  int32_t partitionRowsSharedSize(int32_t numPartitions) {
    using Scan =  cub::BlockScan<int, kBlockSize>;
    auto scanSize = sizeof(typename Scan::TempStorage) + sizeof(int32_t);
    int32_t counterSize = sizeof(int32_t) * numPartitions;
    if (counterSize <= scanSize) {
      return scanSize;
    }
    static_assert(sizeof(typename Scan::TempStorage) >= sizeof(int32_t) * kBlockSize);
    return scanSize + counterSize - (kBlockSize - 2) * sizeof(int32_t);
  }



  /// Partitions a sequence of indices into runs where the indices
  /// belonging to the same partition are contiguous. Indices from 0
  /// to 'numKeys-1' are partitioned into 'partitionedRows', which
  /// must have space for 'numKeys' row numbers. The 0-based partition
  /// number for row 'i' is given by 'getter(i)'.  The row numbers for
  /// partition 0 start at 0. The row numbers for partition i start at
  /// 'partitionStarts[i-1]'.  If 'counters' is non-nullptr it is an
  /// array of 'numPartitions' counters that are expected to be
  /// initialized to 0. If it is nullptr, we use shared memory. There
  /// must be at least the amount of shared memory given by
  /// partitionSharedSize(numPartitions). If counters is non-nullptr
  /// 'numPartitions' to partitionSharedSize can be 0.  'ranks' is a
  /// temporary array of 'numKeys' elements.
  template <int32_t kBlockDimX, typename RowNumber, typename Getter>
  void __device__ partitionRows(Getter getter, uint32_t numKeys, uint32_t numPartitions, uint32_t* counters, RowNumber* ranks, RowNumber* partitionStarts, RowNumber* partitionedRows) {
    using Scan =  cub::BlockScan<int32_t, kBlockDimX>;
    constexpr int32_t kWarpThreads = CUB_LOG_WARP_THREADS(0);
    auto warp = threadIdx.x / kWarpThreads;
    auto lane = cub::LaneId();
    extern __shared__ __align__(16) char smem[];

    if (!counters) {
      // The first kBlockDimX - 2 counters overlap with the smem of the block scan. These will be in registers when the scan starts.
      counters = reinterpret_cast<uint32_t*>(smem + sizeof(typename Scan::TempStorage) - (kBlockDimX - 2) * sizeof(int32_t));
      for (auto i = 0; i < numPartitions; i += kBlockDimX) {
	counters[i] = 0;
      }
    }
    __syncthreads();
    for (auto start = 0; start < numKeys; start += kBlockDimX) {
      int32_t warpStart = start + warp * kWarpThreads;
      if (start >= numKeys) {
	break;
      }
      uint32_t laneMask = warpStart + kWarpThreads <= numKeys ? 0xffffffff : lowMask<uint32_t>(numKeys - warpStart);
      if (warpStart + lane < numKeys) {
	int32_t key = getter(warpStart + lane);
    	uint32_t mask = __match_any_sync(laneMask, key);
	int32_t leader = __ffs(mask) - 1;
	uint32_t cnt = __popc(mask & lowMask<uint32_t>(lane));
	uint32_t base;
	if (lane == leader) {
	  base = atomicAdd(&counters[key], cnt);
	}
	base = __shfl_sync(laneMask, base, leader);
	ranks[warpStart + lane] = base + cnt;
      }
    }
    // Prefix sum the counts.
    auto* temp = reinterpret_cast<typename Scan::TempStorage*>(smem);
    int32_t* aggregate = reinterpret_cast<int32_t*>(smem + sizeof(typename Scan::TempStorage));
    *aggregate = 0;
    for (auto start = 0; start < numPartitions; start += kBlockDimX) {
      int32_t localCount[1];
      localCount[0] = counters[start + threadIdx.x];
      if (threadIdx.x == 0) {
	localCount[0] += *aggregate;
      }
      Scan(*temp).InclusiveSum(localCount, localCount);
      partitionStarts[start + threadIdx.x] = localCount[0];
      if(threadIdx.x == kBlockDimX - 1 && start + kBlockDimX <numPartitions) {
	*aggregate = localCount[0];
      }
      __syncthreads();
    }  
    // Write the row numbers of the inputs into the rankth position in each partition.
    for (auto start = 0; start < numKeys; start += kBlockDimX) {
      auto i = start + threadIdx.x;
      auto key = getter(i);
      auto keyStart = key == 0 ? 0 : partitionStarts[key - 1]; 
      partitionedRows[keyStart + ranks[i]] = i;
    }
  }

  /// Calls partitionRows on each thread block of 256 threads. The parameters correspond to 'partitionRows'. Each is an array subscripted by blockIdx.x. 
  void __global__ partitionShortsKernel(uint16_t** keys, int32_t* numKeys, int32_t numPartitions, int32_t** ranks, int32_t** partitionStarts, int32_t** partitionedRows) {
    partitionRows<256>([&](auto i) {return keys[blockIdx.x][i];},
			      numKeys[blockIdx.x], numPartitions, nullptr, ranks[blockIdx.x], partitionStarts[blockIdx.x], partitionedRows[blockIdx.x]);
  }

  void BlockTestStream::partitionShorts(int32_t numBlocks, uint16_t** keys, int32_t* numKeys, int32_t numPartitions, 
					int32_t** ranks, int32_t** partitionStarts, int32_t**partitionedRows) {
    constexpr int32_t kBlockSize = 256;
    auto shared = partitionRowsSharedSize<kBlockSize>(numPartitions);
    partitionShortsKernel<<<numBlocks, kBlockSize, shared, stream_->stream>>>(keys, numKeys, numPartitions, ranks, partitionStarts, partitionedRows);
  CUDA_CHECK(cudaGetLastError());

  }

REGISTER_KERNEL("testSort", testSort);
REGISTER_KERNEL("boolToIndices", boolToIndices);
REGISTER_KERNEL("sum64", sum64);
REGISTER_KERNEL("partitionShorts", partitionShortsKernel);

} // namespace facebook::velox::wave
