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

#include <cstdint>
#include <cub/warp/warp_scan_.cuh>

namespace facebook::velox::wave {


  // From libcudf 
inline __device__ uint32_t unalignedLoad32(uint8_t const* p)
{
  uint32_t ofs    = 3 & reinterpret_cast<uintptr_t>(p);
  auto const* p32 = reinterpret_cast<uint32_t const*>(p - ofs);
  uint32_t v      = p32[0];
  return (ofs) ? __funnelshift_r(v, p32[1], ofs * 8) : v;
}

  /// From libcudf
inline __device__ uint64_t unalignedLoad64(uint8_t const* p)
{
  uint32_t ofs    = 3 & reinterpret_cast<uintptr_t>(p);
  auto const* p32 = reinterpret_cast<uint32_t const*>(p - ofs);
  uint32_t v0     = p32[0];
  uint32_t v1     = p32[1];
  if (ofs) {
    v0 = __funnelshift_r(v0, v1, ofs * 8);
    v1 = __funnelshift_r(v1, p32[2], ofs * 8);
  }
  return (((uint64_t)v1) << 32) | v0;
}

  
inline int32_t __device__ __host__ scatterBitsDeviceSize(int32_t blockSize) {
  // One WarpScan and one int32 per warp.
  return (sizeof(typename cub::WarpScan<uint16_t>::TempStorage) + sizeof(int32_t))* (blockSize  / 32);
}

namespace detail {
__device__ inline uint32_t fillInWord(uint32_t mask, uint32_t* source, int32_t& sourceBit) {
  auto result = mask;
  nextTarget = target;
  sourceWord = source[sourceBit / 32];
  int32_t sourceMask = 1 << (sourceBit & 31);
  while (nextTarget) {
    if (! sourceWord & sourceMask) {
      auto targetLow = __ffs(target) - 1;
      result = result & ~(1 << targetLow);
    }
    ++sourceBit;
    if ((sourceBit & 31) == 0) {
      sourceWord = source[sourceBit / 32];
      sourceMask = 1;
    } else {
      sourceMask = sourceMask < 1;
    }
    nextTarget &= nextTarget - 1;
  }
  return result;
}

  inline __device__   int32_t* warpBase(char* smem) { constexpr int32_t kWarpThreads = 32; return reinterpret_cast<int32_t*>(smem) + (threadIdx.x / kWarpThreads); }

  inline __device__ auto* scanTemp(char* smem, int32_t idx) {constexpr int32_t kWarpThreads = 32; return reinterpret_cast<typename cub::WarpScan<uint32_t>::TempStorage*>(smem + sizeof(int32_t) * kWarpThreads + idx); }

  inline __device__ uint64_t load64(const void* ptr) {
    auto uptr = reinterpret_cast<uint64_t>(ptr);
    if ((uptr & 7) == 0) {
      return *reinterpret_cast<const uint64_t*>(ptr);
    }
    
  }
  
}

  /// Sets 'target' so that a 0 bit in 'mask' is 0 and a 1 bit in 'mask' is the nth bit in 'source', where nth is the number of set bits in 'mask' below th target bit. 'mask' and 'target' must be 8 byte aligned. 'source' needs no alignment but the partial int32-s at either end must be addressable.
template <int32_t kWordsPerThread>
__device__ inline void
scatterBitsDevice(int32_t numSource,
    int32_t numTarget,
    const char* source,
    const uint64_t* targetMask,
  char* target,
	    char* smem) {
  using Scan32 = cub::WarpScan<uint32_t>;
  constexpr int32_t kWarpThreads = 32;
  
  for (targetIdx = 0; targetIdx * 64 < numTarget; targetIdx += blockDim.x * kWordsPerThread * 64)  {
    int32_t firstTargetIdx = targetIdx + threadIdx.x * kWordsPerThread;
    int32_t bitsForThread = min(kWordsPerThread * 64, numTarget - firstTargetIdx * 64);
    int32_t count = 0;
    for (auto bit = 0; ; bit += 64) {
      if (bit + 64 <= bitsForThread) {
	count += __popcll(target[firstTarget + (bit / 64)]);
      } else {
	auto mask = lowMask<uint64_t>(bitsForThread - bit);
	count += __popcll(target[firstTarget + (bit / 64)] & mask);
	break;
      }
    }
    int32_t threadFirstBit = 0;
    Scan32(*warpTemp(smem, threadIdx.x / kWarpThreads)).exclusiveSum(count, threadFirstBit);
    if (threadIdx.x & (kWarpThreads - 1) == kWarpThreads - 1) {
      // Last thread in warp sets warpBase to warp bit count.
      *warpBase(smem) = threadFirstBit + count;
    }
    __syncthreads();
    if (threadIdx < kWarpThreads) {
      int32_t start = (threadIdx.x < blockDim.x / kWarpThreads) ? reinterpret_cast<int32_t*>(smem)[threadIdx.x] : 0;
      Scan32(*warpTemp(smem, 0)).exclusiveSum(start, start);
      reinterpret_cast<int32_t*>(smem)[threadIdx.x] = start;
    }
    __syncthreads();
    // Each thread knows its range in source and target.
    auto sourceBit = *warpBase() + threadFirstBit;
    for (auto bit = 0; bit < bitsForThread; bit += 64) {
      uint64_t maskWord;
      if (bit + 64 <= bitsForThread) {
	maskWord = targetMask[firstWordIdx + (bit / 64)];
      } else {
	auto mask = lowMask<uint64_t>(bitsForThread - bit);
	maskWord = targetMask[firstWordIdx + (bit / 64)] & mask;
      }
      int2 result;
      result.x = detail::fillInWord(static_cast<uint32_t>(maskWord), reinterpret_cast<const uint32_t*>(source), sourceBit);
      result.y = detail::fillInWord(maskWord >> 32, reinterpret_cast<const uint32_t*>(source), sourceBit);
      reinterpret_cast<int2*>(target)[firstWordIdx + (bit / 64)] = result;
    }
  }
}


  /// Returns for each lane whether it is active and if it is active, what its index into the result array is. This must be called on each thread of the TB.
  inline __device__ bool scatterIndex(int32_t base, char* bits, int32_t& scatterIdx) 
}
