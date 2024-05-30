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
#include "velox/experimental/wave/common/tests/CudaTest.h"

namespace facebook::velox::wave {
constexpr uint32_t kPrime32 = 1815531889;

  struct ResultPair {
    int64_t n;
    bool f;
  };
  
  typedef ResultPair(*TestFunc)(int64_t data, int64_t data2, bool& flag, int32_t* ptr);

__device__ TestFunc testFuncs[2];

  __device__ ResultPair testFunc(int64_t data, int64_t data2, bool& flag, int32_t* ptr){
    return {data + (data2 & 31), false};
}

void   __global__ setupFuncs() {
    testFuncs[0] = testFunc;
    testFuncs[1] = testFunc;
  }


__global__ void
incOneKernel(int32_t* numbers, int32_t size, int32_t stride, int32_t repeats) {
  for (auto counter = 0; counter < repeats; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
         index += stride) {
      ++numbers[index];
    }
    __syncthreads();
  }
}
  
__global__ void
addOneKernel(int32_t* numbers, int32_t size, int32_t stride, int32_t repeats) {
  for (auto counter = 0; counter < repeats; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
         index += stride) {
      numbers[index] += index & 31;
    }
    __syncthreads();
  }
}

__global__ void addOneSharedKernel(
    int32_t* numbers,
    int32_t size,
    int32_t stride,
    int32_t repeats) {
  extern __shared__ __align__(16) char smem[];
  int32_t* temp = reinterpret_cast<int32_t*>(smem);
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
       index += stride) {
    temp[threadIdx.x] = numbers[index];
    for (auto counter = 0; counter < repeats; ++counter) {
      temp[threadIdx.x] += (index + counter) & 31;
    }
    __syncthreads();
    numbers[index] = temp[threadIdx.x];
  }
}

__global__ void addOneRegKernel(
    int32_t* numbers,
    int32_t size,
    int32_t stride,
    int32_t repeats) {
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
       index += stride) {
    auto temp = numbers[index];
    for (auto counter = 0; counter < repeats; ++counter) {
      temp += (index + counter) & 31;
    }
    __syncthreads();
    numbers[index] = temp;
  }
}


#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR "l"
#else
#define __LDG_PTR "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/

__global__ void kernel(uint32_t* tgt)
{
        printf("tgt = %d\n", *tgt);
        asm volatile(".reg .u32 r_tgt;");
        asm volatile("ld.u32 r_tgt, [%0];" :: __LDG_PTR(tgt));
        asm volatile("ts: .branchtargets BLK0, BLK1, BEXIT;");
        asm volatile("brx.idx r_tgt, ts;");
        asm volatile("BLK0:");
        printf("BLK0\n");
        asm volatile("ret;\n");
        asm volatile("BLK1:");
        printf("BLK1\n");
        asm volatile("ret;\n");
        asm volatile("BEXIT:");
        printf("BEXIT\n");
        asm volatile("ret;\n");
}

  
__global__ void addOneFuncKernel(
    int32_t* numbers,
    int32_t size,
    int32_t stride,
    int32_t repeats) {
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
       index += stride) {
    int32_t* ptr = nullptr;
    bool flag;
    auto temp = numbers[index];
    for (auto counter = 0; counter < repeats; ++counter) {
      auto result = testFuncs[counter & 1](temp, counter, flag, ptr);
      temp = result.n;
    }
    __syncthreads();
    numbers[index] = temp;
  }
}

#define TCASE(nn, m) \
        case nn: \
	temp = m + testFunc(temp, counter, flag, ptr).n; \
 break; \

__global__ void addOneFuncSwitchKernel(
    int32_t* numbers,
    int32_t size,
    int32_t stride,
    int32_t repeats) {
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
       index += stride) {
    int32_t* ptr = nullptr;
    bool flag;
    auto temp = numbers[index];
    for (auto counter = 0; counter < repeats; ++counter) {
      switch (counter & 15) {
	TCASE(0, 1);
	TCASE(1, 82);
#if 0
	TCASE(2, 91);
	TCASE(3, 181);
	TCASE(4, 28);
	TCASE(5, 36);
	TCASE(6, 18);
	TCASE(7, 13);
	TCASE(8, 21);
	TCASE(9, 32);
	TCASE(10, 31);
	TCASE(11, 191);
	TCASE(12, 181);
	TCASE(13, 151);
	TCASE(14, 121);
	TCASE(15, 111);
#endif
      }
      }
    __syncthreads();
    numbers[index] = temp;
  }
}

__global__ void addOneFuncStoreKernel(
    int32_t* numbers,
    int32_t size,
    int32_t stride,
    int32_t repeats) {
  for (auto counter = 0; counter < repeats; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
	 index += stride) {
      int32_t* ptr = nullptr;
      bool flag;
      auto temp = numbers[index];
      numbers[index] = testFuncs[counter & 1](temp, counter, flag, ptr).n;
    }
    __syncthreads();  }
    }

void TestStream::incOne(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  incOneKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

void TestStream::addOne(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

void TestStream::addOneReg(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneRegKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

void TestStream::addOneFunc(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  setupFuncs<<<1, 1, 0, stream_->stream>>>();
  CUDA_CHECK(cudaGetLastError());
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneFuncKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}


void TestStream::addOneFuncStore(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  setupFuncs<<<1, 1, 0, stream_->stream>>>();
  CUDA_CHECK(cudaGetLastError());
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneFuncStoreKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

  
void TestStream::addOneShared(
    int32_t* numbers,
    int32_t size,
    int32_t repeats,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneSharedKernel<<<
      numBlocks,
      kBlockSize,
      kBlockSize * sizeof(int32_t),
      stream_->stream>>>(numbers, size, stride, repeats);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void addOneWideKernel(WideParams params) {
  auto numbers = params.numbers;
  auto size = params.size;
  auto repeat = params.repeat;
  auto stride = params.stride;
  for (auto counter = 0; counter < repeat; ++counter) {
    for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
         index += stride) {
      ++numbers[index];
    }
  }
}

void TestStream::addOneWide(
    int32_t* numbers,
    int32_t size,
    int32_t repeat,
    int32_t width) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  WideParams params;
  params.numbers = numbers;
  params.size = size;
  params.stride = stride;
  params.repeat = repeat;
  addOneWideKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void __launch_bounds__(1024) addOneRandomKernel(
    int32_t* numbers,
    const int32_t* lookup,
    uint32_t size,
    int32_t stride,
    int32_t repeats,
    int32_t numLocal,
    int32_t localStride,
    bool emptyWarps,
    bool emptyThreads) {
  for (uint32_t counter = 0; counter < repeats; ++counter) {
    if (emptyWarps) {
      if (((threadIdx.x / 32) & 1) == 0) {
        for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
             index += stride) {
          auto rnd = deviceScale32(index * (counter + 1) * kPrime32, size);
          auto sum = lookup[rnd];
          auto limit = min(rnd + localStride * (1 + numLocal), size);
          for (auto j = rnd + localStride; j < limit; j += localStride) {
            sum += lookup[j];
          }
          numbers[index] += sum;

          rnd = deviceScale32((index + 32) * (counter + 1) * kPrime32, size);
          sum = lookup[rnd];
          limit = min(rnd + localStride * (1 + numLocal), size);
          for (auto j = rnd + localStride; j < limit; j += localStride) {
            sum += lookup[j];
          }
          numbers[index + 32] += sum;
        }
      }
    } else if (emptyThreads) {
      if ((threadIdx.x & 1) == 0) {
        for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
             index += stride) {
          auto rnd = deviceScale32(index * (counter + 1) * kPrime32, size);
          auto sum = lookup[rnd];
          auto limit = min(rnd + localStride * (1 + numLocal), size);
          for (auto j = rnd + localStride; j < limit; j += localStride) {
            sum += lookup[j];
          }
          numbers[index] += sum;

          rnd = deviceScale32((index + 1) * (counter + 1) * kPrime32, size);
          sum = lookup[rnd];
          limit = min(rnd + localStride * (1 + numLocal), size);
          for (auto j = rnd + localStride; j < limit; j += localStride) {
            sum += lookup[j];
          }
          numbers[index + 1] += sum;
        }
      }
    } else {
      for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < size;
           index += stride) {
        auto rnd = deviceScale32(index * (counter + 1) * kPrime32, size);
        auto sum = lookup[rnd];
        auto limit = min(rnd + localStride * (1 + numLocal), size);
        for (auto j = rnd + localStride; j < limit; j += localStride) {
          sum += lookup[j];
        }
        numbers[index] += sum;
      }
    }
    __syncthreads();
  }
  __syncthreads();
}

void TestStream::addOneRandom(
    int32_t* numbers,
    const int32_t* lookup,
    int32_t size,
    int32_t repeats,
    int32_t width,
    int32_t numLocal,
    int32_t localStride,
    bool emptyWarps,
    bool emptyThreads) {
  constexpr int32_t kBlockSize = 256;
  auto numBlocks = roundUp(size, kBlockSize) / kBlockSize;
  int32_t stride = size;
  if (numBlocks > width / kBlockSize) {
    stride = width;
    numBlocks = width / kBlockSize;
  }
  addOneRandomKernel<<<numBlocks, kBlockSize, 0, stream_->stream>>>(
      numbers,
      lookup,
      size,
      stride,
      repeats,
      numLocal,
      localStride,
      emptyWarps,
      emptyThreads);
  CUDA_CHECK(cudaGetLastError());
}

REGISTER_KERNEL("addOne", addOneKernel);
REGISTER_KERNEL("addOneFunc", addOneFuncKernel);
REGISTER_KERNEL("addOneWide", addOneWideKernel);
REGISTER_KERNEL("addOneRandom", addOneRandomKernel);

} // namespace facebook::velox::wave
