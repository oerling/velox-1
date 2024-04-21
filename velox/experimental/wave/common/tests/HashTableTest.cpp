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

#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/tests/BlockTest.h"
#include "velox/experimental/wave/common/tests/CpuTable.h"
#include "velox/experimental/wave/common/Buffer.h"
#include <gtest/gtest.h>

namespace facebook::velox::wave {


  
class HashTableTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
    arena_ = std::make_unique<GpuArena>(1 << 28, allocator_);
  }

  void prefetch(Stream& stream, WaveBufferPtr buffer) {
    stream.prefetch(device_, buffer->as<char>(), buffer->capacity());
  }


  
  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
};

 
TEST_F(HashTableTest, hashMatrix) {
  std::vector<int32_t> sizeValues = {256, 8 << 10}; 
}

  TEST_F(HashTableTest, allocator) {
    constexpr int32_t kNumThreads = 256;
    constexpr int32_t kTotal = 1 << 30;
    WaveBufferPtr data = arena_->allocate<char>(kTotal);
    auto* allocator = data->as<HashPartitionAllocator>();
    new(allocator)  HashPartitionAllocator(data->as<char>() + sizeof(HashPartitionAllocator), kTotal - sizeof(HashPartitionAllocator), 16);
    WaveBufferPtr allResults = arena_->allocate<AllocatorTestResult>(kNumThreads);
    auto results = allResults->as<AllocatorTestResult>();
    for (auto i = 0; i < kNumThreads; ++i) {
      results[i].allocator = reinterpret_cast<RowAllocator*>(allocator);
      results[i].numRows = 0;
      results[i].numStrings = 0;    }
    auto stream1 = std::make_unique<BlockTestStream>();
    auto stream2 = std::make_unique<BlockTestStream>();
    stream1->rowAllocatorTest(2, 4, 3, 2, results);
    stream2->rowAllocatorTest(2, 4, 3, 2, results + 128);

    stream1->wait();
    stream2->wait();
    // Pointer to result idx, position in result;
    std::unordered_map<int64_t*, int32_t> uniques;
  for (auto resultIdx = 0; resultIdx < kNumThreads; ++resultIdx) {
    auto* result = results + resultIdx;
    for (auto i = 0; i < result->numRows; ++i) {
      auto row = result->rows[i];
      EXPECT_GE(reinterpret_cast<uint64_t>(row), allocator->base);
      EXPECT_LT(reinterpret_cast<uint64_t>(row),  allocator->base + allocator->capacity);
      auto it = uniques.find(row);
      EXPECT_TRUE(it == uniques.end())
	<< fmt::format("row {} is also at {} {}", reinterpret_cast<uint64_t>(row),  it->second >> 24, it->second & bits::lowMask(24));
      
      uniques[row] = (resultIdx << 24) | i;
    }
        for (auto i = 0; i < result->numStrings; ++i) {
	  auto string = result->strings[i];
	  EXPECT_GE(reinterpret_cast<uint64_t>(string), allocator->base);
      EXPECT_LT(reinterpret_cast<uint64_t>(string),  allocator->base + allocator->capacity);
      auto it = uniques.find(string);
      EXPECT_TRUE(it == uniques.end())
	<< fmt::format("String {} is also at {} {}", reinterpret_cast<uint64_t>(string),  it->second >> 24, it->second & bits::lowMask(24));
      uniques[string] = (resultIdx << 24) | i;
    }

  }

  }

}
