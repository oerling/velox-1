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

#include <gtest/gtest.h>
#include "velox/common/time/Timer.h"
#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/tests/BlockTest.h"
#include "velox/experimental/wave/common/tests/CpuTable.h"
#include "velox/experimental/wave/common/tests/HashTestUtil.h"

namespace facebook::velox::wave {

class HashTableTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
    arena_ = std::make_unique<GpuArena>(1 << 28, allocator_);
    streams_.push_back(std::make_unique<BlockTestStream>());
  }

  void prefetch(Stream& stream, WaveBufferPtr buffer) {
    stream.prefetch(device_, buffer->as<char>(), buffer->capacity());
  }

  // Tests different styles of updating a group by. Results are returned in
  // 'run'.
  void updateTestCase(int32_t numDistinct, int32_t numRows, HashRun& run) {
    run.numRows = numRows;
    run.numDistinct = numDistinct;
    run.numColumns = 2;
    run.numRowsPerThread = 4;

    initializeHashTestInput(run, arena_.get());
    fillHashTestInput(
        run.numRows,
        run.numDistinct,
        bits::nextPowerOfTwo(run.numDistinct),
        1,
        run.numColumns,
        reinterpret_cast<int64_t**>(run.probe->keys));
    std::vector<TestingRow> reference(run.numDistinct);
    for (auto i = 0; i < run.numDistinct; ++i) {
      reference[i].key = i;
    }
    WaveBufferPtr gpuRowsBuffer = arena_->allocate<TestingRow>(run.numDistinct);
    TestingRow* gpuRows = gpuRowsBuffer->as<TestingRow>();
    memcpy(gpuRows, reference.data(), sizeof(TestingRow) * run.numDistinct);
    prefetch(*streams_[0], gpuRowsBuffer);
    prefetch(*streams_[0], run.gpuData);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    runCpu(reference.data(), run);
    runGpu(gpuRows, run, reference.data());
    std::cout << run.toString() << std::endl;
  }

  void runCpu(TestingRow* rows, HashRun& run) {
    uint64_t micros = 0;
    {
      MicrosecondTimer t(&micros);
      switch (run.testCase) {
        case HashTestCase::kUpdateSum1: {
          int64_t** keys = reinterpret_cast<int64_t**>(run.probe->keys);
          int64_t* indices = keys[0];
          int64_t* data = keys[1];
          auto numRows = run.numRows;
          for (auto i = 0; i < numRows; ++i) {
            rows[indices[i]].count += data[i];
          }
          break;
        }
        default:
          VELOX_FAIL("Unsupported test case");
      }
    }
    run.scores.push_back(std::make_pair<std::string, float>(
        "Cpu1T", run.numRows / (micros / 1e6)));
  }

  void runGpu(TestingRow* rows, HashRun& run, TestingRow* reference) {
    uint64_t micros = 0;
    if (streams_.empty()) {
      streams_.push_back(std::make_unique<BlockTestStream>());
    }
    switch (run.testCase) {
      case HashTestCase::kUpdateSum1: {
        MicrosecondTimer t(&micros);
        streams_[0]->updateSum1Atomic(rows, run);
        streams_[0]->wait();
      }
        run.scores.push_back(std::make_pair<std::string, float>(
            "Gpu atm", run.numRows / (micros / 1e6)));
        compareAndReset(reference, rows, run.numDistinct);
	prefetch(streams_[0], gpuRowsBuffer);
        break;
      default:
        VELOX_FAIL("Unsupported test case");
    }
  }

  void
  compareAndReset(TestingRow* reference, TestingRow* rows, int32_t numRows) {
    for (auto i = 0; i < numRows; ++i) {
      if (rows[i].count == reference[i].count) {
        continue;
      }
      EXPECT_EQ(reference[i].count, rows[i].count) << " at " << i;
      break;
    }
    for (auto i = 0; i < numRows; ++i) {
      new (rows + i) TestingRow();
      rows[i].key = i;
    }
  }

  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
  std::vector<std::unique_ptr<BlockTestStream>> streams_;
};

TEST_F(HashTableTest, hashMatrix) {
  std::vector<int32_t> sizeValues = {256, 8 << 10};
}

TEST_F(HashTableTest, allocator) {
  constexpr int32_t kNumThreads = 256;
  constexpr int32_t kTotal = 1 << 22;
  WaveBufferPtr data = arena_->allocate<char>(kTotal);
  auto* allocator = data->as<HashPartitionAllocator>();
  auto freeSetSize = BlockTestStream::freeSetSize();
  new (allocator) HashPartitionAllocator(
      data->as<char>() + sizeof(HashPartitionAllocator) + freeSetSize,
      kTotal - sizeof(HashPartitionAllocator) - freeSetSize,
      16);
  allocator->freeSet = allocator + 1;
  memset(allocator->freeSet, 0, freeSetSize);
  WaveBufferPtr allResults = arena_->allocate<AllocatorTestResult>(kNumThreads);
  auto results = allResults->as<AllocatorTestResult>();
  for (auto i = 0; i < kNumThreads; ++i) {
    results[i].allocator = reinterpret_cast<RowAllocator*>(allocator);
    results[i].numRows = 0;
    results[i].numStrings = 0;
  }
  auto stream1 = std::make_unique<BlockTestStream>();
  auto stream2 = std::make_unique<BlockTestStream>();
  stream1->initAllocator(allocator);
  stream1->wait();
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
      EXPECT_LT(
          reinterpret_cast<uint64_t>(row),
          allocator->base + allocator->capacity);
      auto it = uniques.find(row);
      EXPECT_TRUE(it == uniques.end()) << fmt::format(
          "row {} is also at {} {}",
          reinterpret_cast<uint64_t>(row),
          it->second >> 24,
          it->second & bits::lowMask(24));

      uniques[row] = (resultIdx << 24) | i;
    }
    for (auto i = 0; i < result->numStrings; ++i) {
      auto string = result->strings[i];
      EXPECT_GE(reinterpret_cast<uint64_t>(string), allocator->base);
      EXPECT_LT(
          reinterpret_cast<uint64_t>(string),
          allocator->base + allocator->capacity);
      auto it = uniques.find(string);
      EXPECT_TRUE(it == uniques.end()) << fmt::format(
          "String {} is also at {} {}",
          reinterpret_cast<uint64_t>(string),
          it->second >> 24,
          it->second & bits::lowMask(24));
      uniques[string] = (resultIdx << 24) | i;
    }
  }
}

TEST_F(HashTableTest, update) {
  {
    HashRun run;
    run.testCase = HashTestCase::kUpdateSum1;
    updateTestCase(1000, 2000000, run);
  }
  {
    HashRun run;
    run.testCase = HashTestCase::kUpdateSum1;
    updateTestCase(10000000, 2000000, run);
  }
}

} // namespace facebook::velox::wave
