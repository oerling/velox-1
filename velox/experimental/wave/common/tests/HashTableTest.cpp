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
#include "velox/experimental/wave/common/test/CpuTable.h"

namespace facebook::velox::wave {

struct HashRun {
    // Number of slots in table.
    int32_t numSlots;

    // Number of probe rows.
    int32_t numRows;

    // Number of distinct keys.
    int32_t numDistinct;

    // Number of distinct hot keys.
    int32_t numHot;

    // Percentage of hot keys over total keys. e.g. with 1000 distinct and 10 hot and hotPct of 50, every second key will be one of 10 and the rest are evenly spread over the remaining 990.
    int32_t hotPct{0};

  // Number of keys processed by each thread of each block.
  int32_t rowsPerThread;

  // Number of blocks of 256 threads.
  int32_t numBlocks;

  // Number of independent hash tables.
  int32_t numTables;

  float gpuRPS;
  float cpuRPS;

  std::string toString() {
    return fmt::format("");
  }
};
 
  
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

  // Returns the byte size for a GpuProbe with numRows.
  int32_t probeSize(int32_t numRows, int32_t numColumns, int32_t rowsPerThread) {
    int32_t roundedRows = bits::roundUp(numRows, 256 * rowsperThread);
    return sizeof(HashProbe) +
      // Column data and hash number array.
      (1 + numColumns) * roundedRows * sizeof(int64_t)
      // Pointers to column starts 
      + sizeof(int64_t*) * numColumns
      // retry lists
      + 2 * sizeof(int32_t) * roundedRows; 
  }


  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
}

 
TEST_F(HashTableTest, hashMatrix) {
  std::vector<int32_t> sizeValues = {256, 8 << 10, 
}
  
}
