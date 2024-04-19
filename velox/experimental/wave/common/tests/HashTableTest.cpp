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

  // Returns the byte size for a GpuProbe with numRows as first, rounded row count as second.
  std::pair<int64_t, int32_t> probeSize(HashRun& run) {
    int32_t roundedRows = bits::roundUp(run.numRows, 256 * run.rowsperThread);
    return {sizeof(HashProbe) +
      // Column data and hash number array.
      (1 + run.numColumns) * roundedRows * sizeof(int64_t)
      // Pointers to column starts 
      + sizeof(int64_t*) * run.numColumns
      // retry lists
      + 2 * sizeof(int32_t) * roundedRows,
      roundedRows}; 
  }

  //
  void prepareInput(HashRun& run, bool isCpu) {
    auto [bytes, roundedRows] = probeSize(run);
    char* data;
    if (run.isCpu) {
      run.probeCpuData = malloc(bytes);
      data = run.probeCpuData;
    } else {
      run.gpuData = arena_->allocate<char>(bytes);
      data = run.gpuData->as<char>();
    }
    HashProbe probe = new(data) HashProbe();
    data += sizeof(HashProbe);
    probe.numKeys = reinterpret_cast<int32_t*>(data);
    data += sizeof(int32_t) * roundedRows / (run.rowsPerThread * 256);
    if (isCpu) {
      probe.numkeys[0] = run.numKeys;
    } else {
      int32_t numBlocks = roundedRows / (256 * run.rowsPerThread);
    }
    probe->hashes = reinterpret_cast<uint64_t*>(data);
    data += sizeof(uint64_t) * roundedRows;
    probe.keys = data;
    data += sizeof(void*) * run.numColumns;
    probe.kernelRetries = reinterpret_cast<int32_t*>(data);
    data += sizeof(int32_t) * roundedRows;
    probe.hostRetries = reinterpret_cast<int32_t*>(data);
    data += sizeof(int32_t) * roundedRows;
    for (auto i = 0; i <run.numColumns; ++i) {
      reinterpret_cast<int64_t**>(probe.keys)[i] = reinterpret_cast<int64_t*>(data);
      data += sizeof(int64_t) * roundedRows;
    }
  }

  
  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
}

 
TEST_F(HashTableTest, hashMatrix) {
  std::vector<int32_t> sizeValues = {256, 8 << 10, 
}
  
}
