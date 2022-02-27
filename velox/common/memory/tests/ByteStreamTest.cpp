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
#include "velox/common/memory/MappedMemory.h"
#include "velox/common/base/test_utils/GTestUtils.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/memory/ByteStream.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DECLARE_int32(velox_memory_pool_mb);

namespace facebook::velox::memory {

static constexpr uint64_t kMaxMappedMemory = 128UL * 1024 * 1024;
static constexpr MachinePageCount kCapacity =
    (kMaxMappedMemory / MappedMemory::kPageSize);

class ByteStreamTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    auto tracker = MemoryUsageTracker::create(
        MemoryUsageConfigBuilder().maxTotalMemory(kMaxMappedMemory).build());
      MmapAllocatorOptions options = {kMaxMappedMemory};
      mmapAllocator_ = std::make_unique<MmapAllocator>(options);
      MappedMemory::setDefaultInstance(mmapAllocator_.get());
    instancePtr_ = MappedMemory::getInstance()->addChild(tracker);
    instance_ = instancePtr_.get();
  }

  void TearDown() override {
    MappedMemory::destroyTestOnly();
  }

  std::unique_ptr<MmapAllocator> mmapAllocator_;
  std::shared_ptr<MappedMemory> instancePtr_;
  MappedMemory* instance_;
};

  TEST_F(ByteStreamTest, basic) {
    auto out = std::make_unique<IOBufOutputStream>(instance_, nullptr, 10000);
    std::stringstream referenceSStream;
    auto reference = std::make_unique<OStreamOutputStream>(&referenceSStream);
    for (auto i = 0; i < 100; ++i) {
      std::string data;
      data.resize(10000);
      std::fill(data.begin(), data.end(), i);
      out->write(data.data(), data.size());
      reference->write(data.data(), data.size());
    }
    EXPECT_EQ(reference->tellp(), out->tellp());
    for (auto i = 0; i < 100; ++i) {
      std::string data;
      data.resize(6000);
      std::fill(data.begin(), data.end(), i+ 10);
      out->seekp(i * 10000 + 5000);
      reference->seekp(i * 10000 + 5000);
      out->write(data.data(), data.size());
      reference->write(data.data(), data.size());
    }
    auto str = referenceSStream.str();
    auto iobuf = out->getIOBuf();
    auto iobuf2 = out->getIOBuf();
    auto out1Data = iobuf->coalesce();
    EXPECT_EQ(out1Data->length(), str.size());
    
    
  }
