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

#include <iostream>

#include <gtest/gtest.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Semaphore.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/time/Timer.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/tests/CudaTest.h"

using namespace facebook::velox;
using namespace facebook::velox::wave;

class CudaTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
  }
  Device* device_;
  GpuAllocator* allocator_;
};

TEST_F(CudaTest, stream) {
  constexpr int32_t kSize = 1000000;
  TestStream stream;
  auto ints =
      reinterpret_cast<int32_t*>(allocator_->allocate(kSize * sizeof(int32_t)));
  for (auto i = 0; i < kSize; ++i) {
    ints[i] = i;
  }
  stream.prefetch(device_, ints, kSize * sizeof(int32_t));
  stream.addOne(ints, kSize);
  stream.prefetch(nullptr, ints, kSize * sizeof(int32_t));
  stream.wait();
  for (auto i = 0; i < kSize; ++i) {
    ASSERT_EQ(ints[i], i + 1);
  }
  allocator_->free(ints, sizeof(int32_t) * kSize);
}

TEST_F(CudaTest, callback) {
  // Makes several parallel streams and enqueues a series of kernel and host
  // function invocations on each. Records the latency at every host function
  // invocation. Finally waits for all the host function to be invoked.
  constexpr int32_t kSize = 1024 * 1024;
  constexpr int32_t kNumStreams = 10;
  constexpr int32_t kNumOps = 10;
  constexpr int32_t kFirstNotify = kNumOps - 1;
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  std::vector<std::unique_ptr<TestStream>> streams;
  std::vector<int32_t*> ints;
  std::mutex mutex;
  int32_t initValues[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto initVector = xsimd::load_unaligned(&initValues[0]);
  auto increment = xsimd::broadcast<int32_t>(1);
  std::vector<int64_t> delay;
  delay.reserve(kNumStreams * (kNumOps + 2));

  auto start = getCurrentTimeMicro();
  for (auto i = 0; i < kNumStreams; ++i) {
    streams.push_back(std::make_unique<TestStream>());
    ints.push_back(reinterpret_cast<int32_t*>(
        allocator_->allocate(kSize * sizeof(int32_t))));
    auto last = ints.back();
    auto data = initVector;
    for (auto i = 0; i < kSize; i += kBatch) {
      data.store_unaligned(last + i);
      data += increment;
    }
  }
  for (auto i = 0; i < kNumStreams; ++i) {
    streams[i]->addCallback([&]() {
      auto d = getCurrentTimeMicro() - start;
      {
        std::lock_guard<std::mutex> l(mutex);
        delay.push_back(d);
      }
    });
    streams[i]->prefetch(device_, ints[i], kSize * sizeof(int32_t));
  }

  Semaphore sem(0);
  for (auto counter = 0; counter < kNumOps; ++counter) {
    for (auto i = 0; i < kNumStreams; ++i) {
      streams[i]->addOne(ints[i], kSize);
      if (counter == 0 || counter >= kFirstNotify) {
	streams[i]->addCallback([&]() {
        auto d = getCurrentTimeMicro() - start;
        {
          std::lock_guard<std::mutex> l(mutex);
          delay.push_back(d);
        }
        sem.release();
      });
      }
	if (counter == kNumOps - 1) {
        streams[i]->prefetch(nullptr, ints[i], kSize * sizeof(int32_t));
      }
    }
  }
  // Destroy the streams while items pending. Items should finish.
  streams.clear();
  for (auto i = 0; i < kNumStreams * (kNumOps + 1 - kFirstNotify); ++i) {
    sem.acquire();
  }
  for (auto i = 0; i < kNumStreams; ++i) {
    auto* array = ints[i];
    auto data = initVector + kNumOps;
    xsimd::batch_bool<int32_t> error;
    error = error ^ error;
    for (auto j = 0; j < kSize; j += kBatch) {
      error = error | (data != xsimd::load_unaligned(array + j));
      data += increment;
    }
    ASSERT_EQ(0, simd::toBitMask(error));
    delay.push_back(getCurrentTimeMicro() - start);
  }
  for (auto i = 0; i < kNumStreams; ++i) {
    allocator_->free(ints[i], sizeof(int32_t) * kSize);
  }
  std::cout << "Delays: ";
  int32_t counter = 0;
  for (auto d : delay) {
    std::cout << d << " ";
    if (++counter % kNumStreams == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  float toDeviceMicros = delay[(2 * kNumStreams) - 1] - delay[0];
  float inDeviceMicros =
      delay[delay.size() - kNumStreams - 1] - delay[kNumStreams * 2 - 1];
  float toHostMicros = delay.back() - delay[delay.size() - kNumStreams];
  float gbSize =
    (sizeof(int32_t) * kNumStreams * static_cast<float>(kSize)) / (1 << 30);
  std::cout << "to device= " << toDeviceMicros << "us ("
            << gbSize / (toDeviceMicros / 1000000) << " GB/s)" << std::endl;
  std::cout << "In device (ex. first pass): " << inDeviceMicros << "us ("
            << gbSize * (kNumOps - 1) / (inDeviceMicros / 1000000) << " GB/s)"
            << std::endl;
  std::cout << "to host= " << toHostMicros << "us ("
            << gbSize / (toHostMicros / 1000000) << " GB/s)" << std::endl;
}
