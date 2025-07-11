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

#include "velox/common/base/BloomFilter.h"
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/time/Timer.h"

#include <folly/Hash.h>
#include <folly/Random.h>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include <unordered_set>
#include "fmt/format.h"

using namespace facebook::velox;

class BloomFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (auto i = 0; i < 64; ++i) {
      masks[i] = 1UL << i;
    }
  }
  template <typename T>
  int8_t low(T& p, int32_t i) {
    return reinterpret_cast<char*>(&p)[i * 8];
  }

  template <int8_t bits>
  void meter(int32_t numEntries, int32_t threads = 1) {
    // Insert even values, test all values. Measure false positives.
    if (!executor_) {
      executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(64);
    }
    auto bloomSize = bits::nextPowerOfTwo(numEntries);
    std::vector<int64_t> toInsert(numEntries);
    std::vector<int64_t> toProbe(numEntries * 2);
    for (auto i = 0; i < numEntries; ++i) {
      toInsert[i] = i * 2;
      toProbe[i * 2] = i * 2;
      toProbe[i * 2 + 1] = i * 2 + 1;
    }
    BloomFilter<std::allocator<uint64_t>, bits> bloom;
    bloom.reset(bloomSize);
    uint64_t serialBuild = 0;
    uint64_t parallelBuild = 0;
    uint64_t serialProbe = 0;
    {
      ClockTimer t(serialBuild);
      for (auto i = 0; i < numEntries; ++i) {
        bloom.insert(simd::crcHash64(toInsert[i]));
      }
    }
    bloom.reset(bloomSize);
    uint64_t sizeMask = bloom.bits().size() - 1;
    {
      ClockTimer t(parallelBuild);
      std::vector<std::shared_ptr<AsyncSource<bool>>> workItems;
      for (auto start = 0; start < numEntries; start += numEntries / 32) {
        auto end = start + numEntries / 32;
        auto item = std::make_shared<AsyncSource<bool>>([start, end, sizeMask, &toInsert, &bloom]() {
          for (auto i = start; i < end; ++i) {
            auto h = simd::crcHash64(toInsert[i]);
	    bloom.atomicInsert(h);
          }
          return std::make_unique<bool>(true);
        });
        workItems.push_back(item);
      }
      for (auto& item : workItems) {
	executor_->add([it = item]() { it->prepare(); });
		
      }
      for (auto& item : workItems) {
        item->move();
      }
    }
    std::vector<int32_t> hits(numEntries * 2);
    int32_t numHits = 0;
    {
      ClockTimer t(serialProbe);
      for (auto i = 0; i < toProbe.size(); ++i) {
        if (bloom.mayContain(simd::crcHash64(toProbe[i]))) {
          hits[numHits++] = i;
        } else {
	  if ((i & 1) == 0) {
	    FAIL() << "Every even entry must be a hit";
	  }
	}
      }
    }
    std::cout << fmt::format(
        "{} entries/{} bits: false%={} serial insert={} parallel insert={} serial probe={}\n",
        numEntries,
        bits,
        100.0 * (numHits - numEntries) / numEntries,
        serialBuild / numEntries,
        parallelBuild / numEntries,
        serialProbe / (numEntries * 2));
  }

  int64_t masks[64];
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
};

TEST_F(BloomFilterTest, basic) {
  constexpr int32_t kSize = 1024;
  BloomFilter bloom;
  bloom.reset(kSize);
  for (auto i = 0; i < kSize; ++i) {
    bloom.insert(folly::hasher<int32_t>()(i));
  }
  int32_t numFalsePositives = 0;
  for (auto i = 0; i < kSize; ++i) {
    EXPECT_TRUE(bloom.mayContain(folly::hasher<int32_t>()(i)));
    numFalsePositives += bloom.mayContain(folly::hasher<int32_t>()(i + kSize));
    numFalsePositives +=
        bloom.mayContain(folly::hasher<int32_t>()((i + kSize) * 123451));
  }
  EXPECT_GT(2, 100 * numFalsePositives / kSize);
}

TEST_F(BloomFilterTest, serialize) {
  constexpr int32_t kSize = 1024;
  BloomFilter bloom;
  bloom.reset(kSize);
  for (auto i = 0; i < kSize; ++i) {
    bloom.insert(folly::hasher<int32_t>()(i));
  }
  std::string data;
  data.resize(bloom.serializedSize());
  bloom.serialize(data.data());
  BloomFilter deserialized;
  deserialized.merge(data.data());
  for (auto i = 0; i < kSize; ++i) {
    EXPECT_TRUE(deserialized.mayContain(folly::hasher<int32_t>()(i)));
  }
  EXPECT_FALSE(
      deserialized.mayContain(folly::hasher<int32_t>()(kSize + 123451)));

  EXPECT_EQ(bloom.serializedSize(), deserialized.serializedSize());
}

TEST_F(BloomFilterTest, staticMayContain) {
  constexpr int32_t kSize = 1024;
  std::string serializedBloom;
  BloomFilter bloom;
  bloom.reset(kSize);
  for (auto i = 0; i < kSize; ++i) {
    bloom.insert(folly::hasher<int32_t>()(i));
  }
  serializedBloom.resize(bloom.serializedSize());
  bloom.serialize(serializedBloom.data());
  int32_t numFalsePositives = 0;
  for (auto i = 0; i < kSize; ++i) {
    EXPECT_TRUE(BloomFilter<>::mayContain(
        serializedBloom.data(), folly::hasher<int32_t>()(i)));

    const uint64_t smallValueHash = folly::hasher<int32_t>()(i + kSize);
    const bool isFalsePositiveForSmallValue =
        BloomFilter<>::mayContain(serializedBloom.data(), smallValueHash);
    EXPECT_EQ(isFalsePositiveForSmallValue, bloom.mayContain(smallValueHash));
    numFalsePositives += isFalsePositiveForSmallValue;

    const uint64_t largeValueHash =
        folly::hasher<int32_t>()((i + kSize) * 123451);
    const bool isFalsePositiveForLargeValue =
        BloomFilter<>::mayContain(serializedBloom.data(), largeValueHash);
    EXPECT_EQ(isFalsePositiveForLargeValue, bloom.mayContain(largeValueHash));
    numFalsePositives += isFalsePositiveForLargeValue;
  }
  EXPECT_GT(2, 100 * numFalsePositives / kSize);
}

TEST_F(BloomFilterTest, merge) {
  constexpr int32_t kSize = 10;
  BloomFilter bloom;
  bloom.reset(kSize);
  for (auto i = 0; i < kSize; ++i) {
    bloom.insert(folly::hasher<int32_t>()(i));
  }

  BloomFilter merge;
  merge.reset(kSize);
  for (auto i = kSize; i < kSize + kSize; i++) {
    merge.insert(folly::hasher<int32_t>()(i));
  }

  std::string data;
  data.resize(bloom.serializedSize());
  merge.serialize(data.data());

  bloom.merge(data.data());

  for (auto i = 0; i < kSize + kSize; ++i) {
    EXPECT_TRUE(bloom.mayContain(folly::hasher<int32_t>()(i)));
  }
  EXPECT_FALSE(bloom.mayContain(folly::hasher<int32_t>()(kSize + 123451)));

  EXPECT_EQ(bloom.serializedSize(), merge.serializedSize());
}

TEST_F(BloomFilterTest, precision) {
  for (auto power = 0; power < 13; ++power) {
    meter<4>(16000 << power);
    meter<8>(16000 << power);
  }
}
