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

#include <folly/container/F14Set.h>
#include <folly/hash/Hash.h>
#include "velox/common/base/BigintIdMap.h"
#include "velox/common/base/SelectivityInfo.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

struct IdMapHasher {
  size_t operator()(const std::pair<int64_t, int32_t>& item) const {
    return folly::hasher<int64_t>()(item.first);
  }
};

struct IdMapComparer {
  bool operator()(
      const std::pair<int64_t, int32_t>& left,
      const std::pair<int64_t, int32_t>& right) const {
    return left.first == right.first;
  }
};

class F14IdMap {
 public:
  F14IdMap(int32_t initial) : set_(initial) {}

  int32_t id(int64_t value) {
    std::pair<int64_t, int32_t> item(
        value, static_cast<int32_t>(set_.size() + 1));
    return set_.insert(item).first->second;
  }

 private:
  folly::F14FastSet<std::pair<int64_t, int32_t>, IdMapHasher, IdMapComparer>
      set_;
};

class IdMapTest : public testing::Test {
 protected:
  void SetUp() override {
    root_ = memory::MemoryManager::getInstance().addRootPool("IdMapRoot");
    pool_ = root_->addLeafChild("IdMapLeakLeaf");
  }

  void testCase(int64_t size, int64_t range) {
    std::vector<int64_t> data;
    testData(size, range, data);
    auto result = test(data);
    std::cout << fmt::format(
                     "Size={} range={} clocks IdMap={} F14={} ({}%)",
                     size,
                     range,
                     result.first,
                     result.second,
                     100 * result.second / result.first)
              << std::endl;
  }

  void testData(int64_t size, int64_t range, std::vector<int64_t>& data) {
    size = bits::roundUp(size, 4);
    data.reserve(size);
    for (auto i = 0; i < size; ++i) {
      data.push_back(1 + (i % range));
    }
  }

  // Feeds 'data' into a BigintIdMap and the F14IdMap reference implementation
  // and checks that the outcome is the same. returns the total clocks for
  // BigIntIdMap and F14IdMap.
  std::pair<float, float> test(const std::vector<int64_t>& data) {
    BigintIdMap map(1024, *pool_);
    F14IdMap f14(1024);
    SelectivityInfo mapInfo;
    SelectivityInfo f14Info;
    {
      SelectivityTimer t(mapInfo, data.size());
      for (auto i = 0; i + 4 <= data.size(); i += 4) {
        map.makeIds(xsimd::batch<int64_t>::load_unaligned(data.data() + i));
      }
    }
    {
      SelectivityTimer t(f14Info, data.size());
      for (auto i = 0; i < data.size(); ++i) {
        f14.id(data[i]);
      }
    }
    for (auto i = 0; i + 4 <= data.size(); i += 4) {
      auto ids =
          map.makeIds(xsimd::batch<int64_t>::load_unaligned(data.data() + i));
      auto idsArray = reinterpret_cast<int64_t*>(&ids);
      for (auto j = 0; j < 4; ++j) {
        auto reference = f14.id(data[i + j]);
        EXPECT_EQ(reference, idsArray[j]);
        if (reference != idsArray[j]) {
          break;
        }
      }
    }
    return std::make_pair<float, float>(
        mapInfo.timeToDropValue(), f14Info.timeToDropValue());
  }

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(IdMapTest, basic) {
  testCase(1000, 3);
  testCase(1000, 1000);
  testCase(10000, 2500);
  testCase(1000000, 1000000);
  testCase(5000000, 1000000);
}
