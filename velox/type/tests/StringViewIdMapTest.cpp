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

#include "velox/type/StringViewIdMap.h"
#include <folly/container/F14Set.h>
#include <folly/hash/Hash.h>
#include "velox/common/base/RawVector.h"
#include "velox/common/base/SelectivityInfo.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

struct IdMapHasher {
  size_t operator()(const std::pair<StringView, int32_t>& item) const {
    return folly::hasher<StringView>()(item.first);
  }
};

struct IdMapComparer {
  bool operator()(
      const std::pair<StringView, int32_t>& left,
      const std::pair<StringView, int32_t>& right) const {
    return left.first == right.first;
  }
};

class F14IdMap {
 public:
  F14IdMap(int32_t initial) : set_(initial) {}

  int32_t id(StringView value) {
    std::pair<StringView, int32_t> item(
        value, static_cast<int32_t>(set_.size()));
    return set_.insert(item).first->second;
  }

  int64_t findId(StringView value) {
    std::pair<StringView, int32_t> item(
        value, static_cast<int32_t>(set_.size() + 1));
    auto it = set_.find(item);
    if (it == set_.end()) {
      return StringViewIdMap::kNotFound;
    }
    return it->second;
  }

  auto size() const {
    return set_.size();
  }

  void reserve(size_t size) {
    set_.reserve(size);
  }

  void clear() {
    set_.clear();
  }

 private:
  folly::F14FastSet<std::pair<StringView, int32_t>, IdMapHasher, IdMapComparer>
      set_;
};

class StringViewIdMapTest : public testing::Test {
 protected:
  static constexpr int32_t kBatchSize = 8;

  static void SetUpTestCase() {}

  void SetUp() override {}

  void testCase(int64_t size, int64_t range, int32_t maxLength) {
    std::vector<StringView> data;
    testData(size, range, maxLength, data);
    auto result = test(data);
    std::cout
        << fmt::format(
               "Size={} range={} 4-{} byte key clocks IdMap={} SIMD={} F14={} ({}%)",
               size,
               range,
               maxLength,
               result.mapTime,
               result.simdMapTime,
               result.f14Time,
               100 * result.f14Time / result.mapTime)
        << std::endl;
  }

  void testData(
      int64_t size,
      int64_t range,
      int32_t maxLength,
      std::vector<StringView>& data) {
    size = bits::roundUp(size, kBatchSize);
    data.reserve(size);
    std::string string;
    string.resize(maxLength);
    for (auto i = 0; i < size; ++i) {
      auto string = fmt::format("{}", (1 + (i % range)) * 123456789);
      int32_t targetSize = std::max(4, i % maxLength);
      while (string.size() < targetSize) {
        string = string + string;
      }
      string.resize(targetSize);
      data.push_back(
          StringView(stringBody(string.data(), string.size()), string.size()));
    }
  }

  struct Times {
    float mapTime;
    float simdMapTime;
    float f14Time;
  };

  // Feeds 'data' into a StringViewIdMap with both scalar and SIMD mode   and
  // the F14IdMap reference implementation and checks that the outcome is the
  // same. returns the total clocks for the three cases.
  Times test(const std::vector<StringView>& data) {
    StringViewIdMap map(1024);
    StringViewIdMap simdMap(1024);
    F14IdMap f14(1024);
    constexpr int32_t kNumRepeats = 10;
    SelectivityInfo mapInfo;
    SelectivityInfo simdMapInfo;
    SelectivityInfo f14Info;
    int32_t startIndices[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    for (auto counter = 0; counter < kNumRepeats; ++counter) {
      {
        SelectivityTimer t(mapInfo, data.size());
        char* copyPtr[8] = {};
        for (auto i = 0; i + kBatchSize <= data.size(); i += kBatchSize) {
          int32_t indices[kBatchSize];
          (xsimd::load_unaligned(&startIndices[0]) + i)
              .store_unaligned(&indices[0]);
          int32_t ids[kBatchSize];
          map.findIds8Scalar(data.data(), indices, true, ids, copyPtr);
        }
      }
      if (counter < kNumRepeats - 1) {
        map.clear();
      }
    }

    for (auto counter = 0; counter < kNumRepeats; ++counter) {
      {
        SelectivityTimer t(simdMapInfo, data.size());
        char* copyPtr[8] = {};
        for (auto i = 0; i + kBatchSize <= data.size(); i += kBatchSize) {
          int32_t indices[kBatchSize];
          (xsimd::load_unaligned(&startIndices[0]) + i)
              .store_unaligned(&indices[0]);
          int32_t ids[kBatchSize];
          simdMap.findIds8(data.data(), indices, true, ids, copyPtr);
        }
      }
      if (counter < kNumRepeats - 1) {
        simdMap.clear();
      }
    }

    
    for (auto counter = 0; counter < kNumRepeats; ++counter) {
      {
        SelectivityTimer t(f14Info, data.size());
        for (auto i = 0; i < data.size(); ++i) {
          f14.id(data[i]);
        }
      }

      if (counter < kNumRepeats - 1) {
        auto size = f14.size();
        f14.clear();
        f14.reserve(size);
      }
    }

    for (auto i = 0; i + kBatchSize <= data.size(); i += kBatchSize) {
      int32_t indices[8];
      (xsimd::load_unaligned(&startIndices[0]) + i)
          .store_unaligned(&indices[0]);
      int32_t ids[kBatchSize];

      map.findIds8Scalar(data.data(), indices, false, ids, nullptr);

      for (auto j = 0; j < kBatchSize; ++j) {
        auto reference = f14.findId(data[i + j]);
        EXPECT_EQ(reference, ids[j]);
        if (reference != ids[j]) {
          break;
        }
      }
    }

    return {
        mapInfo.timeToDropValue(),
        simdMapInfo.timeToDropValue(),
        f14Info.timeToDropValue()};
  }

  // If 'size' is over inline size, saves the characters in 'this' and returns
  // a pointer to the start, else returns 'string'
  char* stringBody(char* string, int32_t size) {
    if (size <= StringView::kInlineSize) {
      return string;
    }
    if (strings_.empty() ||
        strings_.back().size() + size > strings_.back().capacity()) {
      strings_.emplace_back();
      strings_.back().reserve(1'000'000);
    }
    auto start = strings_.back().size();
    strings_.back().resize(strings_.back().size() + size);
    memcpy(strings_.back().data() + start, string, size);
    return strings_.back().data() + start;
  }

  std::vector<raw_vector<char>> strings_;
};

TEST_F(StringViewIdMapTest, basic) {
  testCase(1000, 3, 12);
  return;
  testCase(1000, 3, 40);
  testCase(1000, 1000, 12);
  testCase(1000, 1000, 40);
  testCase(10000, 2500, 40);
  testCase(1000000, 1000000, 40);
  testCase(5000000, 1000000, 30);
}

#if 0
TEST_F(StringViewIdMapTest, zerosAndMasks) {
  constexpr int64_t kNotFound = BigintIdMap::kNotFound;

  BigintIdMap map(1024, *pool_);
  int64_t zeros[4] = {0, 0, 0, 0};
  int64_t oneZero[4] = {1, 0, 2, 3};

  // All lanes disabled makes all 0.
  expect4(0, 0, 0, 0, makeIds4(map, oneZero, 0));

  // Last lane is on, gets first id 1.
  expect4(0, 0, 0, 1, makeIds4(map, oneZero, 8));

  // All lanes are on, the zero gets the next id (2) and the non-zeros get 3
  // and 4.
  expect4(3, 2, 4, 1, makeIds4(map, oneZero));
  expect4(3, 2, 4, 1, findIds4(map, oneZero));

  // All zeros gets 2 (id of 0)  for the active lanes and 0 for inactive.
  expect4(2, 0, 2, 0, makeIds4(map, zeros, 5));

  expect4(2, 2, 2, 2, findIds4(map, zeros));

  BigintIdMap mapWithNoZero(1024, *pool_);
  // We insert the same values and mask out the 0.
  expect4(1, 0, 2, 3, makeIds4(mapWithNoZero, oneZero, 13));
  expect4(
      kNotFound,
      kNotFound,
      kNotFound,
      kNotFound,
      findIds4(mapWithNoZero, zeros));

  // Zero for inactive, not found for active.
  expect4(kNotFound, 0, kNotFound, 0, findIds4(mapWithNoZero, zeros, 5));

  int64_t mix[4] = {10, 1, 0, 2};
  expect4(kNotFound, 1, kNotFound, 2, findIds4(mapWithNoZero, mix));

  expect4(kNotFound, 0, kNotFound, 0, findIds4(mapWithNoZero, mix, 5));
}

TEST_F(StringViewIdMapTest, collisions) {
  constexpr int64_t kNotFound = StringViewIdMap::kNotFound;
  // We check the found and not found stay the same as the table gets filled
  F14IdMap reference(32);
  BigintIdMap map(8, *pool_);
  std::vector<int64_t> data;
  for (auto i = 0; i < 2048; ++i) {
    data.push_back((i + 1) * 0xfeedda7a58ff1e00);
  }
  // Add an empty marker.
  data[1333] = 0;
  for (auto fill = 0; fill < data.size(); fill += kBatchSize) {
    // Check that data not inserted is not found.
    for (auto i = fill; i < data.size(); i += kBatchSize) {
      auto expectedEmpty = map.findIds(xsimd::load_unaligned(data.data() + i));
      for (auto j = 0; j < kBatchSize; ++j) {
        EXPECT_EQ(kNotFound, reinterpret_cast<int64_t*>(&expectedEmpty)[j]);
        EXPECT_EQ(kNotFound, reference.findId(data[i + j]));
      }
    }

    // Add a group of 4 new entries.
    auto ids = map.makeIds(xsimd::load_unaligned(data.data() + fill));
    for (auto j = 0; j < kBatchSize; ++j) {
      // If there is a zero added, add it to 'reference' before the other values
      // to match the special treatment of empty marker.
      if (data[fill + j] == 0) {
        reference.id(0);
      }
    }
    for (auto j = 0; j < kBatchSize; ++j) {
      EXPECT_EQ(
          reference.id(data[fill + j]), reinterpret_cast<int64_t*>(&ids)[j]);
    }

    // Check that all inserted is still found.
    for (auto i = 0; i <= fill; i += kBatchSize) {
      auto ids = map.findIds(xsimd::load_unaligned(data.data() + i));
      for (auto j = 0; j < kBatchSize; ++j) {
        EXPECT_EQ(
            reference.findId(data[i + j]), reinterpret_cast<int64_t*>(&ids)[j]);
      }
    }
  }
}
#endif
