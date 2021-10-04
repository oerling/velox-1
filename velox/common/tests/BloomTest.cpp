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

#include "velox/common/base/Bloom.h"
#include <folly/Random.h>

#include <gtest/gtest.h>

using namespace facebook::velox;
TEST(BloomTest, basic) {
  constexpr int32_t kWords 128;
  constexpr int32_t kSize = kWords * 8;
  folly::Random::DefaultGenerator rng;
  rng.seed(1);
  std::unordered_set<uint64_t> reference;
  std::vector<uint64_t> bits(128);
  // We insert kSize random values and check that they are there.
  for (auto i = 0; i < kSize; ++i) {
    auto value = folly::Random::rand64(rng);
    Bloom::set(bits, kWords, value);
    reference.insert(value);
  }
  for (auto value : reference) {
    EXPECT_TRUE(Bloom::test(bits, kWords, value));
  }
  int32_t hits = 0;
  for (auto i = 0; i < kSize; ++i) {
    auto value = folly::Random::rand64(rng);
    hits += Bloom::test(bits, kWords, value);
  }
  EXPECT_GT(20, hits);
}
  



