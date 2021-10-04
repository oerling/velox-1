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

#pragma once

#include <cstdint>
#include <vector>

#include <folly/Hash.h>

#include "velox/common/base/BitUtil.h"

namespace facebook::velox {

class Bloom {
 public:
  Bloom(int32_t capacity) : bits_(bits::nextPowerOfTwo(capacity)) {}

  // Adds 'value'.
  void insert(uint64_t value) {
    set(bits_.data(), bits_.size(), folly::hasher<uint64_t>()(value));
  }

  bool mayContain(uint64_t value) {
    return test(bits_.data(), bits_.size(), folly::hasher<uint64_t>()(value));
  }

  // We use 4 independent hash functions by taking 24 bits of
  // the hash code and breaking these up into 4 groups of 6 bits. Each group
  // represents a number between 0 and 63 (2^6-1) and maps to one bit in a
  // 64-bit number. We combine these to get a 64-bit number with up to 4 bits
  // set.
  inline static uint64_t bloomMask(uint64_t hashCode) {
    return (1L << (hashCode & 63)) | (1L << ((hashCode >> 6) & 63)) |
        (1L << ((hashCode >> 12) & 63)) | (1L << ((hashCode >> 18) & 63));
  }

  // Skip 24 bits used for bloomMask and use the next N bits of the hash code
  // as index. N = log2(bloomSize). bloomSize must be a power of 2.
  inline static uint32_t bloomIndex(uint32_t bloomSize, uint64_t hashCode) {
    return ((hashCode >> 24) & (bloomSize - 1));
  }

  inline static void
  set(uint64_t* FOLLY_NONNULL bloom, int32_t bloomSize, uint64_t hashCode) {
    auto mask = bloomMask(hashCode);
    auto index = bloomIndex(bloomSize, hashCode);
    bloom[index] |= mask;
  }

  inline static bool test(
      const uint64_t* FOLLY_NONNULL bloom,
      int32_t bloomSize,
      uint64_t hashCode) {
    auto mask = bloomMask(hashCode);
    auto index = bloomIndex(bloomSize, hashCode);
    return mask == (bloom[index] & mask);
  }

 private:
  std::vector<uint64_t> bits_;
};

} // namespace facebook::velox
