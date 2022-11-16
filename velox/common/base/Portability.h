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

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace facebook::velox {

  inline size_t count_trailing_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_ctzll(x);
}

inline size_t count_leading_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_clzll(x);
}

#if defined(__GNUC__) || defined(__clang__)
#define INLINE_LAMBDA __attribute__((__always_inline__))
#else
#define INLINE_LAMBDA
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  template <typename T>
  using asan_atomic = std::atomic<T>;

  template <typename T>
inline   T atomicValue(const std::atomic<T>& x) {
    return x;
  }
#else
  template <typename T>
    using asan_atomic = T;

  template <typename T>
inline   T atomicValue(T x) {
    return x;
  }

#endif
#else
  template <typename T>
    using asan_atomic = T;

  template <typename T>
inline   T atomicValue(T x) {
    return x;
  }
#endif

template <typename T>
  inline void resizeAsanAtomic(std::vector<asan_atomic<T>>& vector, int32_t newSize) {
  std::vector<asan_atomic<T>> newVector(newSize);
  auto numCopy = std::min<int32_t>(newSize, vector.size());
  for (auto i = 0; i < numCopy; ++i) {
    newVector[i] = atomicValue(vector[i]);
  }
  vector = std::move(newVector);
}
}
