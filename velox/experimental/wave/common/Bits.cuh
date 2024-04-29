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

namespace facebook::velox::wave {

template <typename T, typename U>
inline void __device__ setBit(T* bits, U index, bool bit = true) {
  constexpr int32_t kShift = sizeof(T) == 1
      ? 3
      : sizeof(T) == 2 ? 4 : sizeof(T) == 4 ? 5 : sizeof(T) == 8 ? 6 : 0;
  constexpr U kMask = (static_cast<U>(1) << kShift) - 1;
  if (bit == true) {
    bits[index >> kShift] |= static_cast<T>(1) << (index & kMask);
  } else {
    bits[index >> kShift] &= (static_cast<T>(1) << ~(index & kMask));
  }
}

template <typename T, typename U>
inline bool __device__ isBitSet(T* bits, U index) {
  constexpr int32_t kShift = sizeof(T) == 1
      ? 3
      : sizeof(T) == 2 ? 4 : sizeof(T) == 4 ? 5 : sizeof(T) == 8 ? 6 : 0;
  constexpr U kMask = (static_cast<U>(1) << kShift) - 1;
  return (bits[index >> kShift] & static_cast<T>(1) << (index & kMask)) != 0;
}

} // namespace facebook::velox::wave
