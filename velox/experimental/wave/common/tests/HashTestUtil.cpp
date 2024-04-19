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

#include "velox/experimental/wave/common/test/HashTestUtil.h"


namespace facebook::velox::wave {


void makeInput(
    int32_t numRows,
    int32_t keyRange,
    int32_t powerOfTwo,
    int64_t counter,
    uint8_t numColumns,
    int64_t** columns,
    int32_t numHot,
	       int32_t hotPct) {
  int32_t delta = counter & (powerOfTwo - 1);
  for (auto i = 0; i < numRows; ++i) {
    auto previous = columns[0][i];
    auto seed = (previous + delta + i) * kPrime32;
    if (hotPct && scale32(seed >> 32, 100) <= hotPct) {
      int32_t nth = scale32(seed, numHot);
      nth = std::min<int64_t>(keyRange - 1, nth * (static_cast<float>(keyRange) / nth));
      columns[0][i] = nth;
    } else {
      columns[0][i] = scale32(seed, keyRange);
    }
  }
  counter += numRows;
  for (auto c = 1; c < numColumns; ++c) {
    for (auto r = 0; r < numRows; ++r) {
      columns[c][r] = c + (r & 7);
    }
  }
}


  
  
}
