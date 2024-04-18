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
    columns[0][i] = scale32((previous + delta + i) * kPrime32, keyRange);
  }
  counter += numRows;
  for (auto c = 1; c < numColumns; ++c) {
    for (auto r = 0; r < numRows; ++r) {
      columns[c][r] = c + (r & 7);
    }
  }
}


  
  
}
