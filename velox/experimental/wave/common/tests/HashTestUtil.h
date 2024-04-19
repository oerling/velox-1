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
#include "velox/experimental/wave/common/HashTable.h"

namespace facebook::velox::wave {

  /// Describes a hashtable benchmark case.
struct HashRun {
  //CPU/GPU measurement.
  bool isCpu;

  // Number of slots in table.
    int32_t numSlots;

    // Number of probe rows.
    int32_t numRows;

    // Number of distinct keys.
    int32_t numDistinct;

    // Number of distinct hot keys.
    int32_t numHot;

    // Percentage of hot keys over total keys. e.g. with 1000 distinct and 10 hot and hotPct of 50, every second key will be one of 10 and the rest are evenly spread over the 1000.
    int32_t hotPct{0};

  // Number of keys processed by each thread of each block.
  int32_t rowsPerThread;

  // Number of blocks of 256 threads.
  int32_t numBlocks;

  // Number of columns. Key is column 0.
  uint8_t numColumns{1};
  
  // Number of independent hash tables.
  int32_t numTables;

  // Rows processed per second on GPU/CPU.
  float gpuRPS;
  float cpuRPS;

  // Input data, not owned, resident on GPU if GPU run.
  HashProbe* input;
  
  std::string toString() {
    return fmt::format("");
  }
};
 
  
  void makeInput(
    int32_t numRows,
    int32_t keyRange,
    int32_t powerOfTwo,
    int64_t counter,
    uint8_t numColumns,
    int64_t** columns,
    int32_t numHot = 0,
		 int32_t hotPct = 0);

inline uint32_t scale32(uint32_t n, uint32_t scale) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(n)) * scale) >> 32;
}

  
}

