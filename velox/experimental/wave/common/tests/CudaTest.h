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

#include "velox/experimental/wave/common/Cuda.h"

/// Sample header for testing Cuda.h

namespace facebook::velox::wave {


  /// Struct for the state of a probe into MockTable. A struct of arrays. Each thread fills its state at threadIdx.x so the neighbors can decide who does what.
  struct MockProbe {
    static constexpr int32_t kMinBlockSize = 128;
    static constexpr int32_t kMaxBlocks = 8192 / kMinBlockSize;
    // First row of blockIdx.x. Index into 'partitions' of the update kernel.
    int32_t begin[kMaxBlocks];
    // first row of next TB.
    int32_t end[kMaxBlocks];

    // The row of input that corresponds to the probe.
    int32_t* start;
    // Whether the position is a hit.
    bool* isHit;
    // Whether the probe needs to cross to the next partition.
    bool* overflow;
  };

/// Hash table, works in CPU and GPU.
struct MockTable {
  int32_t sizeMask;
  // Size / 64K.o
  int32_t partitionSize{0};
  // Mask to get partition base.
  int32_t partitionMask{0};
  int64_t** rows;
  int32_t numRows{0};
  // Size of row, includes keys and dependents, aligned to 8.
  int32_t rowSize{0};
  // Number of dependent columns.
  uint8_t numColumns;
  // Payload.
  char* columns{nullptr};
};

  
struct WideParams {
  int32_t size;
  int32_t* numbers;
  int32_t stride;
  int32_t repeat;
  char data[4000];
  void* result;
};

class TestStream : public Stream {
 public:
  // Queues a kernel to add 1 to numbers[0...size - 1]. The kernel repeats
  // 'repeat' times.
  void addOne(int32_t* numbers, int size, int32_t repeat = 1);

  void addOneWide(int32_t* numbers, int32_t size, int32_t repeat = 1);

  void addOneRandom(
      int32_t* numbers,
      const int32_t* lookup,
      int size,
      int32_t repeat = 1);

  static int32_t sort8KTempSize();
  
  // Makes random lookup keys and increments, starting at 'startCount' columns[0] is keys.
  void makeInput(int32_t numRows, int32_t keyRange, int32_t startCount, uint8_t numColumns, int64_t** columns);
  
  /// Calculates a hash of each key, stores it in hash and then calculates a 16 bit partition number. Gives each row a sequence number. Sorts by partition, so that the row numbers are in partition order in 'rows'. 
  void hashAndPartition8K(int32_t numRows, int64_t* keys, uint64_t* hashes, uint16_t* partitions, uint16_t* rows);

  void update8K(int32_t numRows, int64_t* key, uint64_t* hash, uint16_t* partitions, uint16_t* rowNumbers, int64_t** args, MockProbe* probe, MockTable* table);



};

} // namespace facebook::velox::wave
