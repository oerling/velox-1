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




#include <cstdint.h>

/// Structs for tagged hash table. Can be inclued in both Velox .cpp and .cu.
namespace facebook::velox::wave {

/// A 32 byte tagged bucket with 4 tags, 4 flag bytes and 4 6-byte
/// pointers. Fits in one 32 byte GPU cache sector.
struct Bucket {
  uint32_t tags;
  uint32_t flags;
  short data[12];
};

/// A set of rows 
struct RowAllocator {
  int32_t fill{0};
  int32_t capacity{0};
  // Array of pointers to starts of preallocated rows.
  char** rows{nullptr};
};

 enum class probeState : uint8_t {kInit, kDone, kMoreValues, kNeedSpace, kRetry };
  
struct HashProbe {
  /// Count of probe keys.
  int32_t numKeys;

  /// Hash numbers for probe keys.
  uint64_t* hashes;

  /// List of input rows to retry in kernel. Sized to one per row of input. Used inside kernel, not meaningful after return.
  int32_t* kernelRetries;

  /// List of input rows to retry after host updated state. Sized to one per row of input.
  int32_t* hostRetries;

  /// ount of valid items in 'hostRetries'.
  int32_t numHostRetries;
  

  Probestatus* status;

  // Optional payload rows hitting from a probe.
  void** hits{nullptr};
};
 
struct HashTable {
  Bucket* buckets;
  uint32_t sizeMask;
  

  // Translates a hash number to a partition number '(hash &
  // partitionMask) >> partitionShift' is a partition number used as
  // a physical partition of the table. Used as index into 'allocators'.
  uint32_t partitionMask{0};
  uint8_t partitionShift{0};
  
  RowAllocators* rowAllocators;
};

}
