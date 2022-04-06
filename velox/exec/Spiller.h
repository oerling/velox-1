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

#include "velox/common/memory/RowContainer.h"




// Describes a bit range inside a 64 bit hash number for use in
// partitioning data over multiple sets of spill files.
struct HashBitRange {
  // Low bit number of hash number bit range.
  uint8_t begin;
  // Bit number of first bit above the hash number bit range.
  uint8_t end;
};




  // Returns which spill partition 'hash' falls into. Returns -1 if the
  // partition of 'hash' has not been started.
  int32_t partition(uint64_t hash) const {
    int32_t field = (hash >> hashBits_.begin) & fieldMask_;
    return field < numPartitions_ ? field : -1;
  }


