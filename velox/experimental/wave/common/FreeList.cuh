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


  /// Lock free stack of pointers into an arena of up to 4GB. The entries are offsets from the base address of the arena.
struct FreeList {
  static constexpr uint32_t kEmpty = ~0;

  void push(uint32_t item) {
    for (;;) {
      auto free = cub::ThreadLoad<cub::LOAD_CV>(&freeRows);
      *reinterpret_cast<uint32_t*>(row) = free;
      __threadfence();
      if (free == atomicCAS(&freeRows, free, offset)) {
        return;
      }
    }
  }

  uuint32_t pop() {
    for (;;) {
      uint64_t free =
          cub::ThreadLoad<cub::LOAD_CG>(reinterpret_cast<uint32_t*>(&freeRows));
      if (free == kEmpty) {
        return kEmpty;
      }
      uint64_t counter = 1 + atomicAdd(&numPops, 1);
      free =
          cub::ThreadLoad<cub::LOAD_CG>(reinterpret_cast<uint32_t*>(&freeRows));
      if (free == kEmpty) {
        return kEmpty;
      }
      uint32_t next = cub::ThreadLoad<cub::LOAD_CG>(
          reinterpret_cast<uint32_t*>(base + free));
      unsigned long long freeAndCount = free | (counter << 32);
      unsigned long long nextFreeAndCount = next | (counter << 32);
      if (freeAndCount ==
          atomicCAS(
              reinterpret_cast<unsigned long long*>(&freeRows),
              freeAndCount,
              nextFreeAndCount)) {
        return free;
      }
    }
  }

  uint64_t base{0};
// Offset of first in free list of rows. Align at 8 bytes.
  uint32_t freeRows{kEmpty};

  // counter of pops from fre list. Must be upper half of 64 bit word with
  // freeRows as lower half. Use for lock free ABA magic.
  uint32_t numPops{0};

};


}
