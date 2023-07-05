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

#include "velox/common/memory/Memory.h"

namespace facebook::velox {
// A set of Allocations holding the fixed width payload
// rows. The Runs are filled to the end except for the last one. This
// is used for iterating over the payload for rehashing, returning
// results etc. This is used via HashStringAllocator for variable length
// allocation for backing ByteStreams for complex objects. In that case, there
// is a current run that is appended to and when this is exhausted a new run is
// started.
class AllocationPool {
 public:
  static constexpr int32_t kMinPages = 16;
  static constexpr int64_t kHugePageSize =
      memory::AllocationTraits::kHugePageSize;

  explicit AllocationPool(memory::MemoryPool* pool)
      : pool_(dynamic_cast<memory::MemoryPoolImpl*>(pool)) {
    VELOX_CHECK_NOT_NULL(pool_);
  }

  ~AllocationPool() {
    clear();
  }

  void clear();

  // Allocate a buffer from this pool, optionally aligned.  The alignment can
  // only be power of 2.
  char* allocateFixed(uint64_t bytes, int32_t alignment = 1);

  // Starts a new run for variable length allocation. The actual size
  // is at least one machine page. Throws std::bad_alloc if no space.
  void newRun(int32_t preferredSize);

  int32_t numRanges() const {
    return allocations_.size() + largeAllocations_.size();
  }

  /// Returns the indexth contiguous range. If the range is a large allocation,
  /// returns the hugepage aligned range of contiguous huge pages in the range.
  folly::Range<char*> rangeAt(int32_t index) const;

  int64_t currentOffset() const {
    return currentOffset_;
  }

  int64_t allocatedBytes() const {
    return usedBytes_;
  }

  // Returns number of bytes left at the end of the current run.
  int32_t availableInRun() const {
    return bytesInRun_ - currentOffset_;
  }

  // Returns pointer to first unallocated byte in the current run.
  char* firstFreeInRun() {
    VELOX_DCHECK_GT(availableInRun(), 0);
    return startOfRun_ + currentOffset_;
  }

  // Sets the first free position in the current run.
  void setFirstFreeInRun(const char* firstFree) {
    auto offset = firstFree - startOfRun_;
    VELOX_CHECK(
        offset >= 0 && offset <= bytesInRun_,
        "Trying to set end of allocation outside of last allocated run");
    currentOffset_ = offset;
  }

  memory::MemoryPool* pool() const {
    return pool_;
  }

  /// true if 'ptr' is inside the active allocation.
  bool isInCurrentAllocation(void* ptr) const {
    return reinterpret_cast<char*>(ptr) >= startOfRun_ &&
        reinterpret_cast<char*>(ptr) < startOfRun_ + bytesInRun_;
  }

  int64_t hugePageThreshold() const {
    return hugePageThreshold_;
  }

  /// Sets the size after which 'this' switches to large mmaps with huge pages.
  void setHugePageThreshold(int64_t size) {
    hugePageThreshold_ = size;
  }

 private:
  static constexpr int64_t kDefaultHugePageThreshold = 256 * 1024;
  static constexpr int64_t kMaxMmapBytes = 512 << 20; // 512 MB

  // Increses the reservation in 'pool_' when 'currentOffset_' goes past
  // 'reservedTo_'.
  void increaseReservation();

  void newRunImpl(memory::MachinePageCount numPages);

  memory::MemoryPoolImpl* pool_;
  std::vector<std::unique_ptr<memory::Allocation>> allocations_;
  std::vector<std::unique_ptr<memory::ContiguousAllocation>> largeAllocations_;
  char* startOfRun_{nullptr};
  int32_t bytesInRun_{0};
  int32_t currentOffset_ = 0;

  // Offset from 'startOfRun_' that is counted as reserved in 'pool_'. This can
  // be less than the mmapped range for large mmaps.
  int32_t reservedTo_{0};

  // Total explicit reservations made in 'pool_' for the items in
  // 'largeAllocations_'.

  int64_t largeReserved_{0};

  // Total space returned to users. Size of allocations can be larger specially
  // if mmapped in advance of use.
  int64_t usedBytes_{0};

  // Start using large mmaps with huge pages after 'usedBytes_' exceeds this.
  int64_t hugePageThreshold_{kDefaultHugePageThreshold};
};

} // namespace facebook::velox
