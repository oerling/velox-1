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

#include "velox/common/memory/AllocationPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox {

folly::Range<char*> AllocationPool::rangeAt(int32_t index) const {
  if (index < allocations_.size()) {
<<<<<<< HEAD
    auto run = allocations_[index]->runAt(0);
    return folly::Range<char*>(run.data<char>(), run.numBytes());
  }
  auto largeIndex = index - allocations_.size();
  if (largeIndex < largeAllocations_.size()) {
    memory::ContiguousAllocation& data = *largeAllocations_[largeIndex];
    return data.hugePageRange();
=======
    auto run = allocations_[index].runAt(0);
    return folly::Range<char*>(
        run.data<char>(),
        run.data<char>() == startOfRun_ ? currentOffset_ : run.numBytes());
  }
  auto largeIndex = index - allocations_.size();
  if (largeIndex < largeAllocations_.size()) {
    auto data = &largeAllocations_[largeIndex];
    auto range = data->hugePageRange();
    if (range.data() == startOfRun_) {
      return folly::Range<char*>(range.data(), currentOffset_);
    }
    return range;
>>>>>>> hp-pool-dev
  }
  VELOX_FAIL("Out of range index for rangeAt(): {}", index);
}

void AllocationPool::clear() {
  allocations_.clear();
  if (largeReserved_) {
    for (auto& large : largeAllocations_) {
<<<<<<< HEAD
      pool_->allocator()->freeContiguous(*large);
    }
    // The treservation may be a fraction of the size of the mmaps.
=======
      pool_->allocator()->freeContiguous(large);
    }
    // The reservation may be a fraction of the size of the mmaps.
>>>>>>> hp-pool-dev
    pool_->release(largeReserved_);
    largeAllocations_.clear();
    largeReserved_ = 0;
  }
  startOfRun_ = nullptr;
  bytesInRun_ = 0;
  currentOffset_ = 0;
  reservedTo_ = 0;
  usedBytes_ = 0;
}

char* AllocationPool::allocateFixed(uint64_t bytes, int32_t alignment) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  if (availableInRun() >= bytes && alignment == 1) {
    auto* result = startOfRun_ + currentOffset_;
    currentOffset_ += bytes;
    if (currentOffset_ > reservedTo_) {
      increaseReservation();
    }
    return result;
  }
  VELOX_CHECK_EQ(
      __builtin_popcount(alignment), 1, "Alignment can only be power of 2");

  auto numPages = memory::AllocationTraits::numPages(bytes + alignment - 1);

<<<<<<< HEAD

=======
>>>>>>> hp-pool-dev
  if (availableInRun() == 0) {
    newRunImpl(numPages);
  } else {
    auto alignedBytes =
        bytes + memory::alignmentPadding(firstFreeInRun(), alignment);
    if (availableInRun() < alignedBytes) {
      newRunImpl(numPages);
    }
  }
  currentOffset_ += memory::alignmentPadding(firstFreeInRun(), alignment);
  VELOX_CHECK_LE(bytes + currentOffset_, bytesInRun_);
  auto* result = startOfRun_ + currentOffset_;
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
  currentOffset_ += bytes;
  if (currentOffset_ > reservedTo_) {
    increaseReservation();
  }
  return result;
}

void AllocationPool::increaseReservation() {
  VELOX_CHECK_GT(bytesInRun_, kHugePageSize);
<<<<<<< HEAD
  int32_t moreNeeded =
      bits::roundUp(currentOffset_ - reservedTo_, kHugePageSize);
=======
  auto moreNeeded = bits::roundUp(currentOffset_ - reservedTo_, kHugePageSize);
>>>>>>> hp-pool-dev
  pool_->reserve(moreNeeded);
  largeReserved_ += moreNeeded;
  usedBytes_ += moreNeeded;
  reservedTo_ += moreNeeded;
<<<<<<< HEAD
}

void AllocationPool::newRunImpl(memory::MachinePageCount numPages) {
  if (usedBytes_ >= hugePageThreshold_) {
    // At least 16 huge pages, no more than kMaxMmapBytes. The next is
    // double the previous. Because the previous is a hair under the
    // power of two because of fractional pages at ends of allocation,
    // add an extra huge page size.
    int64_t nextSize = std::min(
        kMaxMmapBytes,
        std::max<int64_t>(
            16 * kHugePageSize,
            bits::nextPowerOfTwo(usedBytes_ + kHugePageSize)));
    pool_->reserve(kHugePageSize);
    auto largeAlloc = std::make_unique<memory::ContiguousAllocation>();
    pool_->allocator()->allocateContiguous(
        nextSize / memory::AllocationTraits::kPageSize, nullptr, *largeAlloc);
    auto range = largeAlloc->hugePageRange();
    startOfRun_ = range.data();
    bytesInRun_ = range.size();
    largeAllocations_.emplace_back(std::move(largeAlloc));
    currentOffset_ = 0;
    usedBytes_ += kHugePageSize;
    largeReserved_ += kHugePageSize;
    reservedTo_ = kHugePageSize;
    return;
    return;
  }
  memory::Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);
  VELOX_CHECK_EQ(allocation.numRuns(), 1);
  startOfRun_ = allocation.runAt(0).data<char>();
  bytesInRun_ = allocation.runAt(0).numBytes();
  currentOffset_ = 0;
  allocations_.push_back(
      std::make_unique<memory::Allocation>(std::move(allocation)));
  reservedTo_ = bytesInRun_;
  usedBytes_ += bytesInRun_;
=======
>>>>>>> hp-pool-dev
}

void AllocationPool::newRunImpl(memory::MachinePageCount numPages) {
  if (usedBytes_ >= hugePageThreshold_ ||
      numPages > pool_->sizeClasses().back()) {
    // At least 16 huge pages, no more than kMaxMmapBytes. The next is
    // double the previous. Because the previous is a hair under the
    // power of two because of fractional pages at ends of allocation,
    // add an extra huge page size.
    int64_t nextSize = std::min(
        kMaxMmapBytes,
        std::max<int64_t>(
            16 * kHugePageSize,
            bits::nextPowerOfTwo(usedBytes_ + kHugePageSize)));
    pool_->reserve(kHugePageSize);
    if (numPages * kPageSize + kHugePageSize > nextSize) {
      // Extra large single request.
      nextSize = numPages * kPageSize + kHugePageSize;
    }
    memory::ContiguousAllocation largeAlloc;
    pool_->allocator()->allocateContiguous(
        nextSize / kPageSize, nullptr, largeAlloc);
    auto range = largeAlloc.hugePageRange();
    startOfRun_ = range.data();
    bytesInRun_ = range.size();
    largeAllocations_.emplace_back(std::move(largeAlloc));
    currentOffset_ = 0;
    usedBytes_ += kHugePageSize;
    largeReserved_ += kHugePageSize;
    reservedTo_ = kHugePageSize;
    return;
  }
  memory::Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);
  VELOX_CHECK_EQ(allocation.numRuns(), 1);
  startOfRun_ = allocation.runAt(0).data<char>();
  bytesInRun_ = allocation.runAt(0).numBytes();
  currentOffset_ = 0;
  allocations_.push_back(std::move(allocation));
  reservedTo_ = bytesInRun_;
  usedBytes_ += bytesInRun_;
}

void AllocationPool::newRun(int64_t preferredSize) {
  newRunImpl(memory::AllocationTraits::numPages(preferredSize));
}

} // namespace facebook::velox
