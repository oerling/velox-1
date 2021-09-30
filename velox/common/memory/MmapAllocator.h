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

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/MappedMemory.h"

namespace facebook::velox::memory {

// Denotes a number of pages of one size class, i.e. one page consists
// of a size class dependent number of consecutive machine pages.
using ClassPageCount = int32_t;

struct MmapAllocatorOptions {
  //  Capacity in bytes, defult 512MB
  uint64_t capacity = 1L << 29;
};

class MmapAllocator : public MappedMemory {
 public:
  class ContiguousAllocation {
   public:
    ContiguousAllocation() = default;
    ~ContiguousAllocation() {
      if (auto allocator = dynamic_cast<MmapAllocator*>(mappedMemory_)) {
        allocator->freeContiguous(*this);
      } else {
        ::free(data_);
      }
      data_ = nullptr;
    }

    MappedMemory* mappedMemory() const {
      return mappedMemory_;
    }

    MachinePageCount numPages() {
      return bits::roundUp(size_, kPageSize) / kPageSize;
    }

    template <typename T = uint8_t>
    T* data() {
      return reinterpret_cast<T*>(data_);
    }

    uint64_t size() const {
      return size_;
    }

    void reset(MappedMemory* mappedMemory, void* data, uint64_t size) {
      mappedMemory_ = mappedMemory;
      data_ = data;
      size_ = size;
    }

   private:
    MappedMemory* mappedMemory_ = nullptr;
    void* data_ = nullptr;
    uint64_t size_ = 0;
  };

  explicit MmapAllocator(const MmapAllocatorOptions& options);

  virtual ~MmapAllocator() = default;

  bool allocate(
      MachinePageCount numPages,
      int32_t owner,
      Allocation& out,
      std::function<void(int64_t)> beforeAllocCB = nullptr,
      MachinePageCount minSizeClass = 0) override;

  int64_t free(Allocation& allocation) override;

  // Makes a contiguous mmap of 'numPages'. Advises away the required
  // number of free pages so as not to have resident size exceed
  // 'capacity_'. Returns false if sufficient free pages do not
  // exist. If 'collateral' or 'largeCollateral' are non-null their
  // contents are freed to provide building materials for the new
  // allocation. In all cases these will be empty before return,
  // regardless of success.
  virtual bool allocateContiguous(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation* largeCollateral,
      ContiguousAllocation& allocation);

  virtual void freeContiguous(ContiguousAllocation& allocation);

  // Checks internal consistency of allocation data
  // structures. Returns true if OK.
  bool checkConsistency() override;

  const std::vector<MachinePageCount>& sizes() const override {
    return sizes_;
  }

  MachinePageCount capacity() const {
    return capacity_;
  }

  MachinePageCount numAllocated() const override {
    return numAllocated_;
  }

  MachinePageCount numMapped() const override {
    return numMapped_;
  }

  std::string toString() const override;

 private:
  static constexpr uint64_t kAllSet = 0xffffffffffffffff;

  class SizeClass {
   public:
    SizeClass(size_t capacity, MachinePageCount pageSize);

    ~SizeClass();

    MachinePageCount pageSize() const {
      return pageSize_;
    }

    // Allocates 'numPages' from 'this' and appends these to
    // *out. '*numUnmapped' is incremented by the number of pages that
    // are not backed by memory.
    bool allocate(
        ClassPageCount numPages,
        int owner,
        MachinePageCount* numUnmapped,
        MappedMemory::Allocation* out);

    // Frees all pages of 'allocation' that fall in this size
    // class. Erases the corresponding runs from 'allocation'.
    MachinePageCount free(Allocation* allocation);

    ClassPageCount checkConsistency(ClassPageCount* numMapped);

    MachinePageCount adviseAway(
        MachinePageCount numPages,
        MmapAllocator* allocator);

    void setAllMapped(Allocation* allocation, bool value);
    void setMappedBits(MappedMemory::PageRun run, bool value);
    bool isAllocated(uint8_t* ptr) {
      if (ptr >= address_ && ptr < address_ + capacity_ * pageSize_) {
        uint64_t offset = (ptr - address_) / pageSize_;
        return (pageAllocated_[offset / 64] & (1L << offset & 63)) != 0;
      }
      return false;
    }
    bool isInRange(uint8_t* ptr) {
      if (address_ > ptr || address_ + byteSize_ <= ptr) {
        return false;
      }
      // See that ptr falls on a page boundary.
      if ((ptr - address_) % pageSize_ != 0) {
        VELOX_FAIL("Pointer is in a SizeClass but not at page boundary");
      }
      return true;
    }
    std::string toString() const;

   private:
    bool allocateLocked(
        ClassPageCount numPages,
        int owner,
        MachinePageCount* numUnmapped,
        MappedMemory::Allocation* out);

    void adviseAway(Allocation* allocation);

    void allocMapped(
        int wordIndex,
        uint64_t candidates,
        ClassPageCount* needed,
        MappedMemory::Allocation* allocation);

    void allocAny(
        int wordIndex,
        ClassPageCount* needed,
        MachinePageCount* numUnmapped,
        Allocation* allocation);

    size_t capacity_;
    MachinePageCount pageSize_;
    uint8_t* address_;
    size_t byteSize_;
    int clockHand_ = 0;
    // Count of free pages backed by memory.
    ClassPageCount numMappedFreePages_ = 0;
    // 1 bit per page.
    std::vector<uint64_t> pageAllocated_;
    std::vector<uint64_t> pageMapped_;
    std::vector<int> pageOwner_;

    // Statistics.
    uint64_t numAllocatedMapped_ = 0;
    uint64_t numAllocatedUnmapped_ = 0;
    uint64_t numAdvisedAway_ = 0;
    std::mutex mutex_;
  };

  // Marks all the pages backing 'out' to be mapped if one can safely
  // write up to 'numMappedNeeded' pages that have no backing
  // memory. Returns true on success. Returns false if enough backed
  // but unused pages from other size classes cannot be advised
  // away. On failure, also frees 'out'. since the whole allocation
  // could not be completed.
  bool ensureEnoughMappedPages(int32_t newMappedNeeded, Allocation& out);

  // Frees 'allocation and returns the number of freed pages. Does not
  // update 'numAllocated'.
  MachinePageCount freeInternal(Allocation& allocation);

  // Returns the index of the size class 'address' falls in. Throws if 'address'
  // is not in any.
  int sizeClass(uint8_t* address);

  void markAllMapped(Allocation* allocation);

  MachinePageCount adviseAway(MachinePageCount target);

  std::atomic<MachinePageCount> numAllocated_;
  // When using mmap/madvise, the current number pages backed by memory.
  std::atomic<MachinePageCount> numMapped_;
  MachinePageCount capacity_ = 0;
  // The machine page counts corresponding to different sizes in order
  // of increasing size.
  const std::vector<MachinePageCount> sizes_;

  std::vector<std::unique_ptr<SizeClass>> sizeClasses_;
  std::mutex mutex_;
  // Statistics. Not atomic.
  uint64_t numAllocations_ = 0;
  uint64_t numFrees_ = 0;
  uint64_t numAllocatedPages_ = 0;
  uint64_t numAdvisedPages_ = 0;
};

} // namespace facebook::velox::memory
