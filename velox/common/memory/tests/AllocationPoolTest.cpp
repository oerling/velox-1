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

#include <folly/container/F14Map.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace facebook::velox;

class AllocationPoolTest : public testing::Test {
 protected:
  void SetUp() override {
<<<<<<< HEAD
    auto root_ = memory::MemoryManager::getInstance().addRootPool("allocationPoolTestRoot");
=======
    auto root_ = memory::MemoryManager::getInstance().addRootPool(
        "allocationPoolTestRoot");
>>>>>>> hp-pool-dev
    pool_ = root_->addLeafChild("leaf");
  }

  // Writes a byte at pointer so we see RSS change.
  void setByte(void* ptr) {
    *reinterpret_cast<char*>(ptr) = 1;
  }
<<<<<<< HEAD
  
=======

>>>>>>> hp-pool-dev
  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(AllocationPoolTest, hugePages) {
  auto allocationPool = std::make_unique<AllocationPool>(pool_.get());
  allocationPool->setHugePageThreshold(128 << 10);
  int32_t counter = 0;
  for (;;) {
<<<<<<< HEAD
    allocationPool->newRun(32 << 10);
    // Initial allocations round up to 64K
    EXPECT_EQ(1, allocationPool->numRanges());
    EXPECT_EQ(allocationPool->rangeAt(0).size(), 64 << 10);
=======
    int32_t usedKB = 0;
    allocationPool->newRun(32 << 10);
    // Initial allocations round up to 64K
    EXPECT_EQ(1, allocationPool->numRanges());
    EXPECT_EQ(allocationPool->availableInRun(), 64 << 10);
>>>>>>> hp-pool-dev
    allocationPool->newRun(64 << 10);
    EXPECT_LE(128 << 10, pool_->currentBytes());
    allocationPool->allocateFixed(64 << 10);
    // Now at end of second 64K range, next will go to huge pages.
    setByte(allocationPool->allocateFixed(11));
    EXPECT_LE((2 << 20) - 11, allocationPool->availableInRun());
    // The first 2MB of the hugepage run are marked reserved.
    EXPECT_LE((2048 + 128) << 10, pool_->currentBytes());

    // The next allocation starts reserves the next 2MB of the mmapped range.
    setByte(allocationPool->allocateFixed(2 << 20));
    EXPECT_LE((4096 + 128) << 10, pool_->currentBytes());

    // Allocate the rest.
    allocationPool->allocateFixed(allocationPool->availableInRun());

    // We expect 3 ranges, 2 small and one large.
    EXPECT_EQ(3, allocationPool->numRanges());

    // We allocate more, expect a larger mmap.
    allocationPool->allocateFixed(1);

    // The first is at least 15 huge pages. The next is at least 31. The mmaps
    // may have unused addresses at either end, so count one huge page less than
    // the nominal size.
<<<<<<< HEAD
    EXPECT_LE(62 << 20, allocationPool->rangeAt(3).size());
=======
    EXPECT_LE((62 << 20) - 1, allocationPool->availableInRun());

    // We make a 5GB extra large allocation.
    allocationPool->allocateFixed(5UL << 30);
    EXPECT_EQ(5, allocationPool->numRanges());

    // 5G is an even multiple of huge page, no free space at end.
    EXPECT_EQ(0, allocationPool->availableInRun());

    EXPECT_LE(
        (5UL << 30) + (31 << 20) + (128 << 10),
        allocationPool->allocatedBytes());
    EXPECT_LE((5UL << 30) + (31 << 20) + (128 << 10), pool_->currentBytes());
>>>>>>> hp-pool-dev

    if (counter++ >= 1) {
      break;
    }

    // Repeat the above after a clear().
    allocationPool->clear();
    // Should be empty after clear().
    EXPECT_EQ(0, pool_->currentBytes());
  }
  allocationPool.reset();
  // Should be empty after destruction.
  EXPECT_EQ(0, pool_->currentBytes());
}
