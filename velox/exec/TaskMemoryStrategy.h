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

#include "velox/exec/Task.h"

namespace facebook::velox::exec {

class TaskMemoryStrategy : public memory::MemoryManagerStrategyBase {
 public:
  // The minimum Task shrink step.
  static constexpr int64_t kMinSpill = 24 << 20;

  // The minimum amount by which a Task must shrink for a transfer of capacity
  // to make sense. In practice, this is the minimum memory an operator must
  // hold in order to be spillable.
  static constexpr int64_t kGrowQuantum = 8 << 20;
  static constexpr int64_t kMinReclaimableBytes = 3 * kGrowQuantum;

  // When running with memory arbitration, the initial reservation comes from
  // the first action of the first Driver of the new Task. Therefore this is
  // rounded up to at least 24MB per Task.
  static constexpr int64_t kInitialSize = 24 << 20;

  explicit TaskMemoryStrategy(int64_t size) {
    auto tracker =
        memory::getProcessDefaultMemoryManager().getMemoryUsageTracker();
    tracker->updateConfig(
        memory::MemoryUsageConfigBuilder().maxTotalMemory(size).build());
  }

  bool canResize() const override {
    return true;
  }

  // The requester is a Task and the calling thread is a thread running a Driver
  // in the Task.
  bool reclaim(std::shared_ptr<memory::MemoryConsumer> requester, int64_t size)
      override;

 private:
  using ConsumerPtr = std::shared_ptr<memory::MemoryConsumer>;
  struct ConsumerScore {
    ConsumerPtr consumer;
    // 'candidate' if this is a Task.
    Task* FOLLY_NULLABLE task;
    int64_t available;
  };

  std::vector<ConsumerScore> listReclaimCandidates();
  // Serializes calls to reclaim().
  std::mutex mutex_;
};

} // namespace facebook::velox::exec
