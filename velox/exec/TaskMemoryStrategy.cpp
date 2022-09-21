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

#include "velox/exec/TaskMemoryStrategy.h"

namespace facebook::velox::exec {
namespace {
bool maybeUpdate(int64_t size, memory::MemoryUsageTracker& tracker) {
  try {
    tracker.update(size);
    return true;
  } catch (const std::exception& e) {
  }
  return false;
}

inline int64_t roundDown(int64_t n, int64_t quantum) {
  return (n / quantum) * quantum;
}

void setLimit(int64_t bytes, memory::MemoryUsageTracker& tracker) {
  tracker.updateConfig(
      memory::MemoryUsageConfigBuilder().maxTotalMemory(bytes).build());
}

bool pauseIfRunning(
    std::shared_ptr<Task>& task,
    std::vector<std::shared_ptr<Task>>& pausedTasks) {
  if (std::find(pausedTasks.begin(), pausedTasks.end(), task) !=
      pausedTasks.end()) {
    return true;
  }
  auto future = task->requestPause(true);
  auto& exec = folly::QueuedImmediateExecutor::instance();
  std::move(future).via(&exec).wait();
  if (task->state() != TaskState::kRunning) {
    return false;
  }
  VELOX_CHECK_EQ(0, task->numThreads());
  pausedTasks.push_back(task);
  return true;
}
} // namespace

bool TaskMemoryStrategy::reclaim(
    std::shared_ptr<memory::MemoryConsumer> requester,
    int64_t bytes) {
  // Returns true if 'bytes' bytes can be transferred to 'requester' from
  // unused space in topTracker and/or reclaimed space from other consumers.

  std::lock_guard<std::mutex> l(mutex_);
  Task* consumerTask = dynamic_cast<Task*>(requester.get());
  VELOX_CHECK(consumerTask, "Only a Task can request memory via reclaim()");
  auto topTracker =
      memory::getProcessDefaultMemoryManager().getMemoryUsageTracker();
  auto& tracker = consumerTask->tracker();
  // The limits of the consumers are stable inside this section but
  // the allocation sizes are volatile until the consumer in question
  // is stopped.

  if (consumerTask->state() != kRunning) {
    return false;
  }
  int64_t bytesForRequester =
      tracker.maxTotalBytes() < kMinReclaimableBytes ? kInitialSize : bytes;
  auto available =
      topTracker->maxTotalBytes() - topTracker->getCurrentUserBytes();

  // Allocation can be made from unallocated memory.
  if (bytesForRequester <= available) {
    if (maybeUpdate(bytesForRequester, *topTracker)) {
      setLimit(tracker.maxTotalBytes() + bytesForRequester, tracker);
      VLOG(1) << fmt::format(
          "{} gets {} from unused", consumerTask->taskId(), bytesForRequester);
      return true;
    }
  }
  std::vector<std::shared_ptr<Task>> pausedTasks;
  auto continueGuard = folly::makeGuard([&]() {
    for (auto& task : pausedTasks) {
      if (task->state() == TaskState::kRunning) {
        Task::resume(task);
      }
    }
  });

  // Get reclaim candidates, stop and spill enough of them to meet the demand.
  bool success = false;
  constexpr int32_t kMaxReclaimRounds = 2;
  int32_t numReclaimRounds = 0;
  int64_t reclaimedBytes = 0;
  for (;;) {
    auto candidates = listReclaimCandidates();
    for (auto& candidate : candidates) {
      if (auto task = std::dynamic_pointer_cast<Task>(candidate.consumer)) {
        if (!pauseIfRunning(task, pausedTasks)) {
          continue;
        }
        auto& taskTracker = task->tracker();
        auto previousBytes = taskTracker.maxTotalBytes();
        auto potentialBytes = task->reclaimableBytes();
        auto tryBytes =
            std::max(kMinReclaimableBytes, bytesForRequester - reclaimedBytes);
        task->reclaim(tryBytes);
        auto reclaimedFromTask =
            previousBytes - taskTracker.getCurrentTotalBytes();
        if (reclaimedFromTask < kGrowQuantum) {
          if (potentialBytes > tryBytes) {
            LOG(INFO) << "Task did not shrink as promised";
          }
          // Too little reclaimed. No change to limit.
          continue;
        }
        VLOG(1) << fmt::format(
            "{} drops limit by {} to {}",
            task->taskId(),
            reclaimedFromTask,
            taskTracker.maxTotalBytes() - reclaimedFromTask);
        setLimit(taskTracker.maxTotalBytes() - reclaimedFromTask, taskTracker);

        reclaimedBytes += reclaimedFromTask;
        if (reclaimedBytes >= bytesForRequester) {
          break;
        }
      }
    }
    success = reclaimedBytes >= bytesForRequester;
    if (success) {
      break;
    }
    if (++numReclaimRounds >= kMaxReclaimRounds) {
      break;
    }
  }
  // Updating downward, no throw.
  if (success) {
    topTracker->update(bytesForRequester - reclaimedBytes);
    VLOG(1) << fmt::format(
        "{} increases limit by {} to {}",
        consumerTask->taskId(),
        bytesForRequester,
        tracker.maxTotalBytes() + bytesForRequester);
    setLimit(tracker.maxTotalBytes() + bytesForRequester, tracker);
  } else {
    topTracker->update(-reclaimedBytes);
  }
  return success;
}

std::vector<TaskMemoryStrategy::ConsumerScore>
TaskMemoryStrategy::listReclaimCandidates() {
  std::vector<ConsumerScore> candidates;
  for (auto [raw, weak] : consumers_) {
    auto ptr = weak.lock();
    if (ptr) {
      auto otherTask = dynamic_cast<Task*>(ptr.get());
      if (otherTask) {
        auto size = otherTask->tracker().getCurrentTotalBytes();
        if (size > kInitialSize) {
          auto reclaimable = otherTask->reclaimableBytes();
          if (reclaimable >= kMinReclaimableBytes) {
            candidates.push_back({ptr, otherTask, reclaimable});
          }
        }
      }
    }
  }

  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](const ConsumerScore& left, const ConsumerScore& right) {
        // Most available first.
        return left.available > right.available;
      });
  return candidates;
}

} // namespace facebook::velox::exec
