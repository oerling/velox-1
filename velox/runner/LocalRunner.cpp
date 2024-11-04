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

#include "velox/runner/LocalRunner.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"

namespace facebook::velox::runner {
namespace {
std::shared_ptr<exec::RemoteConnectorSplit> remoteSplit(
    const std::string& taskId) {
  return std::make_shared<exec::RemoteConnectorSplit>(taskId);
}
} // namespace

RowVectorPtr LocalRunner::next() {
  if (!cursor_) {
    start();
  }
  bool isNext = cursor_->moveNext();
  if (!isNext) {
    state_ = RunnerState::kFinished;
    return nullptr;
  }
  return cursor_->current();
}

void LocalRunner::start() {
  VELOX_CHECK_EQ(state_, RunnerState::kInitialized);
  auto lastStage = makeStages();
  params_.planNode = plan_->fragments().back().fragment.planNode;
  auto cursor = exec::test::TaskCursor::create(params_);
  stages_.push_back({cursor->task()});
  // If the plan only has the last stage, there are no shuffles between the last
  // and previous stages to set up.
  if (!lastStage.empty()) {
    auto node = fragments_.back().inputStages[0].consumer;
    for (auto& remote : lastStage) {
      cursor->task()->addSplit(node, exec::Split(remote));
    }
    cursor->task()->noMoreSplits(node);
  }
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!error_) {
      cursor_ = std::move(cursor);
      state_ = RunnerState::kRunning;
    }
  }
  if (!cursor_) {
    // The cursor was not set because previous fragments had an error.
    abort();
    std::rethrow_exception(error_);
  }
}

void LocalRunner::abort() {
  // If called without previous error, we set the error to be cancellation.
  if (!error_) {
    try {
      state_ = RunnerState::kCancelled;
      VELOX_FAIL("Query cancelled");
    } catch (const std::exception& e) {
      error_ = std::current_exception();
    }
  }
  VELOX_CHECK(state_ != RunnerState::kInitialized);
  // Setting errors is thred safe. The stages do not change after
  // initialization.
  for (auto& stage : stages_) {
    for (auto& task : stage) {
      task->setError(error_);
    }
  }
  if (cursor_) {
    cursor_->setError(error_);
  }
}

void LocalRunner::waitForCompletion(int32_t maxWaitMicros) {
  VELOX_CHECK_NE(state_, RunnerState::kInitialized);
  std::vector<ContinueFuture> futures;
  {
    std::lock_guard<std::mutex> l(mutex_);
    for (auto& stage : stages_) {
      for (auto& task : stage) {
        futures.push_back(task->taskDeletionFuture());
      }
      stage.clear();
    }
  }
  for (auto& future : futures) {
    auto& executor = folly::QueuedImmediateExecutor::instance();

    std::move(future)
        .within(std::chrono::microseconds(maxWaitMicros))
        .via(&executor)
        .wait();
  }
}

std::vector<std::shared_ptr<exec::RemoteConnectorSplit>>
LocalRunner::makeStages() {
  std::unordered_map<std::string, int32_t> stageMap;
  auto sharedRunner = shared_from_this();
  auto onError = [self = sharedRunner, this](std::exception_ptr error) {
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (error_) {
        return;
      }
      state_ = RunnerState::kError;
      error_ = error;
    }
    if (cursor_) {
      abort();
    }
  };

  for (auto fragmentIndex = 0; fragmentIndex < fragments_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = fragments_[fragmentIndex];
    stageMap[fragment.taskPrefix] = stages_.size();
    stages_.emplace_back();
    for (auto i = 0; i < fragment.width; ++i) {
      exec::Consumer consumer = nullptr;
      auto task = exec::Task::create(
          fmt::format(
              "local://{}/{}.{}",
              params_.queryCtx->queryId(),
              fragment.taskPrefix,
              i),
          fragment.fragment,
          i,
          params_.queryCtx,
          exec::Task::ExecutionMode::kParallel,
          consumer,
          onError);
      stages_.back().push_back(task);
      if (fragment.numBroadcastDestinations) {
        task->updateOutputBuffers(fragment.numBroadcastDestinations, true);
      }
      task->start(options_.numDrivers);
    }
  }

  for (auto fragmentIndex = 0; fragmentIndex < fragments_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = fragments_[fragmentIndex];
    for (auto& scan : fragment.scans) {
      auto source = splitSourceFactory_->splitSourceForScan(*scan);
      bool allDone = false;
      do {
        for (auto i = 0; i < stages_[fragmentIndex].size(); ++i) {
          auto split = source->next(i);
          if (!split.hasConnectorSplit()) {
            allDone = true;
            break;
          }
          stages_[fragmentIndex][i]->addSplit(scan->id(), std::move(split));
        }
      } while (!allDone);
    }
    for (auto& scan : fragment.scans) {
      for (auto i = 0; i < stages_[fragmentIndex].size(); ++i) {
        stages_[fragmentIndex][i]->noMoreSplits(scan->id());
      }
    }

    for (auto& input : fragment.inputStages) {
      const auto sourceStage = stageMap[input.producerTaskPrefix];
      std::vector<std::shared_ptr<exec::RemoteConnectorSplit>> sourceSplits;
      for (auto i = 0; i < stages_[sourceStage].size(); ++i) {
        sourceSplits.push_back(remoteSplit(stages_[sourceStage][i]->taskId()));
      }
      for (auto& task : stages_[fragmentIndex]) {
        for (auto& remote : sourceSplits) {
          task->addSplit(input.consumer, exec::Split(remote));
        }
        task->noMoreSplits(input.consumer);
      }
    }
  }
  VELOX_CHECK(!stages_.empty());
  std::vector<std::shared_ptr<exec::RemoteConnectorSplit>> lastStage;
  for (auto& task : stages_.back()) {
    lastStage.push_back(remoteSplit(task->taskId()));
  }
  return lastStage;
}

exec::Split LocalSplitSource::next(int32_t /*worker*/) {
  if (currentFile_ >= static_cast<int32_t>(table_->files().size())) {
    return exec::Split();
  }
  if (currentSplit_ >= fileSplits_.size()) {
    fileSplits_.clear();
    ++currentFile_;
    if (currentFile_ >= table_->files().size()) {
      return exec::Split();
    }
    currentSplit_ = 0;
    auto filePath = table_->files()[currentFile_];
    const int fileSize = fs::file_size(filePath);
    // Take the upper bound.
    const int splitSize = std::ceil((fileSize) / splitsPerFile_);
    for (int i = 0; i < splitsPerFile_; i++) {
      fileSplits_.push_back(
          connector::hive::HiveConnectorSplitBuilder(filePath)
              .connectorId(table_->schema()->connector()->connectorId())
              .fileFormat(table_->format())
              .start(i * splitSize)
              .length(splitSize)
              .build());
    }
  }
  return exec::Split(std::move(fileSplits_[currentSplit_++]));
}

std::unique_ptr<SplitSource> LocalSplitSourceFactory::splitSourceForScan(
    const core::TableScanNode& tableScan) {
  auto tableHandle = dynamic_cast<const connector::hive::HiveTableHandle*>(
      tableScan.tableHandle().get());
  VELOX_CHECK(tableHandle);
  auto* table = reinterpret_cast<LocalTable*>(
      schema_->findTable(tableHandle->tableName()));
  return std::make_unique<LocalSplitSource>(table, splitsPerFile_);
}

std::vector<exec::TaskStats> LocalRunner::stats() const {
  std::vector<exec::TaskStats> result;
  std::lock_guard<std::mutex> l(mutex_);
  for (auto i = 0; i < stages_.size(); ++i) {
    auto& tasks = stages_[i];
    VELOX_CHECK(!tasks.empty());
    auto stats = tasks[0]->taskStats();
    for (auto j = 1; j < tasks.size(); ++j) {
      auto moreStats = tasks[j]->taskStats();
      for (auto pipeline = 0; pipeline < stats.pipelineStats.size();
           ++pipeline) {
        for (auto op = 0;
             op < stats.pipelineStats[pipeline].operatorStats.size();
             ++op) {
          stats.pipelineStats[pipeline].operatorStats[op].add(
              moreStats.pipelineStats[pipeline].operatorStats[op]);
        }
      }
    }
    result.push_back(std::move(stats));
  }
  return result;
}

std::string MultiFragmentPlan::toString() const {
  std::stringstream out;
  for (auto i = 0; i < fragments_.size(); ++i) {
    out << fmt::format(
        "Fragment {}: {} numWorkers={}:\n",
        i,
        fragments_[i].taskPrefix,
        fragments_[i].width);
    out << fragments_[i].fragment.planNode->toString(true, true) << std::endl;
    if (!fragments_[i].inputStages.empty()) {
      out << "Inputs: ";
      for (auto& input : fragments_[i].inputStages) {
        out << fmt::format(
            " {} <- {} ", input.consumer, input.producerTaskPrefix);
      }
      out << std::endl;
    }
  }
  return out.str();
}

std::string runnerStateString(RunnerState state) {
  switch (state) {
    case RunnerState::kInitialized:
      return "initialized";
    case RunnerState::kRunning:
      return "running";
    case RunnerState::kCancelled:
      return "cancelled";
    case RunnerState::kError:
      return "error";
    case RunnerState::kFinished:
      return "finished";
  }
  return "invalid state";
}

} // namespace facebook::velox::runner
