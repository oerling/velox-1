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

#include "velox/experimental/query/LocalRunner.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

namespace facebook::velox::exec {

test::TaskCursor* LocalRunner::cursor() {
  auto lastStage = makeStages();
  params_.planNode = plan_.back().fragment.planNode;
  cursor_ = std::make_unique<test::TaskCursor>(params_);
  if (!lastStage.empty()) {
    auto node = plan_.back().inputStages[0].consumer; // NOLINT
    for (auto& remote : lastStage) {
      cursor_->task()->addSplit(node, Split(remote));
    }
    cursor_->task()->noMoreSplits(node);
  }
  return cursor_.get();
}

std::vector<std::shared_ptr<RemoteConnectorSplit>> LocalRunner::makeStages() {
  std::unordered_map<std::string, int32_t> prefixMap;
  for (auto fragmentIndex = 0; fragmentIndex < plan_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = plan_[fragmentIndex];
    prefixMap[fragment.taskPrefix] = stages_.size();
    stages_.emplace_back();
    for (auto i = 0; i < fragment.width; ++i) {
      auto task = std::make_shared<Task>(
          fmt::format("{}.{}", fragment.taskPrefix, i),
          fragment.fragment,
          i,
          params_.queryCtx);
      stages_.back().push_back(task);
      Task::start(task, options_.numDrivers);
    }
  }

  for (auto fragmentIndex = 0; fragmentIndex < plan_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = plan_[fragmentIndex];
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
      auto sourceStage = prefixMap[input.producerTaskPrefix];
      std::vector<std::shared_ptr<RemoteConnectorSplit>> sourceSplits;
      for (auto i = 0; i < stages_[sourceStage].size(); ++i) {
        sourceSplits.push_back(std::make_shared<RemoteConnectorSplit>(
            stages_[sourceStage][i]->taskId()));
      }
      for (auto& task : stages_[fragmentIndex]) {
        for (auto& remote : sourceSplits) {
          task->addSplit(input.consumer, Split(remote));
        }
        task->noMoreSplits(input.consumer);
      }
    }
  }
  if (stages_.empty()) {
    return {};
  }
  std::vector<std::shared_ptr<RemoteConnectorSplit>> lastStage;
  for (auto& task : stages_.back()) {
    lastStage.push_back(std::make_shared<RemoteConnectorSplit>(task->taskId()));
  }
  return lastStage;
}

Split LocalSplitSource::next(int32_t /*worker*/) {
  if (currentFile_ >= table_->files.size()) {
    return Split();
  }
  if (currentSplit_ >= fileSplits_.size()) {
    fileSplits_.clear();
    ++currentFile_;
    if (currentFile_ >= table_->files.size()) {
      return Split();
    }
    currentSplit_ = 0;
    auto filePath = table_->files[currentFile_];
    const int fileSize = fs::file_size(filePath);
    // Take the upper bound.
    const int splitSize = std::ceil((fileSize) / splitsPerFile_);
    for (int i = 0; i < splitsPerFile_; i++) {
      fileSplits_.push_back(test::HiveConnectorSplitBuilder(filePath)
                                .fileFormat(table_->format)
                                .start(i * splitSize)
                                .length(splitSize)
                                .build());
    }
  }
  return Split(std::move(fileSplits_[currentSplit_++]));
}

std::unique_ptr<SplitSource> LocalSplitSourceFactory::splitSourceForScan(
    const core::TableScanNode& tableScan) {
  auto tableHandle = dynamic_cast<const connector::hive::HiveTableHandle*>(
      tableScan.tableHandle().get());
  VELOX_CHECK(tableHandle);
  auto it = schema_.tables().find(tableHandle->tableName());
  VELOX_CHECK(it != schema_.tables().end());
  auto table = it->second.get();
  return std::make_unique<LocalSplitSource>(table, splitsPerFile_);
}

} // namespace facebook::velox::exec
