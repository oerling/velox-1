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

#include "velox/runner/Runner.h"
#include "velox/connectors/Connector.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/runner/MultiFragmentPlan.h"
#include "velox/runner/LocalSchema.h"

namespace facebook::velox::runner {

  
  /// 
  class LocalRunner : public Runner, public std::enable_shared_from_this<LocalRunner> {
 public:
  LocalRunner(
	      MultiFragmentPlanPtr plan,
      std::shared_ptr<core::QueryCtx> queryCtx,
      std::shared_ptr<SplitSourceFactory> splitSourceFactory)
      : plan_(std::move(plan)),
	fragments_(plan_->fragments()),
	options_(plan_->options()),
        splitSourceFactory_(std::move(splitSourceFactory)) {
    params_.queryCtx = std::move(queryCtx);
  }

    exec::test::TaskCursor* start() override;

    std::vector<exec::TaskStats> stats() const override;

    void abort() override;
    
    void waitForCompletion( int32_t maxWaitMicros) override;

 private:
    
    std::vector<std::shared_ptr<exec::RemoteConnectorSplit>> makeStages();
  // Serializes 'cursor_' and 'error_'.
    mutable std::mutex mutex_;

    exec::test::CursorParameters params_;
    MultiFragmentPlanPtr const plan_;
    std::vector<ExecutableFragment> fragments_;
    const MultiFragmentPlan::Options& options_;

    std::shared_ptr<SplitSourceFactory> splitSourceFactory_;
    std::unique_ptr<exec::test::TaskCursor> cursor_;
    std::vector<std::vector<std::shared_ptr<exec::Task>>> stages_;
  std::exception_ptr error_;
  bool tasksCreated_{false};
};

  /// 
class LocalSplitSource : public SplitSource {
 public:
  LocalSplitSource(const LocalTable* table, int32_t splitsPerFile)
      : table_(table), splitsPerFile_(splitsPerFile) {}

  exec::Split next(int32_t worker) override;

 private:


  const LocalTable* const table_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> fileSplits_;
  const int32_t splitsPerFile_;
  int32_t currentFile_{-1};
  int32_t currentSplit_{0};
};

class LocalSplitSourceFactory : public SplitSourceFactory {
 public:
  LocalSplitSourceFactory(
      std::shared_ptr<LocalSchema> schema,
      int32_t splitsPerFile)
      : schema_(schema), splitsPerFile_(splitsPerFile) {}

  std::unique_ptr<SplitSource> splitSourceForScan(
      const core::TableScanNode& scan) override;

 private:
  std::shared_ptr<LocalSchema> schema_;
  const int32_t splitsPerFile_;
};

} // namespace facebook::velox::exec
