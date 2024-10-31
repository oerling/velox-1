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

#include "velox/connectors/Connector.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/runner/LocalSchema.h"
#include "velox/runner/MultiFragmentPlan.h"

/// Base classes for multifragment Velox query execution.
namespace facebook::velox::runner {

/// Iterator for obtaining splits for a scan. One is created for each table
/// scan.
class SplitSource {
 public:
  virtual ~SplitSource() = default;
  /// Returns a split for 'worker'. This may implement soft affinity or strict
  /// bucket to worker mapping.
  virtual exec::Split next(int32_t worker) = 0;
};

/// A factory for getting a SplitSource for each TableScan. The splits produced
/// may depend on partition keys, buckets etc mentioned by each tableScan.
class SplitSourceFactory {
 public:
  virtual ~SplitSourceFactory() = default;

  /// Returns a splitSource for one TableScan across all Tasks of
  /// the fragment. The source will be invoked to produce splits for
  /// each individual worker running the scan.
  virtual std::unique_ptr<SplitSource> splitSourceForScan(
      const core::TableScanNode& scan) = 0;
};

enum class RunnerState { kRunning, kFinished, kError, kCancelled };

/// Base class for executing multifragment Velox queries. One instance
/// of a Runner coordinates the execution of one multifragment
/// query. Different derived classes can support different shuffles
/// and different scheduling either in process or in a cluster. Unless
/// otherwise stated, the member functions are thread safe as long as
/// the caller holds an owning reference to the runner.
class Runner {
 public:
  virtual ~Runner() = default;

  /// Starts execution and returns a TaskCursor for consuming the results.
  /// moveNext on the cursor throws the first error encountered in the
  /// execution. This may be called only once.
  virtual exec::test::TaskCursor* start() = 0;

  /// Returns Task stats for each fragment of the plan. The stats correspond 1:1
  /// to the stages in the MultiFragmentPlan. This may be called at any time.
  /// before waitForCompletion() or abort().
  virtual std::vector<exec::TaskStats> stats() const = 0;

  /// Returns the state of execution.
  virtual RunnerState state() const = 0;

  /// Cancels the possibly pending execution. Returns before the execution is
  /// finished. Use waitForCompletion() to wait for all execution resources to
  /// be freed.
  virtual void abort() = 0;

  /// Waits up to 'maxWaitMicros' for all activity of the execution to cease.
  /// Call this in a test to make sure execution threads have stopped and memory
  /// is freed before destroying the test fixture and its memory pools.
  virtual void waitForCompletion(int32_t maxWaitMicros) = 0;
};

} // namespace facebook::velox::runner
