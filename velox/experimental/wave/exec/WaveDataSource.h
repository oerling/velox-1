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

#include "velox/common/time/Timer.h"
#include "velox/connectors/Connector.h"
#include "velox/exec/Task.h"
#include "velox/experimental/wave/exec/WaveOperator.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::wave {

/// A delegate produced by a regular Velox connector::DataSource for reading its
/// particular file format on GPU.
class WaveDataSource {
 public:
  virtual void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) = 0;

  virtual void addSplit(std::shared_ptr<connector::ConnectorSplit> split) = 0;

  virtual int32_t canAdvance() = 0;

  virtual void schedule(WaveStream& stream, int32_t maxRows = 0) = 0;
 
  virtual vector_size_t outputSize(WaveStream& stream) const = 0;

  virtual bool isFinished() const = 0;

  virtual uint64_t getCompletedBytes() = 0;

  virtual uint64_t getCompletedRows() = 0;

  virtual std::unordered_map<std::string, RuntimeCounter> runtimeStats() = 0;
  void setSplitFromShell(std::unique_ptr<connector::DataSource> source) {
    VELOX_UNSUPPORTED();
  }
  
  /// Initializes a SplitReader for 'split' and encapsulates it inside the returned DataSource. 
  std::unique_ptr<connector::DataSource> createShellForSplit(std::shared_ptr<connector::ConnectorSplit> split) {
    VELOX_UNSUPPORTED();
  }
};
} // namespace facebook::velox::wave
