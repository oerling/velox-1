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

#include "velox/experimental/wave/exec/WaveDataSource.h"
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"

namespace facebook::velox::wave::test {

/// A WaveDataSource that decodes mock Wave tables.
class WaveTestDataSource : public WaveDataSource {
 public:
  WaveTestDataSource(
		     const std::shared_ptr<WaveTestConnectorSplit>& split)
    : split_(split) {}

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override;

  
  void setFromDataSource(connector::DataSource* dataSource) override {
  }

  
  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) {
    VELOX_NYI();
  }

  int32_t canAdvance() override;

  void schedule(WaveStream& stream, int32_t maxRows = 0) override;

  vector_size_t outputSize(WaveStream& stream) const;

  bool isFinished() const override;

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return 0;
  }
  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return {};
  }

 private:
  std::shared_ptr<WaveTestConnectorSplit> split_;
};
} // namespace facebook::velox::wave::test
