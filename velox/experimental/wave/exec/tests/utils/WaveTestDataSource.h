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
namespace facebook::velox::wave::test {

/// A WaveDataSource that decodes mock Wave tables.
class WaveTestDataSource : public WaveDataSource {
 public:
 WaveTestDataSource(const std::shared_ptr<WaveTestConnectorSplit>& split)
    : split_(split);

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) {
    VELOX_NYI();
  }

  
  int32_t canAdvance() overide;

  void schedule(WaveStream& stream, int32_t maxRows = 0) override;

  bool isFinished() const override;

  
 private:
  std::shared_pptr<WaveTestConnectorSplit> split_;
};
}

