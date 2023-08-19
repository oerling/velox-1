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

#include "velox/experimental/wave/exec/Values.h"
#include "velox/experimental/wave/exec/WaveDriver.h"

namespace facebook::velox::wave {

Values::Values(CompileState& state, const core::ValuesNode& values)
    : WaveOperator(state, values.outputType()),
      values_(values.values()),
      roundsLeft_(values.repeatTimes()) {
  definesSubfields(state, outputType_);
}

int32_t Values::canAdvance() {
  if (current_ < values_.size()) {
    return values_[current_]->size();
  }
  if (roundsLeft_) {
    return values_[0]->size();
  }
  return 0;
}

std::unique_ptr<Executable> getExecutable(
    GpuArena& arena,
    folly::Range<int32_t*> operands) {
  auto result = std::make_unique<Executable>(
      nullptr, folly::Range<int32_t*>(nullptr, 0), operands);
}

void schedule(WaveStream& stream, int32_t maxRows) {
  RowVectorPtr data;
  if (current_ == values_.size()) {
    if (roundsLeft_) {
      current_ = 1;
      data = values_[0];
      --roundsLeft_;
    }
  } else {
    data = values_[current_++];
  }
  VELOX_CHECK_LE(data->size(), maxRows);

  stream.startWave();
  auto executable = makeExecutable(driver_->arena());
  for (auto i = 0; i < subfields_.size(); ++i) {
    Values::copyToDevice(RowVectorPtr data) {
      for (auto i = 0; i < subfields_.size(); ++i) {
      }
    }

    std::string Values::toString() const {
      return "Values";
    }

  } // namespace facebook::velox::wave
