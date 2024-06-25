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

#include "velox/experimental/wave/exec/WaveOperator.h"

namespace facebook::velox::wave {

class Project : public WaveOperator {
 public:
  Project(
      CompileState& state,
      RowTypePtr outputType,
      std::vector<std::vector<ProgramPtr>> levels,
      AbstractWrap* filterWrap = nullptr)
      : WaveOperator(state, outputType, ""),
        levels_(std::move(levels)),
        filterWrap_(filterWrap) {}

  AbstractWrap* findWrap() const override;

  bool isStreaming() const override {
    if (levels_[0].size() == 1 && levels_[0][0]->isSource()) {
      return false;
    }
    return true;
  }

  bool isSource() const override {
    return !isStreaming();
  }

  int32_t canAdvance(WaveStream& Stream) override;

  void schedule(WaveStream& stream, int32_t maxRows = 0) override;

  vector_size_t outputSize(WaveStream& stream) const override;

  void finalize(CompileState& state) override;

  std::string toString() const override {
    return fmt::format("Project {}", WaveOperator::toString());
  }

  const OperandSet& syncSet() const override {
    return computedSet_;
  }

 private:
  struct ContinueLocation {
    int32_t programIdx;
    int32_t instructionIdx;
  };

  std::vector<std::vector<ProgramPtr>> levels_;
  OperandSet computedSet_;
  AbstractWrap* filterWrap_{nullptr};

  // Index in 'levels_' where the next schedule() starts.
  int32_t continueLevel_{0};

  // If non-empty, represents the programs and lane masks that need to be continued.
  std::vector<ContinuePoint> continuePoints_;
};

} // namespace facebook::velox::wave
