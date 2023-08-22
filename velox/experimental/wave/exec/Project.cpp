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

#include "velox/experimental/wave/exec/Project.h"

namespace facebook::velox::wave {

void Project::schedule(WaveStream& stream, int32_t maxRows) {
  for (auto& level : levels_) {
    std::vector<std::unique_ptr<Executable>> exes(level.size());
    for
      auto i = 0;
    i < level.size(); ++i) {
      auto* program = level[i];
      exes[i] = program->executable();
    }
    auto blocksPerExe = bits::roundUp(maxRows, kBlockSize) / kBlockSize;
    stream.installExecutables(
        range, [&](Stream* out, folly::Range<Executable**> exes) {
          auto control = makeControl(stream, out, exes, blocksPerExe);
        });
  }

  void Project::finalize() {
    for (auto& level : levels_) {
      for (auto& program : level) {
        program->prepareForDevice(state.arena());
      }
    }
  }

} // namespace facebook::velox::wave
