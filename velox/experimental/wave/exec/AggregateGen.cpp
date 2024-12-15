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

#include "velox/experimental/wave/exec/AggregateGen.h"

namespace facebook::velox::wave {

  std::string makeAggregateRow(CompileState& state, const AggregateProbe& probe) {
  std::stringstream out;
  out << "struct AggregateRow {\n"
         "  int32_t flags;\n"
      << std::endl;

  makeKeyMembers(probe.keys, out);
  int32_t numNullable = probe.keys.size() + probe.updates.size();
  auto numFlagWords = bits::roundUp(numNullable, 32) / 32;
  out << fmt::format("  nullFlags[{}];\n", numFlagWords);
  for (auto i = 0; i < probe.updates.size(); ++i) {

    out << probe.updates[i]->generator->generateAccumulator(state, probe, *probe.updates[i])
        << std::endl
	<< " acc" << i << ";\n";
  }
  out << "};\n\n";
  return out.str();
}

void makeAggregateOps(
    CompileState& state,
    const AggregateProbe& probe,
    bool forRead) {
  auto& out = state.inlines();
  out << makeAggregateRow(state, probe);

  out << "struct AggregateOps {\n"
      << "  AggreegateOps(uintt64_t hash, WaveShared* shared) : hash(hash), shared(shared){}\n"
      << "  uint64_t hash = 1;\n"
      << "  WaveShared* shared;\n";
  if (forRead) {

    } else {
      out << "  uint64_t hash() const { return hash; }\n";
      makeRowHash(state, probe.keys, true);
  }
      out << "};\n\n";
}

 /// Emits a lambda that performs the inlined aggregate update. This is run on warp-level dedupped rows. The signature is [&](this, bucket, writable, peers, idxToUpdate).
 void makeUpdate(CompileState& state, AggregateUpdate* update);

  
void makeAggregateProbe(CompileState& state, const AggregateProbe& probe) {
  auto& out = state.generated();
  makeHash(state, probe.keys, true, "");
    }
  
} // namespace facebook::velox::wave
