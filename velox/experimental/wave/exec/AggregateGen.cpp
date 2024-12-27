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
  out << "struct HashRow {\n"
         "  int32_t flags;\n"
      << std::endl;

  makeKeyMembers(probe.keys, out);
  int32_t numNullable = probe.keys.size() + probe.updates.size();
  for (auto n = 0; n < numNullable; ++n) {
    out << fmt::format("  uint32_t nulls{} = ~0;\n", n / 32);
  }
  for (auto i = 0; i < probe.updates.size(); ++i) {
    out << probe.updates[i]->generator->generateAccumulator(
               state, probe, *probe.updates[i])
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

/// Emits a lambda that performs the inlined aggregate update.

void makeUpdateLambda(
    CompileState& state,
    const AggregateProbe& probe,
    std::vector<const KernelStep*> updates) {
  auto& out = state.generated();

  out << "  [&](GpuHashTableBase* table, HashRow* row, uint32_t peers, int32_t leader, int32_t laneId) {\n";
  std::vector<const AggregateUpdate*> deferred;

  auto emitUpdates = [&]() {
    if (deferred.size() > 4) {
      for (auto& update : deferred) {
        update->generator->makeDeduppedUpdate(state, probe, *update);
      }
    }
    deferred.clear();
  };
  for (auto lastIdx = 0; lastIdx < updates.size(); ++lastIdx) {
    auto* step = updates[lastIdx];
    if (step->kind() != StepKind::kAggregateUpdate) {
      const_cast<KernelStep*>(step)->generateMain(state);
      continue;
    }
    auto& update = step->as<AggregateUpdate>();
    update.generator->loadArgs(state, probe, update);
    deferred.push_back(&update);
    emitUpdates();
  }
  emitUpdates();

  out << "  }";
}

void makeAggregateProbe(CompileState& state, const AggregateProbe& probe) {
  auto& out = state.generated();
  makeHash(state, probe.keys, true, "");
  out << "  AggregateOps ops(hash, shared);\n"
      << fmt::format(
             "  auto state =\n"
             "    reinterpret_cast<DeviceAggregation*>(shared->states[{}]);\n",
             state.stateOrdinal(*probe.state));
  out << "  state->table->updatingProbe<HashRow>(threadIdx.x, laneId(), laneStatus == ErrorCode::kOk, \n";
  makeCompareLambda(state, probe.keys, true);
  out << ",\n";
  makeInitKey(state, probe.keys, {}, probe.updates, true);
  out << ",\n";
  makeUpdateLambda(state, probe, probe.inlinedUpdates);
  out << ");\n";
  out << "      __syncthreads();\n"
         "  laneStatus = shared->status->errors[threadIdx.x];\n"
         "  if (threadIdx.x == 0 && shared->hasContinue) {\n"
         "    auto ret = gridStatus<AggregateReturn>(shared, agg->status);\n"
         "    ret->numDistinct = table->numDistinct;\n"
         "  }\n"
         "  __syncthreads();\n"
         "  if (threadIdx.x == 0 && shared->isContinue) {\n"
         "    shared->isContinue = false;\n"
         "  }\n"
         "  __syncthreads();\n";
}

std::string readAggRow(CompileState& state, const ReadAggregation& read) {
  std::stringstream out;
  for (auto i = 0; i < read.funcs.size(); ++i) {
    auto& func = *read.funcs[i];
    out << func.generator->generateExtract(state, *read.probe, func);
  }
  return out.str();
}
void makeReadAggregation(CompileState& state, const ReadAggregation& read) {
  auto& out = state.generated();

  out << "  AggregateOps ops(hash, shared);\n";
  out << "  if (threadIdx.x != 0) { lanestatus = ErrorCode::kInactive; } else {\n"
      << fmt::format(
             "  auto state =\n"
             "    reinterpret_cast<DeviceAggregation*>(shared->states[{}]);\n",
             state.stateOrdinal(*read.state));
  if (read.probe->keys.empty()) {
    out << "  HashRow* row = reinterpret_cast<HashRow*>(state->singleRow);\n";
    out << readAggRow(state, read);
    out << "    shared->status->numRows = 1;\n";
    out << "  }\n";
    return;
  }
  out << "  auto rowIdx = blockIdx.x * kBlockSize + threadIdx.x + 1;\n"
         "  auto numRows = state->resultRowPointers[shared->streamIdx][0];\n"
         "  if (rowIdx <= numRows) {\n"
         "    auto* row = reinterpret_cast<HashRow*>(\n"
         "      state->resultRowPointers[shared->streamIdx][rowIdx]);\n";
  // Copy keys and accumulators to output.
  for (auto i = 0; i < read.probe->keys.size(); ++i) {
    out << extractColumn(
        "row",
        fmt::format("key{}", i),
        state.ordinal(*read.keys[i]),
        *read.keys[i]);
  }
  out << readAggRow(state, read);
  out << "  if (threadIdx.x == 0) {\n"
      << "    shared->numRows = rowIdx + kBlockSize <= numRows \n"
      << "   ? kBlockSize \n"
      << "    : numRows - blockIdx.x * kBlockSize;\n"
      << "    }\n";
}

} // namespace facebook::velox::wave
