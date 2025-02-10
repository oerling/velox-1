

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

#include "velox/experimental/wave/exec/HashGen.h"

namespace facebook::velox::wave {



  std::string makeJoinRow(CompileState& state, const std::string& planNodeId, const std::vector<AbstractOperand*>& keys, std::vector<AbstractOperand*>& dependent, core::JoinType joinType) {
    bool nullableKeys = joinType == core::JoinType::kRight || joinType == core::JoinType::kFull;
    std::stringstream out;
  out << "struct HashRow" << planNodeId
      << " {\n";
  int32_t numNullableKeys = nullableKeys ? keys.size() : 0;
  int32_t numNullable = numNullableKeys + dependent.size();
  for (auto n = 0; n < numNullable; n += 32) {
    out << fmt::format("  uint32_t nulls{};\n", n / 32);
  }
  makeKeyMembers(keys, out);
  makeKeyMembers(dependent, out);

  out << "};\n\n";
  return out.str();
}


void AggregateProbe::generateMain(CompileState& state, int32_t syncLabel) {
  makeAggregateOps(state, *this, false);
  makeAggregateProbe(state, *this, syncLabel);
}

std::string AggregateProbe::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

std::unique_ptr<AbstractInstruction> AggregateProbe::addInstruction(
    CompileState& state) {
  RowTypePtr type;
  static std::vector<AbstractAggInstruction> empty;
  auto agg = std::make_unique<AbstractAggregation>(
      state.nextSerial(), keys, empty, this->state, type);
  int32_t offset =
      sizeof(int32_t) + bits::roundUp(keys.size() + updates.size(), 32) / 8;
  for (auto& key : keys) {
    int32_t align = cudaTypeAlign(*key->type);
    int32_t width = cudaTypeSize(*key->type);
    offset = bits::roundUp(offset, align) + width;
  }
  for (auto& update : updates) {
    auto [size, align] = update->generator->accumulatorSizeAndAlign(*update);
    offset = bits::roundUp(offset, align) + size;
  }
  agg->roundedRowSize = bits::roundUp(offset, 8);
  abstractAggregation = agg.get();
  agg->continueLabel = continueLabelN;
  return agg;
}

std::string AggregateProbe::toString() const {
  std::stringstream out;
  out << "aggregateProbe {";
  for (auto& key : keys) {
    out << key->toString() << " ";
  }
  out << "}\n";
  if (rows) {
    out << "  row=" << rows->toString() << "\n";
  }
  if (!inlinedUpdates.empty()) {
    out << "  inlined {\n";
    for (auto& update : inlinedUpdates) {
      out << update->toString();
    }
    out << "\n}\n";
  }

  return out.str();
}




}
