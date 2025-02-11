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

std::string makeJoinRow(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    const std::vector<AbstractOperand*>& dependent,
    core::JoinType joinType,
    int32_t id,
    bool hasNext) {
  bool nullableKeys =
      joinType == core::JoinType::kRight || joinType == core::JoinType::kFull;
  std::stringstream out;
  out << "struct HashRow" << id << " {\n";
  int32_t numNullableKeys = nullableKeys ? keys.size() : 0;
  int32_t numNullable = numNullableKeys + dependent.size();
  for (auto n = 0; n < numNullable; n += 32) {
    out << fmt::format("  uint32_t nulls{};\n", n / 32);
  }
  makeKeyMembers(keys, out);
  makeKeyMembers(dependent, out);
  if (hasNext) {
    out << "HashRow" << id
        << "* next;\n"
           "  HashRow"
        << id << "** nextPtr() { return &next; }\n";
  } else {
    out << "  HashRow** nextPtr() { return nullptr;}\n";
  }
  out << ";\n\n}";
  return out.str();
}

void makeInitJoinRow(
    CompileState& state,
    const OpVector& keys,
    const OpVector& dependent,
    int32_t id,
    bool nullableKeys) {
  auto& out = state.generated();
  out << "  [&](HashRow" << id << "* row) {\n";
  int32_t numNullFlags = dependent.size() + (nullableKeys ? keys.size() : 0);
  for (auto i = 0; i < keys.size(); ++i) {
    auto* op = keys[i];
    if (nullableKeys) {
      out << fmt::format(
          "   if (!{}) {{ row->key{} = {};}}\n",
          state.isNull(op),
          i,
          state.operandValue(op));
    } else {
      out << "    row->key" << i << " = " << state.operandValue(op);
    }
  }

  for (auto i = 0; i < dependent.size(); ++i) {
    auto op = dependent[i];
    out << fmt::format(
        "   if (!{}) {{ row->key{} = {};}}\n",
        state.isNull(op),
        i,
        state.operandValue(op));
  }
  for (auto i = 32; i < numNullFlags; i += 32) {
    out << fmt::format("   row->nulls{} = 0;\n", i / 32);
  }
  OpVector allColumns;
  if (nullableKeys) {
    allColumns = keys;
  }
  allColumns.insert(allColumns.end(), dependent.begin(), dependent.end());
  out << fmt::format(
      "  row->nulls0 = {};\n", initRowNullFlags(state, 0, keys.size(), keys));
  out << "}\n";
}

void makeRowRowCompare(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    int32_t id) {
  auto& out = state.generated();
  out << "  bool __device__ compare(HashRow" << id << "* left, HashRow" << id
      << "* right) {\n";
  for (auto i = 0; i < keys.size(); ++i) {
    out << "    if (left.key" << i << " != right.key" << i
        << ") { return false; }\n";
  }
  out << "  return true;\n}\n";
}

void makeBuildOps(CompileState& state, const JoinBuild& build) {
  state.addInclude("velox/experimental/wave/common/Hash.h");
  state.addInclude("velox/experimental/wave/common/BitUtil.cuh");
  state.addInclude("velox/experimental/wave/common/HashTable.cuh");
  auto& out = state.inlines();
  out << makeJoinRow(
      state, build.keys, build.dependent, build.joinType, build.id);
  auto id = build.id;
  out << "struct HashOps" << id << " {\n"
      << "  BuildOps" << id
      << "() = default;\n"

      makeRowHash(state, build.keys, false, id);
  makeRowRowCompare(state, build.keys, id);

  out << "};\n\n";

  state.addEntryPoint("facebook::velox::wave::buildTable");
  out << "void __global__ buildTableKernel(GpuHashTable* table, HashRow" << id
      << "** rows, int32_t numRows) {\n"
         "  hashOps"
      << id
      << " ops();\n"
         "  table->buildTable<HashRow"
      << id << ", HashOps" << id
      << ">(rows, numRows);\n"
         "}\n";
}

std::string JoinBuild::toString() const {
  std::stringstream out;
  out << "JoinBuild {";
  for (auto& key : keys) {
    out << key->toString() << " ";
  }
  out << " -> ";
  for (auto& dep : dependent) {
    out << dep->toString() << " ";
  }
  out << std::endl;
  return out.str();
}
return out.str();
} // namespace facebook::velox::wave

void JoinBuild::generateMain(CompileState& state, int32_t syncLabel) {
  makeJoinOps(state, *this, false);
  auto& out = state.generated();

  out << "  if (laneStatus == ErrorCode::kOk) {\n"
         "    BuildOps"
      << id
      << " ops;\n"
         "    auto* table  = reinterpret_cast>GpuHashTable*>(shared->state["
      << stateId
      << "]);\n"
         "    if (!table->addRow("
      << makeBuildInit(state, build)
      << ")) {\n"
         "     laneStatus = ErrorCode::kInsufficientMemory;\n"
         "      shared->hasContinue = true;\n"
         "    }\n"
}

std::string JoinBuild::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

void JoinProbe::generateMain(CompileState& state, int32_t syncLabel) {
  makeJoinOps(state, *this, false);
  makeJoinProbe(state, *this, syncLabel);
}

std::string JoinProbe::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

std::unique_ptr<AbstractInstruction> JoinProbe::addInstruction(
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

std::string JoinProbe::toString() const {
  std::stringstream out;
  out << "JoinProbe {";
  for (auto& key : keys) {
    out << key->toString() << " ";
  }
  out << "}\n";
  if (hits) {
    out << "  row=" << hits->toString() << "\n";
  }

  return out.str();
}

std::string JoinExpand::toString() const {
  std::stringstream out;
  out << "JoinExpand {";
  if (filter) {
    out << " filter = " << filter->toString() << std::endl;
  }
  out << " result = {{";
  for (auto& dep : dependent) {
    out << dep->toString() << " ";
  }
  out << "}\n";

  return out.str();
}
}
