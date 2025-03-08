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
      state, build.keys, build.dependent, build.joinType, build.id, true);
  auto id = build.id;
  out << "struct HashOps" << id << " {\n"
      << "  BuildOps" << id << "() = default;\n";

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

void JoinBuild::visitReferences(
    std::function<void(AbstractOperand*)> visitor) const {
  for (auto& k : keys) {
    visitor(k);
  }
  for (auto& d : dependent) {
    visitor(d);
  }
}

void JoinBuild::generateMain(CompileState& state, int32_t syncLabel) {
  makeBuildOps(state, *this);
  auto& out = state.generated();

  out << "  if (laneStatus == ErrorCode::kOk) {\n"
         "    BuildOps"
      << id
      << " ops;\n"
         "    auto* table  = reinterpret_cast>GpuHashTable*>(shared->states["
      << state.stateOrdinal(*this->state)
      << "]);\n"
         "    if (!table->addRow(";
  makeInitJoinRow(state, keys, dependent, id, false);
  out << ")) {\n"
         "     laneStatus = ErrorCode::kInsufficientMemory;\n"
         "      shared->hasContinue = true;\n"
         "    }\n";
}

std::string JoinBuild::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

std::unique_ptr<AbstractInstruction> JoinBuild::addInstruction(
    CompileState& state) {
  auto result =
      std::make_unique<AbstractHashBuild>(state.nextSerial(), this->state);
  result->continueLabel = continueLabel_;
  return result;
}

void JoinProbe::visitReferences(
    std::function<void(AbstractOperand*)> visitor) const {
  for (auto& key : keys) {
    visitor(key);
  }
}

void JoinProbe::visitResults(
    std::function<void(AbstractOperand*)> visitor) const {
  visitor(hits);
}

const char* probeBoilerPlate =
    "  table$I$ = reinterpret_cast<GpuHashTable*>(shared->states[$SI$]);\n"
    "  hit$I$ = table$I$->joinProbe(hash$I$, ";

void JoinProbe::generateMain(CompileState& state, int32_t syncLabel) {
  state.addInclude("velox/experimental/wave/common/Hash.h");
  state.addInclude("velox/experimental/wave/common/HashTable.cuh");

  makeJoinRow(state, keys, expand->dependent, joinType, id, true);

  auto& out = state.generated();
  state.declareNamed(fmt::format("bool nullProbe{};", id));
  state.declareNamed(fmt::format("uint64_t hash{};", id));

  auto stateOrd = state.stateOrdinal(*this->state);
  state.declareNamed(fmt::format("  GpuHashTable* table{};", id));
  state.declareNamed(fmt::format("  HashRow{}* hit{};", id, id));
  out << fmt::format("  nullProbe{} = false;\n", id);
  makeHash(state, keys, false, fmt::format("  nullProbe{} = true;", id), id);
  auto temp = replaceAll(probeBoilerPlate, "$I$", fmt::format("{}", id));
  out << replaceAll(temp, "$SI$", fmt::format("{}", stateOrd));
  makeCompareLambda(state, keys, false, id);
  out << ");\n";
  out << fmt::format("  continue{}: ;\n", expand->continueLabel_);
}

void JoinExpand::visitReferences(
    std::function<void(AbstractOperand*)> visitor) const {
  visitor(hits);
  if (filter) {
    visitor(filter);
  }
}

void JoinExpand::visitResults(
    std::function<void(AbstractOperand*)> visitor) const {
  visitor(indices);
  for (auto& r : dependent) {
    visitor(r);
  }
}

void makeCopyRow(CompileState& state, const JoinExpand& expand) {
  auto& out = state.generated();
  out << "[&](HashRow" << expand.id << "* hit, int32_t nth) {\n";
  for (auto i = 0; i < expand.dependent.size(); ++i) {
    auto tableOrd = expand.tableChannels[i];
    std::string field;
    int32_t nullFlag;
    auto* op = expand.dependent[i];
    if (tableOrd < expand.numKeys) {
      field = fmt::format("key{}", tableOrd);
      nullFlag = expand.nullableKeys ? tableOrd : -1;
    } else {
      field = fmt::format("dep{}", tableOrd - expand.numKeys);
      nullFlag =
          expand.nullableKeys ? tableOrd : tableOrd - expand.nullableKeys;
    }
    if (nullFlag != -1) {
      out << fmt::format(
          "   setNull(operands, {}, blockBase, (row->nulls{} & {}) == 0);\n",
          state.ordinal(*op),
          nullFlag / 32,
          (1 << (nullFlag & 31)));
    }
    out << fmt::format(
        "  flatOperand(operands, {}, blockBase) = row->{};\n",
        state.ordinal(*op),
        field);
  }
  out << "}";
}

void JoinExpand::generateMain(CompileState& state, int32_t syncLabel) {
  state.addInclude("velox/experimental/wave/exec/Join.cuh");
  auto& out = state.generated();
  if (filter) {
    state.generateIsTrue(*filter);
  }
  out << fmt::format(
      "  joinResult<hashRow{}, {}, {}, {}, {}>(",
      id,
      state.ordinal(*indices),
      status.gridState,
      status.gridStateSize,
      status.blockState);
  out << state.operandValue(hits) << ", "
      << (filter ? state.operandValue(filter) : "true")
      << ", shared->startLabel == " << continueLabel_ << ",  shared, true, ";
  makeCopyRow(state, *this);
  out << ");\n";
}

std::string JoinExpand::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

std::unique_ptr<AbstractInstruction> JoinExpand::addInstruction(
    CompileState& state) {
  auto result = std::make_unique<AbstractHashJoinExpand>(state.nextSerial());
  result->state = this->state;
  result->continueLabel = continueLabel_;
  return result;
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
} // namespace facebook::velox::wave
