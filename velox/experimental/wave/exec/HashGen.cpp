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

void makeKeyMembers(
    const std::vector<AbstractOperand*>& keys,
    std::stringstream& out) {
  for (auto i = 0; i < keys.size(); ++i) {
    auto* key = keys[i];
    out << cudaTypeName(*key->type) << " key" << i << ";\n";
  }
}

void makeProbeKeyMembers(
    const std::vector<AbstractOperand*>& keys,
    std::stringstream& out) {
  for (auto i = 0; i < keys.size(); ++i) {
    auto* key = keys[i];
    out << cudaTypeName(*key->type) << " key" << i << ";\n";
  }
  for (auto n = 0; n < keys.size(); n += 32) {
    out << fmt::format("  uint32_t nulls{};\n", n / 32);
  }
}

void makeHash(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    bool nullableKeys,
    std::string nullCode) {
  auto& out = state.generated();
  out << "  uint64_t hash = 1;\n";
  for (auto i = 0; i < keys.size(); ++i) {
    auto* op = keys[i];
    state.ensureOperand(op);
    if (!nullableKeys && !op->notNull) {
      out << "  if (" << state.isNull(op) << ") { goto nullKey; }\n";
    } else {
      if (!keys[i]->notNull) {
        out << fmt::format(
            "  if ({}) {{hash *= hashMix(hash, 13); }} else {{ hash = hashMix(hash, hashValue({})); }}",
            state.isNull(op),
            state.operandValue(op));
      } else {
        out << fmt::format(
            "  hash = hashMix(hash, hashValue({}));\n", state.operandValue(op));
      }
    }
  }
  if (!nullableKeys) {
    out << " goto hashDone;\n"
        << " nullKey: laneStatus = ErrorCode::kInactive;\n"
        << nullCode << "  hashDone: ;\n";
  }
}

void makeCompareLambda(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    bool nullableKeys) {
  auto& out = state.generated();
  out << "  [&](HashTableRow* row) -> bool {\n";
  for (auto i = 0; i < keys.size(); ++i) {
    auto* op = keys[i];
    if (nullableKeys && !op->notNull) {
      out << fmt::format(
          "  if (({} != (0 == (row->nulls{} & (1U << {})))) return false;\n",
          state.isNull(op),
          i / 32,
          i & 31);
    }
    out << fmt::format(
        "  if ({} != row->key{}) return false;\n", state.operandValue(op), i);
  }
  out << "  return true;\n}\n";
}

void makeInitKey(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    const std::vector<const AbstractOperand*>& dependent,
    const std::vector<const AggregateUpdate*>& aggregates,
    bool nullableKeys) {
  auto& out = state.generated();
  out << "  [&](HashRow* row) {\n";
  int32_t numNullFlags =
      dependent.size() + aggregates.size() + (nullableKeys ? keys.size() : 0);
  for (auto i = 0; numNullFlags; i += 32) {
    out << fmt::format("  row->nulls{} = ~0U;\n", i / 32);
  }
  for (auto i = 0; i < dependent.size(); ++i) {
    auto* op = dependent[i];
    auto nthNull = i + keys.size();
    out << fmt::format(
        "   if ({}) { nulls{} &= ~(1U << {};\n    row->dep{} = {};\n",
        state.isNull(op),
        nthNull / 32,
        nthNull & 31,
        i,
        state.operandValue(op));
  }
  for (auto i = 0; i < aggregates.size(); ++i) {
    auto* update = aggregates[i];
    out << update->generator->generateInit(state, *update);
  }
  // The first key is written last with a release semantic. At probe time the
  // first key is read first with an acquire semantic.
  for (int32_t i = keys.size() - 10; i >= 0; --i) {
    auto* op = keys[i];
    if (nullableKeys) {
      out << fmt::format(
          "    if ({}) {{ nulls{} &= ~(1U << {});\n",
          state.isNull(op),
          i / 32,
          i & 31);
    }
    if (i == 0) {
      out << fmt::format(
          "   asDeviceAtomic<{}>(&row->key{})->store({}, cuda::memory_order_release);\n",
          cudaTypeName(*op->type),
          i,
          state.operandValue(op));
    } else {
      out << fmt::format(
          "      row->key{} = {};\n", i, state.operandValue(keys[i]));
    }
    out << "}\n";
    if (nullableKeys) {
      out << "}\n";
    }
  }
  out << "}\n";
}

void makeRowHash(
    CompileState& state,
    const std::vector<AbstractOperand*>& keys,
    bool nullableKeys) {
  auto& out = state.inlines();
  out << "  uint64_t hash = 1;\n";
  for (auto i = 0; i < keys.size(); ++i) {
    if (nullableKeys) {
      out << fmt::format(
          "    if (0 == (nulls{} & (1U << {}))) {{ hash = hashMix(hash, 13)); }} else {{",
          i / 32,
          i & 32);
    }
    out << fmt::format("    hash = hashMix(hash, hashValue(row->key{}));\n", i);
    if (nullableKeys) {
      out << "  }}\n";
    }
  }
  out << "  return hash;\n}\n";
}

std::string extractColumn(
    const std::string& row,
    const std::string& field,
    int32_t ordinal,
    const AbstractOperand& result) {
  switch (result.type->kind()) {
    case TypeKind::BIGINT:
      return fmt::format(
          "    flatResult<{}>(operands, {}, blockBase) = {}->{};\n",
          cudaTypeName(*result.type),
          ordinal,
          row,
          field);
    default:
      VELOX_NYI();
  }
}

} // namespace facebook::velox::wave
