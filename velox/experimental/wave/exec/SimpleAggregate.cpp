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
#include "velox/experimental/wave/exec/ToWave.h"

namespace facebook::velox::wave {

class SimpleAggregate : public AggregateGenerator {
 public:
  explicit SimpleAggregate(const std::string& binaryFunc)
      : AggregateGenerator(false), binaryFunc_(binaryFunc) {}

  void generateInline(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    std::vector<TypePtr> types;
    types.push_back(update.result->type);
    types.push_back(update.result->type);
    state.functionReferenced(binaryFunc_, types, types[0]);
  }

  std::string generateAccumulator(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    std::stringstream out;
    out << cudaTypeName(*update.result->type) << " ";
    return out.str();
  }

  std::string generateInit(
      CompileState& state,
      const AggregateUpdate& update) const override {
    return fmt::format("  acc{} = 0;\n", update.accumulatorIdx);
  }
  
  std::string generateUpdate(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    std::stringstream out;
    auto nullable = !update.args[0]->notNull;
    if (nullable) {
      out << fmt::format("   if (!{}) {{\n", state.isNull(update.args[0]));
    }
    if (binaryFunc_ == "plus") {
      out << fmt::format("      atomicAdd(reinterpret_cast<{}*>(&row->acc{}), {});\n", cudaAtomicTypeName(*update.args[0]->type), update.accumulatorIdx, state.operandValue(update.args[0]));
    } else {
      VELOX_NYI("Only plus is supported as aggregate reduction");
    }
    if (nullable) {
      out << "}}\n";
    }
    return out.str();
  }

  std::string generateExtract(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    auto ord = state.ordinal(*update.result);
    auto nthNull = update.accumulatorIdx + probe.keys.size();
    return fmt::format("   setNull(operands, {}, blockBase, (row->nulls{} & (1U << {})) == 0);\n"
		       "    flatValue<T>(operands, {}, blockBase) = {};\n" , ord, nthNull / 32, nthNull & 31,
		       ord, update.accumulatorIdx);
  }

 protected:
  std::string binaryFunc_;
};

namespace {
bool temp = CompileState::aggregateRegistry().registerGenerator(
    "SUM",
    std::make_unique<SimpleAggregate>("plus"));
}

} // namespace facebook::velox::wave
