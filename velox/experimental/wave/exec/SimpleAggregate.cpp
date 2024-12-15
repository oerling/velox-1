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

  std::string generateUpdate(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    return "";
  }

  std::string generateExtract(
      CompileState& state,
      const AggregateProbe& probe,
      const AggregateUpdate& update) const override {
    return "";
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
