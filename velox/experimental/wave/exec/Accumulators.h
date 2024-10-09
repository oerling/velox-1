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

#pragma once

#include "velox/experimental/wave/exec/Instruction.h"

namespace facebook::velox::wave::aggregation {

/// Generates code for updating and accessing accumulators. Specialized for
/// different aggregate functions.
class AccumulatorGenerator {
 public:
  /// Emits a target language declaration for an accumulator.
  virtual void declare(
      std::vector<TypePtr>& types,
      const std::string& fieldName,
      std::ostream& out);

  virtual void addRaw(
      const char* group,
      const std::string& fieldName,
      bool isGroup,
      std::vector<AbstractOperand*> operands,
      std::ostream& out);

  virtual void addAccumulator(
      const char* group,
      const std::string& fieldName,
      bool isGroup,
      std::vector<TypePtr>& types,
      std::vector<AbstractOperand*> operands,
      std::ostream& out);

  virtual void extractValue(
      AbstractOperand* result,
      const std::string& fieldName,
      std::ostream& out);

  virtual void extractAccumulator(
      AbstractOperand* result,
      const std::string& fieldName,
      std::ostream& out);

  virtual void resultElements(
      AbstractOperand& result,
      const std::string& fieldName,
      std::ostream& out);
};

} // namespace facebook::velox::wave::aggregation
