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

#include "velox/experimental/wave/vector/Operand.h"

namespace facebook::velox::wave::aggregation {

  /// Describes an aggregate host side for generating device side operations.
  class AggregateOperation {
  public:
    /// Emits a target language declaration for an accumulator.
    declare(std::vector<TypePtr>& types, const std::string& fieldName, std::ostream& out);

    void addRaw(const char* group, bool isGroup, std::vector<AbstractOperand*> operands, std::ostream& out);

    void addAccumulator(const char* group, bool isGroup, std::vector<TypePtr>& types, std::vector<AbstractOperand*> operands, std::ostream& out);

    void extractValue(AbstractOperand* result, std::ostream& out);
    
    void extractAccumulator(AbstractOperand* result, std::ostream& out);

    void resultElements(AbstractOperand& result, std::ostream& out);


    
    

  };

  struct AggregateFunction {
  int accumulatorSize;

  ErrorCode (
      *addRawInput)(int numInputs, Operand* inputs, int i, void* accumulator);

  ErrorCode (*extractValues)(void* accumulator, Operand* result, int i);

  void* (*mergeAccumulators)(void* a, void* b);
};

} // namespace facebook::velox::wave::aggregation
