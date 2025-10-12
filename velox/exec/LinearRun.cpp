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

#include "velox/exec/Linear.h"
#include "velox/core/Expressions.h"
#include "velox/exec/ProjectSequence.h"
#include "velox/exec/Task.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec {

void ExprProgram::eval(EvalCtx* ctx, int32_t begin, int32_t end, RunState& runState) {
  for (auto i = begin; i < end; ++i) {
    const auto& instruction = *instructions_[i];
    switch (instruction.opCode()) {
      case Instruction::OpCode::kIf: {
        auto ifInst = instruction.as<If>();
        auto flags = runState.state[ifInst->condition()]->as<FlatVector<bool>>();
        // TODO: Implement If instruction logic
        break;
      }
      case Instruction::OpCode::kNulls:
        // TODO: Implement Nulls instruction logic
        break;
      case Instruction::OpCode::kNullsEnd:
        // TODO: Implement NullsEnd instruction logic
        break;
      default:
        break;
    }
  }
}
}





