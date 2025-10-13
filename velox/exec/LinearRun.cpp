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
#include "velox/common/base/BitUtil.h"
#include "velox/core/Expressions.h"
#include "velox/exec/ProjectSequence.h"
#include "velox/exec/Task.h"
#include "velox/expression/BooleanMix.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec {

void ExprProgram::eval(EvalCtx* ctx, int32_t begin, int32_t end, RunState& runState) {
  for (auto pc = begin; pc < end; ++pc) {
    const auto& instruction = *instructions_[pc];
    switch (instruction.opCode()) {
      case Instruction::OpCode::kIf: {
        auto ifInst = instruction.as<If>();
        auto* conditionVec = runState.vectorAt(ifInst->condition()).get();

        // Use getFlatBool to interpret the condition
        const uint64_t* values = nullptr;
        const auto booleanMix = getFlatBool(
            conditionVec,
            *runState.active,
            *ctx,
            &runState.temp1,
            &runState.temp2,
            true, // mergeNullsToValues
            &values,
            nullptr);

        switch (booleanMix) {
          case BooleanMix::kAllTrue:
            // All true: evaluate then branch, skip else
            eval(ctx, pc + 1, ifInst->elseIdx(), runState);
            pc = ifInst->endIdx() - 1; // -1 because loop will increment
            break;

          case BooleanMix::kAllFalse:
          case BooleanMix::kAllNull:
            // All false or null: evaluate else branch
            eval(ctx, ifInst->elseIdx(), ifInst->endIdx(), runState);
            pc = ifInst->endIdx() - 1; // -1 because loop will increment
            break;

          default: {
            // Mixed: need to evaluate both branches with different selections
            auto* prevSelection = runState.active;

            // Push new selection for then branch
            auto* thenSelection = runState.pushSelection();
            thenSelection->copy(*prevSelection);

            // AND with condition values to get rows where condition is true
            bits::andBits(
                thenSelection->asMutableRange().bits(),
                prevSelection->asRange().bits(),
                values,
                0,
                prevSelection->end());
            thenSelection->updateBounds();

            // Evaluate then branch
            eval(ctx, pc + 1, ifInst->elseIdx(), runState);

            // Copy previous selection back to active
            runState.active->copy(*prevSelection);

            // AND with negated condition values for else branch
            bits::andWithNegatedBits(
                runState.active->asMutableRange().bits(),
                prevSelection->asRange().bits(),
                values,
                0,
                prevSelection->end());
            runState.active->updateBounds();

            // Evaluate else branch
            eval(ctx, ifInst->elseIdx(), ifInst->endIdx(), runState);

            // Pop selection
            runState.popSelection();

            pc = ifInst->endIdx() - 1; // -1 because loop will increment
            break;
          }
        }
        break;
      }
      case Instruction::OpCode::kNulls: {
        auto nullsInst = instruction.as<Nulls>();
        const auto& operands = nullsInst->operands();

        // Check if any operand may have nulls
        bool anyNulls = false;
        for (auto operandIdx : operands) {
          if (runState.vectorAt(operandIdx)->mayHaveNulls()) {
            anyNulls = true;
            break;
          }
        }

        // If no operand has nulls, set noNulls flag and skip
        if (!anyNulls) {
          runState.noNulls = true;
          break;
        }

        // Some operands have nulls
        runState.noNulls = false;

        // Get the size from the active selection
        auto size = runState.active->end();

        // Allocate pendingNulls buffer if not already allocated
        if (!runState.pendingNulls || runState.pendingNulls->size() < bits::nwords(size) * sizeof(uint64_t)) {
          runState.pendingNulls = AlignedBuffer::allocate<bool>(size, ctx->pool());
        }

        auto* tempNulls = runState.pendingNulls->asMutable<uint64_t>();

        // Find first operand with nulls and copy its null bits
        bool firstFound = false;
        for (auto operandIdx : operands) {
          auto& vec = runState.vectorAt(operandIdx);
          if (vec->mayHaveNulls()) {
            auto* rawNulls = vec->rawNulls();
            if (!firstFound) {
              // Copy first null buffer
              std::memcpy(
                  tempNulls,
                  rawNulls,
                  bits::nwords(size) * sizeof(uint64_t));
              firstFound = true;
            } else {
              // AND subsequent null buffers
              bits::andBits(
                  tempNulls,
                  tempNulls,
                  rawNulls,
                  runState.active->begin(),
                  runState.active->end());
            }
          }
        }

        // Save current active selection before pushing
        auto* prevSelection = runState.active;

        // Push a new selection and copy the previous selection
        auto* newSelection = runState.pushSelection();
        newSelection->copy(*prevSelection);

        // Deselect rows that have nulls
        newSelection->deselectNulls(tempNulls, prevSelection->begin(), prevSelection->end());

        break;
      }
      case Instruction::OpCode::kNullsEnd: {
        // If no nulls were present, do nothing
        if (runState.noNulls) {
          break;
        }

        auto nullsEndInst = instruction.as<NullsEnd>();
        auto& result = runState.vectorAt(nullsEndInst->result());

        // Ensure result has a nulls buffer
        result->ensureNulls();

        // Pop the selection back to the previous level
        runState.popSelection();

        // Get pendingNulls buffer
        auto* tempNulls = runState.pendingNulls->asMutable<uint64_t>();

        // AND negated selection over tempNulls
        // This marks as null any rows that were deselected
        bits::andWithNegatedBits(
            tempNulls,
            runState.active->asRange().bits(),
            runState.active->begin(),
            runState.active->end());

        // AND tempNulls over rawNulls in the result
        // This propagates the null bits to the result
        bits::andBits(
            const_cast<uint64_t*>(result->rawNulls()),
            tempNulls,
            runState.active->begin(),
            runState.active->end());

        break;
      }
      default:
        break;
    }
  }
}
}





