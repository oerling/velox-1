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

namespace {
template <TypeKind Kind>
void setAtNulls(
    const uint64_t* nulls,
    vector_size_t end,
    const uint64_t* active,
    uint64_t* temp,
    const VectorPtr& constant,
    VectorPtr& target) {
  using T = typename TypeTraits<Kind>::NativeType;

  // Cast constant to ConstantVector and get value
  auto* constantVec = constant->as<ConstantVector<T>>();
  auto value = constantVec->valueAt(0);

  // Cast target to FlatVector
  auto* flatTarget = target->as<FlatVector<T>>();
  auto* rawValues = flatTarget->mutableRawValues();

  // Prepare bit mask: OR of negated active and nulls
  const uint64_t* bitMask;
  if (active) {
    // OR negated active with nulls into temp
    auto numWords = bits::nwords(end);
    for (size_t i = 0; i < numWords; ++i) {
      temp[i] = (~active[i]) | nulls[i];
    }
    bitMask = temp;
  } else {
    // Use nulls directly as bit mask
    bitMask = nulls;
  }

  // Use forEachUnsetBit to loop over nulls and set positions to constant
  bits::forEachUnsetBit(bitMask, 0, end, [&](vector_size_t i) {
    rawValues[i] = value;
  });
}

void setEmptyAtNull(
    const uint64_t* nulls,
    vector_size_t end,
    const uint64_t* active,
    uint64_t* temp,
    VectorPtr& target) {
  // Cast target to ArrayVectorBase (works for both ArrayVector and MapVector)
  auto* arrayBase = target->as<ArrayVectorBase>();

  // Make offsets and sizes mutable
  arrayBase->mutableOffsets(end);
  arrayBase->mutableSizes(end);

  // Prepare bit mask: OR of negated active and nulls
  const uint64_t* bitMask;
  if (active) {
    // OR negated active with nulls into temp
    auto numWords = bits::nwords(end);
    for (size_t i = 0; i < numWords; ++i) {
      temp[i] = (~active[i]) | nulls[i];
    }
    bitMask = temp;
  } else {
    // Use nulls directly as bit mask
    bitMask = nulls;
  }

  // Use forEachUnsetBit to loop over nulls and set offset and size to 0
  bits::forEachUnsetBit(bitMask, 0, end, [&](vector_size_t i) {
    arrayBase->setOffsetAndSize(i, 0, 0);
  });
}

void evalCoalesce(const Coalesce* coalesceInst, RunState& runState, EvalCtx* ctx) {
  auto resultIdx = coalesceInst->result();
  auto inputIdx = coalesceInst->input();

  // If result and input are the same
  if (resultIdx == inputIdx) {
    auto& inputVec = runState.vectorAt(inputIdx);

    // If mayHaveNulls is false, return
    if (!inputVec->mayHaveNulls()) {
      return;
    }

    // Make input vector writable
    inputVec->setMutable(true);

    auto encoding = inputVec->encoding();

    // Consider MAP and ARRAY encodings as flat, otherwise flatten
    if (encoding != VectorEncoding::Simple::FLAT &&
        encoding != VectorEncoding::Simple::ARRAY &&
        encoding != VectorEncoding::Simple::MAP) {
      inputVec->ensureWritable(*runState.active);
    }

    // Get the default value vector
    auto& defaultVec = runState.vectorAt(coalesceInst->defaultValue());

    // Get nulls buffer and active selection
    auto* nulls = inputVec->rawNulls();
    auto end = runState.active->end();
    const uint64_t* activeBits = runState.active ? runState.active->asRange().bits() : nullptr;

    // Allocate temp buffer for bit operations if needed
    if (!runState.temp1 || runState.temp1->size() < bits::nwords(end) * sizeof(uint64_t)) {
      runState.temp1 = AlignedBuffer::allocate<bool>(end, ctx->pool());
    }
    auto* temp = runState.temp1->asMutable<uint64_t>();

    // Handle MAP and ARRAY types differently
    if (encoding == VectorEncoding::Simple::ARRAY ||
        encoding == VectorEncoding::Simple::MAP) {
      // Call setEmptyAtNull for MAP and ARRAY encodings
      setEmptyAtNull(nulls, end, activeBits, temp, inputVec);
    } else {
      // Call setAtNulls with dynamic type dispatch for scalar types
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          setAtNulls,
          inputVec->typeKind(),
          nulls,
          end,
          activeBits,
          temp,
          defaultVec,
          inputVec);
    }
    inputVec->clearAllNulls();
  }
}
} // namespace

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
      case Instruction::OpCode::kCoalesce: {
        auto coalesceInst = instruction.as<Coalesce>();
        evalCoalesce(coalesceInst, runState, ctx);
        break;
      }
      default:
        break;
    }
  }
}
}





