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

#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

struct FunctionLinearMetadata {
  /// True if the function can move a whole argument to the result. For example,
  /// plus can update arguments in place if there are no other uses.
  bool mayMoveArgToResult{false};

  /// True if an argument may be contained inside a complex type result. This is
  /// true of row/map/array constructors and similar.
  bool resultMayContainArg{false};
};

FunctionLinearMetadata linearMetadata(const std::string& name);

using OperandIdx = uint32_t;
constexpr OperandIdx kNoOperand = ~0;
  constexpr OperandIdx kMultiple = 0x80000000;

  inline   uint32_t operandIdx(OperandIdx idx) {
    return idx & ~kMultiple;
  }

  class Instruction {
 public:
  enum class OpCode : uint8_t { kNulls, kIf, kCall };

  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(this);
  }

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(this);
  }

 protected:
  OpCode opCode_;
  OperandIdx result_;
};

class Field : public Instruction {
 public:
  OperandIdx input_;
  int32_t childIdx_;
};

class If : public Instruction {
 public:
  OperandIdx condition;
  int32_t elseIdx;
  int32_t endIdx;
};

class Nulls : public Instruction {
 public:
  std::vector<OperandIdx> operands_;
  int32_t end;
};

class Coalesce : public Instruction {
 public:
  OperandIdx input_;
  OperandIdx default_;
  OperandIdx result_;
};

class Call : public Instruction {
 public:
  int32_t result_;
  std::vector<int32_t> args_;
  TypePtr type_;
  bool mayReturnInput_;
};

/// Describes how to move elements of a RowVector to an operand in state. May
/// have multiple steps, e.g. row.features.the_feature would have first the
/// index of 'features' and then 'the_feature'. Applies to both input and output
/// of a LinearExprSet.
struct Assignment {
  Assignment(std::vector<int32_t> path, int32_t operand, int32_t sourceRow)
    : path{path}, operand{operand}, sourceRow{sourceRow} {}

  /// The index in the outer row, next inner etc.
  std::vector<column_index_t> path;

  // The position in state.
  OperandIdx operand;

  /// Designates the RowVector 'pth' starts from. 0 is input, then consecutive outputs, then temporary
  int32_t sourceRow;
};

/// Represents a sequential set of instructions for computing
/// mulytiple projections base on data in a state. The result is
/// deposited into the state. The program is single threaded but
/// multiple programs can run in parallel on the same state as long
/// as their leaf inputs and outputs do not overlap.
class ExprProgram {
 public:
  ExprProgram(std::vector<std::unique_ptr<Instruction>>& instructions);

  void eval(EvalCtx* ctx, int32_t begin, int32_t end, VectorPtr* state);

  std::vector<std::unique_ptr<Instruction>> instructions_;

  // A tenporary vector for use as arguments to a Velox function.
  std::vector<std::vector<VectorPtr>> argTemp_;
  std::vector<SelectivityVector> rowsStack_;
};

struct TypeHasher {
  size_t operator()(const velox::TypePtr& type) const {
    // hash on recursive TypeKind. Structs that differ in field names
    // only or decimals with different precisions will collide, no
    // other collisions expected.
    return type->hashKind();
  }
};

struct TypeComparer {
  bool operator()(const velox::TypePtr& lhs, const velox::TypePtr& rhs) const {
    return *lhs == *rhs;
  }
};

struct ITypedExprHasher {
  size_t operator()(const velox::core::ITypedExpr* expr) const {
    return expr->hash();
  }
};

struct ITypedExprComparer {
  bool operator()(
      const velox::core::ITypedExpr* lhs,
      const velox::core::ITypedExpr* rhs) const {
    return *lhs == *rhs;
  }
};

/// Map from leaf expr to OperandIdx. The leaf expr can be a input field
/// reference or stack of struct field getters, an named intermediate or
/// subfield thereof.
using ExprOperandMap = folly::F14FastMap<
    const velox::core::ITypedExpr*,
    OperandIdx,
    ITypedExprHasher,
    ITypedExprComparer>;

/// State during conversion from TypedExpr to ExprProgram
class TranslateCtx {
 public:
  void translateExpr(const core::TypedExprPtr&, ExprProgram& program);

  RowTypePtr inputType_;
  RowTypePtr outputType_;
  // Operands are checked non-nul for active rows.
  bool inNullPropagating_{false};

  // Maps from any accessed field or subfield of input to OperandIdx.
  std::vector<Assignment> inputAssignments_;

  // Maps from any field or struct subfield  of the final result to OperandIdx.
  std::vector<Assignment> outputAssignments_;

  ExprOperandMap fieldToOperand_;

  /// Map from type to operand index for temporary variables. A temp is a vector
  /// that is in none of   input, named intermediate  or final output.
  std::unordered_map<TypePtr, std::vector<OperandIdx>, TypeHasher, TypeComparer>
      tempVectors_;
};

  bool isField(const TypedExprPtr& expr, std::vector<int32_t>& path);

      } // namespace facebook::velox::exec
