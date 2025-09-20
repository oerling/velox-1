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
    /// True if the function can move a whole argument to the result. For example, plus can update arguments in place if there are no other uses.
    bool mayMoveArgToResult{false};

    /// True if an argument may be contained inside a complex type result. This is true of row/map/array constructors and similar.
    bool resultMayContainArg{false};
  };

  FunctionLinearMetadata linearMetadata(const std::string& name);

  using OperandIdx = int32_t;
  constexpr OperandIdx kNoOperand = -1;
class Instruction {
 public:
  enum class OpCode : uint8_t { kNulls, kIf, kCall };

  template <typename>
  as() const {
    return reinterpret_cast<const T*>(this);
  }

 protected:
  OpCode opCode_;
};

 class Field : public Instruction {
 public:
   OperandIdx input_;
   int32_t childIdx_;
 };

 class If : public Instruction {
 public:
  OperandIdx condition;
  int32_t else;
  int32_t end;
} if;

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

/// Describes how to move elements of a RowVector to an operand in state. May have multiple steps, e.g. row.features.the_feature would have first the index of 'features' and then 'the_feature'. Applies to both input and output of a LinearExprSet.
 struct Assignment {
   std::vector<column_index_t> path;
   OperandIdx operand;
 };
 
  /// Represents a sequential set of instructions for computing
  /// mulytiple projections base on data in a state. The result is
  /// deposited into the state. The program is single threaded but
  /// multiple programs can run in parallel on the same state as long
  /// as their outputs do not overlap.
  class ExprProgram {
 public:
    ExprProgram(std::vector<std::unique_ptr<Instruction>>& instructions  );

    eval(EvalCtx* ctx, int32_t begin, int32_t end, VectorPtr* state);

  

  std::vector<std::unique_ptr<Instruction>> instructions_;


  RowVectorPtr result_;
  RowVectorPtr input_;
  
  /// Children of context RowVector.
  std::vector<VectorPtr> input_;
  //A tenporary vector for use as arguments to a Velox function. 
  std::vector<std::vector<VectorPtr>> argTemp_;
  std::vector<SelectivityVector> rowsStack_;
};




} // namespace facebook::velox::exec
