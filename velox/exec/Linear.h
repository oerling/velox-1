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

#include "velox/core/ITypedExpr.h"
#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

struct ValueInfo {
  ValueInfo() = default;
  ValueInfo(
      bool notNull,
      bool recursiveNotNull,
      std::vector<ValueInfo> children = {},
      bool defaultNulls = false)
      : notNull(notNull),
        recursiveNotNull(recursiveNotNull),
        allDefaultNullBehavior(defaultNulls),
        children(std::move(children)) {}

  bool notNull{false};
  bool recursiveNotNull{false};
  bool allDefaultNullBehavior{false};
  core::TypedExprPtr constant;
  std::vector<ValueInfo> children;
};

using ValueInfoMap = std::unordered_map<const core::ITypedExpr*, ValueInfo>;

const ValueInfo* valueInfo(
    const core::ITypedExpr* expr,
    const ValueInfoMap& map);

ValueInfo makeEmptyValueInfo(const TypePtr& type);

ValueInfo makeDefaultValueInfo(const TypePtr& type);

/// Map from expr to nullness etc for value and its subfields.
struct ValueCtx {
  std::unordered_map<const core::ITypedExpr*, ValueInfo> valueInfo;
};

/// Function that can rewrite an expr based on knowledge of input ValueInfo.
/// Returns 'expr' if no change, else returns the new expr. Must fill in
/// ValueInfo for any new expr, this produces.
using LinearRewrite = std::function<
    core::TypedExprPtr(const core::TypedExprPtr& expr, ValueCtx& ctx)>;

using MakeValueInfo =
    std::function<ValueInfo(const core::TypedExprPtr& expr, ValueCtx& ctx)>;

struct FunctionLinearMetadata {
  /// True if the function can move a whole argument to the result. For example,
  /// plus can update arguments in place if there are no other uses.
  bool mayMoveArgToResult{false};

  /// True if an argument may be contained inside a complex type result. This is
  /// true of row/map/array constructors and similar.
  bool resultMayContainArg{false};

  /// True if all non-null arguments can produce a null.
  bool nullFromNonNull{false};

  /// Trus if can operate on elements given by SelectivityVector, leaving
  /// non-selected elements in place. False for Koski functions.
  bool supportsSelectivityVector{true};

  /// Ordinal of argument that may be moved unmodified to result.
  std::optional<int32_t> maybeMovedArg;

  MakeValueInfo makeValueInfo;

  LinearRewrite rewrite;
};

FunctionLinearMetadata linearMetadata(const std::string& name);

void registerLinearMetadata(
    const std::string& name,
    FunctionLinearMetadata metadata);

void setupLinearMetadata();

using OperandIdx = uint32_t;
constexpr OperandIdx kNoOperand = ~0;
constexpr OperandIdx kMultiple = 0x80000000;

inline bool isOnlyUse(OperandIdx idx) {
  return (idx & kMultiple) == 0;
}

inline uint32_t operandIdx(OperandIdx idx) {
  return idx & ~kMultiple;
}

class TranslateCtx;

using TranslateSpecialForm = std::function<OperandIdx(
    TranslateCtx& ctx,
    const core::CallTypedExpr* call,
    std::optional<OperandIdx> result)>;

TranslateSpecialForm specialForm(const std::string& name);

void registerSpecialForm(const std::string& name, TranslateSpecialForm form);

void setupSpecialForms();

class Instruction {
 public:
  enum class OpCode : uint8_t {
    kNulls,
    kNullsEnd,
    kIf,
    kCoalesce,
    kCall,
    kField
  };

  Instruction(OpCode opCode, OperandIdx result)
      : opCode_(opCode), result_(result) {}

  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(this);
  }

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(this);
  }

  OpCode opCode() const {
    return opCode_;
  }

  OperandIdx result() const {
    return result_;
  }

 protected:
  OpCode opCode_;
  OperandIdx result_;
  OperandIdx standbyResult{kNoOperand};
};

class Field : public Instruction {
 public:
  Field(OperandIdx result, OperandIdx input, int32_t childIdx)
      : Instruction(OpCode::kField, result),
        input_(input),
        childIdx_(childIdx) {}

  OperandIdx input() const {
    return input_;
  }
  int32_t childIdx() const {
    return childIdx_;
  }

 private:
  OperandIdx input_;
  int32_t childIdx_;
};

class If : public Instruction {
 public:
     OperandIdx cond,
     int32_t _elseIdx,
     int32_t _endIdx)
      : Instruction(OpCode::kIf, kNoOperand),
        condition_(cond),
        elseIdx_(_elseIdx),
        endIdx_(_endIdx) {}

  OperandIdx condition() const {
    return condition_;
  }

  int32_t elseIdx() const {
    return elseIdx_;
  }
  int32_t endIdx() const {
    return endIdx_;
  }

  void setElse(int32_t idx) {
    elseIdx_ = idx;
  }
  void setEnd(int32_t idx) {
    endIdx_ = idx;
  }

 private:
  OperandIdx condition_;
  OperandIdx temp_{kNoOperand};
  int32_t elseIdx_;
  int32_t endIdx_;
};

class Nulls : public Instruction {
 public:
  Nulls(
      OperandIdx result,
      std::vector<OperandIdx> operands,
      int32_t nullFlagIdx)
      : Instruction(OpCode::kNulls, result),
        operands_(std::move(operands)),
        nullFlagIdx_(nullFlagIdx) {}

  const std::vector<OperandIdx>& operands() const {
    return operands_;
  }
  int32_t nullFlagIdx() const {
    return nullFlagIdx_;
  }

  std::vector<OperandIdx> operands_;
  int32_t nullFlagIdx_;
};

class NullsEnd : public Instruction {
 public:
  NullsEnd(OperandIdx result, int32_t nullFlagIdx, OperandIdx value)
      : Instruction(OpCode::kNullsEnd, result),
        nullFlagIdx_(nullFlagIdx),
        value_(value) {}

  int32_t nullFlagIdx() const {
    return nullFlagIdx_;
  }
  OperandIdx value() const {
    return value_;
  }

 private:
  int32_t nullFlagIdx_;
  OperandIdx value_;
};

class Coalesce : public Instruction {
 public:
  Coalesce(OperandIdx result, OperandIdx input, OperandIdx defaultVal)
      : Instruction(OpCode::kCoalesce, result),
        input_(input),
        default_(defaultVal) {}

  OperandIdx input() const {
    return input_;
  }
  OperandIdx defaultValue() const {
    return default_;
  }

 private:
  OperandIdx input_;
  OperandIdx default_;
};

class Call : public Instruction {
 public:
  Call(
      OperandIdx result,
      std::vector<OperandIdx> args,
      TypePtr type,
      int32_t returnedArg,
      std::shared_ptr<VectorFunction> vectorFunction,
      VectorFunctionMetadata vectorFunctionMetadata)
      : Instruction(OpCode::kCall, result),
        args_(std::move(args)),
        vectorFunction_(std::move(vectorFunction)),
        vectorFunctionMetadata_(std::move(vectorFunctionMetadata)),
        type_(std::move(type)),
        returnedArg_(returnedArg) {}

  const std::vector<OperandIdx>& args() const {
    return args_;
  }
  const TypePtr& type() const {
    return type_;
  }

  int32_t returnedArg() const {
    return returnedArg_;
  }

  const std::shared_ptr<VectorFunction>& vectorFunction() const {
    return vectorFunction_;
  }
  const VectorFunctionMetadata& vectorFunctionMetadata() const {
    return vectorFunctionMetadata_;
  }

  std::vector<OperandIdx> args_;
  std::shared_ptr<VectorFunction> vectorFunction_;
  VectorFunctionMetadata vectorFunctionMetadata_;

  // Temporary vector for flat contiguous copies of an argument for Koski
  // wrapper. kNoOperand if the corresponding arg can be used.
  std::vector<OperandIdx> argCopies_;
  TypePtr type_;

  // Ordinal of argument that has the same slot as return value. -1 if none.
  int32_t returnedArg_;
};

/// Describes how to move elements of a RowVector to an operand in state. May
/// have multiple steps, e.g. row.features.the_feature would have first the
/// index of 'features' and then 'the_feature'. Applies to both input and output
/// of a LinearExprSet.
struct Assignment {
  Assignment(std::vector<int32_t> path, OperandIdx operand, int32_t sourceRow)
      : path{path}, operand{operand}, sourceRow{sourceRow} {}

  /// The index in the outer row, next inner etc.
  std::vector<int32_t> path;

  // The position in state.
  OperandIdx operand;

  /// Designates the RowVector 'pth' starts from. 0 is input, then consecutive
  /// outputs, then temporary
  int32_t sourceRow;

  /// Designates the leaf row into which path.back() is an index.
  int32_t leafRow{-1};
};

struct RunState {
  SelectivityVector* active;
  VectorPtr* state;
  std::vector<std::unique_ptr<SelectivityVector>> selectionStack;

  BufferPtr pendingNulls;
  BufferPtr temp1;
  BufferPtr temp2;
};

/// Represents a sequential set of instructions for computing
/// mulytiple projections base on data in a state. The result is
/// deposited into the state. The program is single threaded but
/// multiple programs can run in parallel on the same state as long
/// as their leaf inputs and outputs do not overlap.
class ExprProgram {
 public:
  ExprProgram() {}

  void eval(EvalCtx* ctx, int32_t begin, int32_t end, RunState& state);

  std::vector<std::unique_ptr<Instruction>>& instructions() {
    return instructions_;
  }

  std::vector<std::unique_ptr<Instruction>> instructions_;

  // A tenporary vector for use as arguments to a Velox function.
  std::vector<std::vector<VectorPtr>> argTemp_;
  std::vector<SelectivityVector> rowsStack_;
};

bool isField(const core::TypedExprPtr& expr, std::vector<int32_t>& path);

} // namespace facebook::velox::exec
