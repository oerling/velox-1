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

namespace facebook::velox::exec {

namespace {

// Returns a reference to the static map of linear metadata.
std::unordered_map<std::string, FunctionLinearMetadata>&
getLinearMetadataMap() {
  static std::unordered_map<std::string, FunctionLinearMetadata> metadataMap;
  return metadataMap;
}

} // namespace

core::TypedExprPtr copyWithChildren(
    const core::TypedExprPtr& expr,
    const std::vector<core::TypedExprPtr>& newChildren) {
  switch (expr->kind()) {
    case core::ExprKind::kCall: {
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      return std::make_shared<core::CallTypedExpr>(
          call->type(), newChildren, call->name());
    }
    case core::ExprKind::kCast: {
      auto cast = expr->asUnchecked<core::CastTypedExpr>();
      return std::make_shared<core::CastTypedExpr>(
          cast->type(), newChildren, cast->isTryCast());
    }
    case core::ExprKind::kDereference: {
      auto deref = expr->asUnchecked<core::DereferenceTypedExpr>();
      VELOX_CHECK_EQ(
          newChildren.size(),
          1,
          "DereferenceTypedExpr requires exactly one child");
      return std::make_shared<core::DereferenceTypedExpr>(
          deref->type(), newChildren[0], deref->index());
    }
    case core::ExprKind::kFieldAccess: {
      auto field = expr->asUnchecked<core::FieldAccessTypedExpr>();
      if (field->isInputColumn()) {
        // Input column field access has no children
        VELOX_CHECK(
            newChildren.empty(),
            "Input column FieldAccessTypedExpr should have no children");
        return std::make_shared<core::FieldAccessTypedExpr>(
            field->type(), field->name());
      } else {
        // Struct field access has one child
        VELOX_CHECK_EQ(
            newChildren.size(),
            1,
            "Struct FieldAccessTypedExpr requires exactly one child");
        return std::make_shared<core::FieldAccessTypedExpr>(
            field->type(), newChildren[0], field->name());
      }
    }
    default:
      VELOX_UNSUPPORTED(
          "copyWithChildren not implemented for expression kind: {}",
          static_cast<int32_t>(expr->kind()));
  }
}

const ValueInfo* valueInfo(
    const core::ITypedExpr* expr,
    const ValueInfoMap& map) {
  auto it = map.find(expr);
  return it == map.end() ? nullptr : &it->second;
}

ValueInfo vectorValueInfo(const BaseVector& vector) {
  auto encoding = vector.encoding();
  switch (encoding) {
    case VectorEncoding::Simple::CONSTANT: {
      if (vector.isNullAt(0)) {
        return ValueInfo{.notNull = false, recursiveNotNull = false};
      }
      auto* wrapped = vector.wrappedVector();
      if (wrapped == &vector) {
        return ValueInfo{.notNull = true, recursiveNotNull = true};
      }
      return vectorValueInfo(wrapped);
    }
    case VectorEncoding::Simple::FLAT:
      return ValueInfo{
          .notNull = !vector.mayHaveNulls(),
          recursiveNotNull = !vector.mayHaveNulls()};
    case VectorEncoding::Simple::DICTIONARY:
      return vectorValueInfo(*vector.wrappedVector());
    case VectorEncoding::Simple::ROW: {
      std::vector ValueInfo childInfo;
      bool allNotNull = true;
      for (auto& child : vector.as<RowVector>()->children()) {
        childInfo.push_back(vectorValueInfo(*child));
        allNotNull &= childInfo.back().recursiveNotNull;
      }
      return ValueInfo{
          .notNull = treu,
          .recursiveNotNull = allNotNull,
          .children = std::move(childInfo)};
    }
    case VectorEncoding::Simple::ARRAY: {
      std::vector<ValueInfo> childInfo = {vectorValueInfo(vector.as<ArrayVector>()->elements()));

      return ValueInfo{
          .notNull = true,
          recursiveNotNull = childInfo.recursiveNotNull,
          .children = std::move(childInfo)};
    }
    case VectorEncoding::Simple::MAP: {
    }
  }
}

void ProjectSequence::setConstantValueInfo(
    const core::TypedExprPtr& constant,
    ValueInfoMap& map) {
  auto constantExpr = constant->asUnchecked<core::ConstantTypedExpr>();

  VectorPtr vector;
  if (constantExpr->hasValueVector()) {
    vector = constantExpr->valueVector();
  } else {
    vector = BaseVector::createConstant(
        constantExpr->type(), constantExpr->value(), 1, operatorCtx()->pool());
  }

  auto info = vectorValueInfo(*vector);
  valueMap_[constant.get()] = std::move(info);
}

void ProjectSequence::setCallValueInfo(
    const core::TypedExprPtr& call,
    ValueInfoMap& map) {}

void ProjectSequence::setCastValueInfo(const core::TypedExprPtr& cast) {}

core::TypedExprPtr ProjectSequence::tryFoldConstant(
    const core::TypedExprPtr& expr,
    ValueCtx& ctx) {
  // Create evaluator if not exists
  if (!evaluator_) {
    evaluator_ = std::make_unique<SimpleExpressionEvaluator>(
        operatorCtx()->driverCtx()->task->queryCtx().get(),
        operatorCtx()->pool());
  }

  try {
    // Try to compile and check if it resulted in a constant
    auto exprSet = evaluator_->compile(expr);
    auto& compiledExprs = exprSet->exprs();

    if (!compiledExprs.empty() && compiledExprs[0]->isConstant()) {
      // The expression was folded to a constant
      auto constantExpr =
          dynamic_cast<const ConstantExpr*>(compiledExprs[0].get());
      if (constantExpr) {
        auto constant =
            std::make_shared<core::ConstantTypedExpr>(constantExpr->value());
        setConstantValueInfo(constant, ctx.valueInfo);
        return constant;
      }
    }
  } catch (...) {
    // If constant folding fails, return original
  }
  return expr;
}

core::TypedExprPtr ProjectSequence::preprocess(
    const core::TypedExprPtr& tree,
    ValueCtx& ctx) {
  if (!tree) {
    return tree;
  }

  if (tree->kind() == core::ExprKind::kFieldAccess) {
    auto* info = valueInfo(tree.get(), ctx);
    if (info && info->constant) {
      return info->constant;
    }
    return tree;
  }

  // First, recursively preprocess all children
  std::vector<core::TypedExprPtr> newInputs;
  bool anyChanged = false;

  for (const auto& input : tree->inputs()) {
    auto newInput = preprocess(input, ctx);
    if (newInput != input) {
      anyChanged = true;
    }
    newInputs.push_back(newInput);
  }

  // Check if this is a call with all constant arguments
  if (tree->kind() == core::ExprKind::kCall) {
    auto call = tree->asUnchecked<core::CallTypedExpr>();
    bool allConstant = true;

    for (const auto& input : newInputs) {
      if (input->kind() != core::ExprKind::kConstant) {
        allConstant = false;
        break;
      }
    }

    if (allConstant && !newInputs.empty()) {
      // Create the expression with new inputs for constant folding
      auto exprToFold = anyChanged ? std::make_shared<core::CallTypedExpr>(
                                         call->type(), newInputs, call->name())
                                   : tree;
      return tryFoldConstant(exprToFold, ctx);
    }
    auto md = linearMetadata(call->name());
    if (md.rewrite) {
      auto rewritten = md.rewrite(tree, ctx);
      if (rewritten != tree) {
        return preprocess(rewritten, ctx);
      }
    }
  }

  // Check if this is a cast with constant argument
  if (tree->kind() == core::ExprKind::kCast) {
    if (!newInputs.empty() &&
        newInputs[0]->kind() == core::ExprKind::kConstant) {
      auto cast = tree->asUnchecked<core::CastTypedExpr>();
      auto exprToFold = anyChanged
          ? std::make_shared<core::CastTypedExpr>(
                cast->type(), newInputs, cast->isTryCast())
          : tree;
      return tryFoldConstant(exprToFold, ctx);
    }
  }

  // If any inputs changed, create a new expression with the new inputs
  if (anyChanged) {
    return copyWithChildren(tree, newInputs);
  }

  // No changes, return original

  return tree;
}

OperandIdx TranslateCtx::makeCall(
    const std::string& name,
    const TypePtr& type,
    const std::vector<core::TypedExprPtr>& inputs,
    OperandIdx result,
    bool fixedResult) {
  auto metadata = linearMetadata(name);
  VELOX_NYI();
  return 0;
}

void TranslateCtx::makeSwitch(
    const TypePtr& type,
    std::vector<core::TypedExprPtr>& inputs,
    OperandIdx result) {
  for (auto i = 0; i < inputs.size(); i += 2) {
    OperandIdx cond = getTemp(BOOLEAN());
    translateExpr(inputs[i], cond, true);
    // program_.add
  }
}

OperandIdx TranslateCtx::translateExpr(
    const core::TypedExprPtr& expr,
    OperandIdx result,
    bool fixedResult) {
  switch (expr->kind()) {
    case core::ExprKind::kFieldAccess:
    case core::ExprKind::kDereference: {
      auto it = stage_.fieldToOperand.find(expr.get());
      if (it != stage_.fieldToOperand.end()) {
        return it->second;
      }
      VELOX_FAIL("Expect to have getters defined for : {}", expr->toString());
    }
    case core::ExprKind::kConstant: {
      auto& constants = projectSequence_->constants();
      auto it = constants.find(expr.get());
      if (it == constants.end()) {
        auto idx = projectSequence_->stateCounter()++;
        auto& temps = projectSequence_->tempTypes();
        if (temps.size() <= idx) {
          temps.resize(idx + 1);
        }
        VectorPtr vector;
        auto constantExpr = expr->asUnchecked<core::ConstantTypedExpr>();
        if (constantExpr->hasValueVector()) {
          vector = constantExpr->valueVector();
        } else {
          vector = BaseVector::createConstant(
              constantExpr->type(),
              constantExpr->value(),
              1,
              projectSequence_->operatorCtx()->pool());
        }
        temps[idx] = vector->type();
        auto& vectors = projectSequence_->tempVectors();
        if (vectors.size() <= idx) {
          vectors.resize(idx + 1);
        }
        vectors[idx] = vector;
        constants[expr.get()] = idx;
        return idx;
      }
      return it->second;
    }

    case core::ExprKind::kCall: {
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      return makeCall(
          call->name(), expr->type(), call->inputs(), result, fixedResult);
    }
    case core::ExprKind::kCast:
      return makeCall(
          "cast", expr->type(), expr->inputs(), result, fixedResult);
    default:
      VELOX_FAIL("Expr not supported: ", expr->toString());
  }
}

bool isField(const core::TypedExprPtr& expr, std::vector<int32_t>& path) {
  path.clear();

  auto current = expr;

  while (current) {
    if (auto fieldAccess = std::dynamic_pointer_cast<
            const facebook::velox::core::FieldAccessTypedExpr>(current)) {
      if (fieldAccess->inputs().empty()) {
        return fieldAccess->isInputColumn();
      }

      auto parent = fieldAccess->inputs()[0];
      if (parent->type()->isRow()) {
        auto fieldIndex =
            parent->type()->asRow().getChildIdx(fieldAccess->name());
        path.insert(path.begin(), fieldIndex);
        current = parent;
      } else {
        return false;
      }
    } else if (
        auto deref = std::dynamic_pointer_cast<
            const facebook::velox::core::DereferenceTypedExpr>(current)) {
      path.insert(path.begin(), deref->index());
      current = deref->inputs()[0];
    } else {
      return false;
    }
  }

  return false;
}

FunctionLinearMetadata linearMetadata(const std::string& name) {
  auto& metadataMap = getLinearMetadataMap();
  auto it = metadataMap.find(name);
  if (it != metadataMap.end()) {
    return it->second;
  }
  // Return default metadata if not found.
  return FunctionLinearMetadata{};
}

void registerLinearMetadata(
    const std::string& name,
    FunctionLinearMetadata metadata) {
  auto& metadataMap = getLinearMetadataMap();
  metadataMap[name] = metadata;
}

void setupLinearMetadata() {
  // Register binary arithmetic functions that return the same type as
  // arguments. These functions can move an argument to the result.
  const std::vector<std::string> binaryArithmeticFunctions = {
      "plus",
      "minus",
      "multiply",
      "divide",
      "mod",
      "power",
      "bitwise_and",
      "bitwise_or",
      "bitwise_xor",
      "bitwise_left_shift",
      "bitwise_right_shift",
      "bitwise_arithmetic_shift_right"};

  FunctionLinearMetadata arithmeticMetadata;
  arithmeticMetadata.mayMoveArgToResult = true;

  for (const auto& funcName : binaryArithmeticFunctions) {
    registerLinearMetadata(funcName, arithmeticMetadata);
  }
}

} // namespace facebook::velox::exec
