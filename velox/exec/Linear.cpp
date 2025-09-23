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

namespace facebook::velox::exec {

  OperandIdx TranslateCtx::makeCall(std::string& name, TypePtr type, std::vector<core::TypedExprPtr>& inputs) {
    
  }
  
OperandIdx TranslateCtx::translateExpr(
    const core::TypedExprPtr& expr,
    ExprProgram& program,
    OperandIdx result,
    bool fixedResult) {
  switch (expr->kind()) {
    case core::ExprKind::kFieldAccess:
    case core::ExprKind::kDereference: {
      auto it = stage_.fieldToOperand.find(expr.get());
      if (it != state_.fieldToOperand.end()) {
        return it->second;
      }
      VELOX_FAIL("Would expect to have getters defined for :" << expr->toString();
    }

    case core::ExprKind::kConstant: {
      auto it = constants_.find(expr);
      if (it == constants_.end()) {
        auto idx = stateCounter_++;
        constants_[expr.get()] = idx;
        return idx;
      }
      return it->second;
    }

    case core::ExprKind::kCall:
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      return makeCall(call->name(), expr->type(), call->inputs());
  }
  case core::ExprKind::kCast:
    return makeCall("cast", expr->type(), call->inputs());
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

} // namespace facebook::velox::exec
