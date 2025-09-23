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


  
  OperandIdx TranslateCtx::makeCall(std::string& name, const TypePtr& type, std::vector<core::TypedExprPtr>& inputs, OperandIdx result, bool fixedResult) {
    auto metadata = linearMetadata(name);
    
  }

  void  TranslateCtx::makeSwitch(const TypePtr& type, std::vector<core::TypedExprPtr>& inputs) {
    for (auto i = 0; i < inputs.size(); i += 2) {
      OperandIdx cond = getTemp(BOOLEAN());
      translateExpr(inputs[i], cond, true);
      program.add
    }
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
      auto& constants = projectSequence_->constants();
      auto it = constants.find(expr);
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
	  vector = BaseVector::createConstant(constantExpr->type(), constantExpr->value(), 1, projectSequence_->operatorCtx_->pool());
	}
	  temps[idx] = vector;
        constants[expr.get()] = idx;
        return idx;
      }
      return it->second;
    }

    case core::ExprKind::kCall:
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      return makeCall(call->name(), expr->type(), call->inputs());
  }
  case core::ExprKind::kCast:
    return makeCall("cast", expr->type(), expr->inputs());
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
