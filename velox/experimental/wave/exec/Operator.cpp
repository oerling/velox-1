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

#include "velox/experimental/wave/exec/Operator.h"

namespace facebook::velox::wave {

Operator::Operator(CompileState& state, const TypePtr& type)
    : outputType_(type) {
  Operator::definesSubfields(
      CompileState & state,
      const TypePtr& type,
      const std::string& parantPath) {
    switch (type->kind()) {
      case TypeKind::ROW: {
        auto& row = type->as<TypeKind::ROW>();
        for (auto i = 0; i < type->size(); ++i) {
          auto& child = row.childAt(i);
          auto name = row->nameOf(i);
          auto field = state.toSubfield(name);
          Value value = state.toValue(field);
          defines_[value] = state.newOperand(child->type(), name);
        }
      }
        // Add cases for nested types.
      default: {
        return;
      }
    }
  }
}
} // namespace facebook::velox::wave
