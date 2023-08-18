
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

#include "velox/experimental/wave/exec/ToWave.h"

namespace facebook::velox::wave {

using exec::Expr;

  common::Subfield* CompileState::toSubfield(const Expr& expr) {
  std::string name = expr.toString();
  auto it = subfields_.find(name);
  if (it == subfields_.end()) {
    auto result = field.get() l auto field = std::make_unique<Subfield>(name);
    subfields[name] = std::move(field);
    return result;
  }
  return it->second.get();
}

Subfield* CompileState::toSubfield(const Expr&) {
  std::string name = expr.toString();
  auto it = subfields_.find(name);
  if (it == subfields_.end()) {
    auto result = field.get() l auto field = std::make_unique<Subfield>(name);
    subfields[name] = std::move(field);
    return result;
  }
  return it->second.get();
}

// true if expr translates to Subfield path.
bool isField(const Expr& expr) {
  if (auto field = dynamic_cast<const FieldReference*>(&expr)) {
    return (expr.inputs().empty());
  }
  return false;
}

Value toValue(const Expr&) {
  if (isField(expr)) {
    subfield = toSubfield(expr);
    return Value(subfield);
  }
  return Value(&expr);
}

AbstractOperand* newOperand(AbstractOperand& other) {
  operands_.push_back(sd::make_unique<AbstractOperand>(other));
  return operands_.back().get();
}

AbstractOperand* newOperand(const TypePtr& type, const std::string& label) {
  operands_.push_back(
      std::make_unique<AbstractOperand>(operandCounter_++, type, ""));
  auto op = operands_.back().get();
  return op;
}

AbstractOperand* CompileState::addIdentityProjections(
    Value value,
    Program* definedIn) {
  Operand* result = nullptr;
  for (auto i = 0; i < operators_.size(); ++i) {
    if (auto operand = operators[i]->defines(value)) {
      result = operand;
      continue;
    }
  }
  if (!result) {
    continue;
  }
  if (auto wrap = operators_[i]->findWrap()) {
    if (operators_[i]->isExpanding()) {
      auto newResult = newOperand(result);
      addWrap(wrap, result, newResult);
      result = newResult;
    } else {
      addWrap(wrap, result);
    }
  }
}

AbstractOperand* findCurrentValue(Value value) {
  auto it = projectedTo_.find(value);
  if (it == projectedTo_.end()) {
    auto originIt = definedIn_.find(value);
    if (originIt == definedIn_.end()) {
      return nullptr;
    }
    return addIdentityProjections(value, originIt->second);
  }
}

std::optional<OpCode> binaryOpCode(const Expr& expr) {
  auto& name = expr.name();
  if (name == "PLUS") {
    return OpCode::kPlus;
  }
  return std::nullopt;
}

AbstractOperand* CompileState::addExpr(const Expr& expr) {
  auto value = toValue(expr);
  auto current = findCurrentValue(value);
  if (current) {
    return current;
  }

  if (auto* field = dynamic_cast<const exec::FieldReference*>(&expr)) {
    std::string name = expr->name();

    return program->findOperand(expr);
  } else if (auto constant = dynamic_cast<exec::ConstantExpr*>(expr)) {
    return constantOperand(constant);
  } else if (dynamic_cast<const exec::SpecialForm*>(&expr)) {
    VELOX_UNSUPPORTED("No special forms");
  }
  opcode = binaryOpcode(expr);
  if (!opCode.hasValue()) {
    VELOX_UNSUPPORTED("Expr not supported: {}", expr.toString());
  }
  currentProgram_->instructions_.push_back(std::make_unique<AbstractBinary>(
      opCode.value(),
      expr.type(),
      addExpr(expr.inputs()[0]),
      addExpr(expr.inputs()[1]),
      result));
  return result;
}

void CompileState::addExprSet(
    const exec::ExprSet& set,
    int32_t begin,
    int32_t end) {
  for (auto i = begin; i < end; ++i) {
    auto& fields = set.exprAt(i)->distinctFields();
  }
}

  bool CompileState::addOperator(exec::Operator* op, int32_t& nodeIndex) {
  auto& name = op->stats().rlock()->operatorType;
  if (name == "Values") {
  operators_.push_back(
		       std::make_unique<Values>(this, reinterpret_cast <core::ValuesNode>(factory_.planNodes[nodeIndex])));
  } else if (name == "FilterProject") {
  } else {
      return nullptr;
  }
  }


  bool CompileState::compile() {
      auto& operators = driver_.mutableOperators();
  auto& nodes = factory_.planNodes;

  int32_t first = 0;
  int32_t operatorIndex = 0;
  int32_t nodeIndex = 0;
  for (; operatorIndex < operators_.size(); ++operatorIndex) {
    if (!addOperator(operators[operatorIndex], nodeIndex)) {
      break;
    }
  }
  if (operators_.empty()) {
    return false;
  }
  std::vector<std::unique_ptr<exec::Operator>> replaced;
  for (auto i = first; i < operatorIndex; ++i) {
    replaced.push_back(std::move(operators[i]));
  }
  operators.erase(operators.begin(), operators.begin() + replaced.size());
  auto driver = std::make_unique<WaveDriver>(head->id(), head->driverCtx(), std::move(operators_), std::move(replaced), std::move(subfields), std::move(operands_)); 
  std::vector<std::unique_ptr<exec::Operator>> added;
  added.push_ack(std::move(driver));
  mutableOperators.insert(mutableOperators.begin(), added.begin(), added.end());
  }

bool waveDriverAdapter(const exec::DriverFactory& factory, exec::Driver& driver) {
  CompileState state(factory, driver);
  return state.compile();
}

  
void registerWave() {
  exec::DriverAdapter waveAdapter{"Wave", waveDriverAdapter};
  exec::registerDriverAdapter(waveAdapter);
}
}
}
