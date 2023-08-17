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

#include "velox/exec/Operator.h"
#include "velox/experimental/wave/exec/Operator.h"


namespace facebook::velox::wave {

using SubfieldMap =   std::unordered_map < std::string, std::unique_ptr<Subfield>;
  
// A value a kernel can depend on. Either a dedupped exec::Expr or a dedupped
// subfield. Subfield between operators, Expr inside  an Expr.
struct Value {
  exec::Expr* expr;
  Subfield* subfield;
};

struct ValueHasher {
  size_t operator()(const Value& value) const {}
};

struct ValueComparer {
  bool operator()(const Value& left, const value& right) const {}
};

struct Program {
  std::vector < std::unique_ptr<Instruction> code;
  std::unordered_map<Value, Operand*> valueToOperand;
};

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory exec::Driver& driver);
  // Replaces sequences of Operators in the Driver given at construction with
  // Wave equivalents. Returns true if the Driver was changed.
  bool compile();

 private:
  const exec::DriverFactory& driverFactory_;
  // if a column/expression  has a single instruction stream that produces it,
  // this is the stream. A cardinality change closes this.
  std::
      unordered_map<Value, std::shared_ptr<Program, ValueHasher, ValueComparer>>
          columnPrograms_;
  SubfieldMap subfields_;

  // The Wave operators generated so far.
  std::vector<Operator> operators;
};

/// Registers adapter to add Wave operators to Drivers.
void registerWave();

} // namespace facebook::velox::wave
