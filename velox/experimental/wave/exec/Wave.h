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


#include "velox/experimental/wave/exec/Wave.h"

#include "velox/experimental/wave/exec/ExprKernel.h"

namespace facebook::velox::wave {

// A value a kernel can depend on. Either a dedupped exec::Expr or a dedupped
// subfield. Subfield between operators, Expr inside  an Expr.
struct Value {
  Value(exec::Expr* expr) : expr(expr), subfield(nullptr) {}

  Value(Subfield* subfield) : expr(nullptr), subfield(subfield) {}

  const exec::Expr* const expr;
  const Subfield* const subfield;
};

struct ValueHasher {
  size_t operator()(const Value& value) const {}
};

struct ValueComparer {
  bool operator()(const Value& left, const value& right) const {}
};

class Wave;

class Program {
  // places the code and related structures in 'arena'.
  void prepareForDevice(GpuArena& arena);

  void prepareBuffers(GpuArena& arena, int32_t numRows, Wave& wave);

  const std::vector<Operand*> dependsOn() const {
    return dependsOn_;
  }
  Operand* findOperand(const Value& value){};

  std::vector<Value> dependsOn;
  folly::F14FastMap<Value, Operand*> produces;

  std::vector<std::unique_ptr<AbstractInstruction>> code;
};

class Wave {
 public:
  Wave(GpuArena& arena);

  /// Adds program and reserves space for maxRows of output.
  void addProgram(Program*, int32_t maxRows);

  // Returns Event for syncing with the arrival of 'this'.
  Event* event() {}

  void start(Stream* stream);

 private:
  GpuArena& arena_;
  Stream* stream_;

  // Event recorded on 'stream_' right after kernel launch.
  std::unique_ptr<Event> event_;

  // At start, errors are clear and row counts are input row counts. On return,
  // errors are set and the output row count of each block may be set if this
  // has cardinality change.
  BlockStatus* statuses;
  ThreadBlockProgram** programs_;
};

} // namespace facebook::velox::wave
