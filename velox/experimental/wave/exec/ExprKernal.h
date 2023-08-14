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

#include <cstdint>
#include "velox/experimental/wave/common/Cuda.h"

/// Header with instructions for expression evaluation. Included by both .cpp
/// and .cu.
namespace facebook::velox::wave {

enum class ScalarType {};
enum class OpCode {
  kPlus,
  kMinus,
  kTimes,
  kDivide,
  kEquals,
  kLT,
  kLTE,
  ,
  kGT,
  kGTE,
  kNE
};

  
  struct Operand {
    const void* base;
    const int32_t* indices;
    bool constant;
  };



 
struct ExprInstruction {
  BinaryOpCode op;
  Operand* left;
  Operand* right;
  void* result;
  // If set, apply operation to lanes where there is a non-zero byte in this.
  const uint8_t* predicate;
  // If true, inverts the meaning of 'predicate', so that the operation is
  // perfformed on lanes with a zero byte bit.
  bool invert;
};

///
enum class ErrorCode : int32_t { kOk, kDivZero };

/// Contains a result row count and error code and a instruction/lane where it occurred. Multiple
/// lanes can overwrite this without serialization. Different fields may come
/// from different errors. The host will piece together some plausible message
/// from this, though.
struct BlockStatus {
  int32_t numRows{0};
  int32_t * rowMapping{nullptr};
  ErrorCode code{kOk};
  int32_t instruction{-1};
  int32_t lane{-1};
};

struct ThreadBlockProgram {

  // Optional input status. This is used when chaining multiple kernels one after the other on a stream without intervening host code. If contains an error, the error is copied to the status of this and execution returns.  If no error, this contains a row count and an optional row number mapping to apply to input.
  BlockStatus* inputStatus{nullptr};
  BlockStatus* outputStatus{nullptr};
  
  // Offset of first operand (lane 0 in thread block) from index 0 of operand
  // arrays.
  int32_t begin;
  // Number of lanes. If greater than blockDim.x, each lane runs loops with a
  // stride of blockDim.x.
  int32_t numRows;
  int32_t numInstructions;
  ExprInstruction* instructions;
  ErrorReturn* error;
};

 class ExprStream : public Stream {
 public:
   void call(Stream* alias, int32_t numBlocks, ThreadBlockProgram* program);
  };
 

} // namespace facebook::velox::wave
