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

#include "velox/experimental/wave/vector/WaveVector.h"

namespace facebook::velox::wave {

void WaveVector::resize(vector_size_t size, bool nullable) {
  if (size > size_) {
    int64_t bytes = type_->cppSizeInBytes() * size;
    if (!values_ || bytes > values_->capacity()) {
      values_ = arena_->allocateBytes(bytes);
    }
    if (nullable) {
      if (!nulls_ || nulls_->capacity() < size) {
        nulls_ = arena_->allocateBytes(size);
      }
    } else {
      nulls_.reset();
    }
    size_ = size;
  }
}

void WaveVector::toOperand(Operand* operand) const {
  if (encoding_ == VectorEncoding::Simple::CONSTANT) {
    operand->indexMask = 0;
    if (nulls_) {
      operand->nulls = nulls_->as<uint8_t>();
    } else {
      operand->nulls = nullptr;
    }
    operand->base = values_->as<uint64_t>();
    return;
  }
  if (encoding_ == VectorEncoding::Simple::FLAT) {
    operand->indexMask = ~0;
    operand->base = values_->as<int64_t>();
    operand->indices = nullptr;
  } else {
    VELOX_UNSUPPORTED();
  }
}

} // namespace facebook::velox::wave
