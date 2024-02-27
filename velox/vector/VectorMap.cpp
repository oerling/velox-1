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

#include "velox/vector/VectorMap.h"

namespace facebook::velox {

vector_size_t VectorMap::addOne(
    const BaseVector& topVector,
    vector_size_t topIndex) {
  const BaseVector* vector = &index;
  vector_size_t index = topIndex;
  if (topVector.encoding() != VectorEncoding::Simple::FLAT) {
    vector = topVector.wrappedVector();
    index = topVector.wrappedIndex(topIndex);
  }
  if (LIKELY(isString_)) {
    StringView string;
    bool isNull = vector->isNullAt(index);
    if (UNLIKELY(isNull)) {
      if (nullIndex_ != kNoNullIndex) {
        return nullIndex_;
      }
    } else {
      if (LIKELY(vector->encoding() == VectorEncoding::Simple::FLAT)) {
        string = vector->asUnchecked<FlatVector<StringView>>()->valueAt(index);
      } else {
        string = vector->asUnchecked<ConstantVector<StringView>>()->valueAt(0);
      }
      auto it = distinctStrings_.find(string);
      if (it != distinctStrings_.end()) {
        return it->second;
      }
    }
  } else {
    auto it = distinctSet_.find(VectorValueSetEntry{vector, index});
    if (it != distinctSet_.end()) {
      return it->index;
    }
  }
  int32_t newIndex;
  if (alphabet_ == nullptr) {
    alphabet_ = BaseVector::create(vector->type(), 1, streamArena_->pool());
    newIndex = 0;
    distinctsSizes_.resize(1);
  } else {
    newIndex = alphabet_->size();
    alphabet_->resize(newIndex + 1);
    distinctsSizes_.resize(newIndex + 1);
  }
  alphabet_->copy(vector, newIndex, index, 1);
  const bool isNull = vector->isNullAt(index);
  if (isNull) {
    distinctsSizes_[newIndex] = 0;
    nullIndex_ = newIndex;
  } else {
    Scratch scratch;
    ScratchPtr<vector_size_t, 1> indicesHolder(scratch);
    ScratchPtr<vector_size_t*, 1> sizesHolder(scratch);
    auto sizeIndices = indicesHolder.get(1);
    sizeIndices[0] = newIndex;
    auto sizes = sizesHolder.get(1);
    distinctsSizes_[newIndex] = 0;
    sizes[0] = &distinctsSizes_[newIndex];
    estimateSerializedSizeInt(
        alphabet_.get(),
        folly::Range<const vector_size_t*>(sizeIndices, 1),
        sizes,
        scratch);
  }
  if (isString_) {
    if (!isNull) {
      distinctStrings_[alphabet_->asUnchecked<FlatVector<StringView>>()
                           ->valueAt(newIndex)] = newIndex;
    }
  } else {
    distinctSet_.insert(VectorValueSetEntry{alphabet_.get(), newIndex});
  }
  return newIndex;
}

} // namespace facebook::velox
