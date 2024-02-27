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

#include "velox/vector/BaseVector.h"

namespace facebook::velox {

struct VectorValueSetEntry {
  const BaseVector* vector;
  vector_size_t index;
};

struct VectorValueSetHasher {
  size_t operator()(const VectorValueSetEntry& entry) const {
    return entry.vector->hashValueAt(entry.index);
  }
};

struct VectorValueSetComparer {
  bool operator()(
      const VectorValueSetEntry& left,
      const VectorValueSetEntry& right) const {
    return left.vector->equalValueAt(right.vector, left.index, right.index);
  }
};

using VectorValueSet = folly::F14FastSet<
    VectorValueSetEntry,
    VectorValueSetHasher,
    VectorValueSetComparer>;

/// A map translating values in a vector to positions in the mapped vector.
class VectorMap {
 public:
  VectorMap();

  explicit VectorMap(const BaseVector& alphabet);

  static std::unique_ptr<VectorMap> create(const TypePtr& type);

  /// Assigns a zero-based id to each distinct value in 'vector' at positions
  /// 'rows'. The ids are returned in indices. If new ids were assigned, the row
  /// where the value first occurred is written to 'newIds'. The number of
  /// values in newIds is returned. 'newIds' mustr have space for up to
  /// 'rows.size()' entries. Stops assigning ids if maxId is exceeded.
  std::pair<int32_t, bool> maybeAdd(
      BaseVector& vector,
      folly::Range<const vector_size_t*> rows,
      int32_t maxDistincts,
      vector_size_t* ids,
      vector_size_t* newIds);

  std::pair<vector_size_t, bool> addOne(
      const BaseVector& vector,
      vector_size_t row);

 private:
  // Vector containing all the distinct values.
  VectorPtr alphabet_;
  // Map from value in 'alphabet_' to the index in 'alphabet_'.
  VectorValueSet distinctSet_;

  // Map from string value in 'alphabet_' to index in 'alphabet_'. Used only if
  // 'alphabet_' is a FlatVector<StringView>.
  folly::F14FastSet<std::pair<StringView, int32_t>> distinctStrings_;

  // True if  using 'distinctStrings_'
  bool isString_;
};

} // namespace facebook::velox
