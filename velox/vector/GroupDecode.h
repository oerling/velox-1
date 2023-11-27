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

#include "velox/common/base/RawVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

/// Set of vectors sharing a wrapper.
struct WrapGroup {
  /// The indices that wrap columns in this group. nullptr if the columns are
  /// not wrapped.
  vector_size_t* indices;

  /// The indices of columns in this group.
  raw_vector<int32_t> columnIndicess;

  /// The data wrapped by 'indices'. 1:1 with 'columnIndices'.
  raw_vector<BaseVector*> valueVectors;

  /// If any of 'valueVectors' is a wrapper, there is one child group per
  /// distinct wrapper.
  std::vector<WrapGroup> children;
};

///  Mechanism for peeling wrappers off multiple vectors potentially sharing
///  wrappers. This is useful for operators like repartitioning that process all
///  columns of of input and where different inputs come from different places
///  in a pipeline.
struct GroupDecode {
  /// Divides the columns in 'row' into groups where columns with the same
  /// wrappers go together. To process the columns, take the columns in each
  /// WrapGroup and process them together.
  void decode(const RowVector& row);

  WrapGroup decoded;
};

} // namespace facebook::velox
