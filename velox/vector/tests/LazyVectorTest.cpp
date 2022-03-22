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

#include <gtest/gtest.h>

#include "velox/vector/tests/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class LazyVectorTest : public testing::Test, public VectorTestBase {};

TEST_F(LazyVectorTest, lazyInDoubleDict) {
  // We have dictionaries over LazyVector. We load for some indices in
  // the top dictionary. The intermediate dictionaries refer to
  // non-loaded items in the base of the LazyVector, including indices
  // past its end. We check that we end up with one level of
  // dictionary and no dictionaries that are invalid by through
  // referring to uninitialized/nonexistent positions.
  static constexpr int32_t kInnerSize = 100;
  static constexpr int32_t kOuterSize = 1000;
  auto base = makeFlatVector<int32_t>(kInnerSize, [](auto row) { return row; });
  vector_size_t loadEnd = 0;
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      kInnerSize,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        loadEnd = rows.back() + 1;
        return base;
      }));
  auto wrapped = BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(kInnerSize, [](auto row) { return row; }),
      kInnerSize,
      BaseVector::wrapInDictionary(
          nullptr,
          makeIndices(kOuterSize, [](auto row) { return row; }),
          kOuterSize,
          lazy));

  // We expect a single level of dictionary and rows loaded for kInnerSize first
  // elements of 'lazy'.
  SelectivityVector rows(kInnerSize);
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(wrapped->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  EXPECT_EQ(kInnerSize, loadEnd);
  assertEqualVectors(wrapped, base);
}
