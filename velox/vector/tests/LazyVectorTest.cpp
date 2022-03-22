
#include <gtest/gtest.h>

#include "velox/vector/test/VectorTestBase.h"

class LazyVectorTest : public testing::Test, public VectorTestBase {};

TEST_F(VectorTest, lazyInDict) {
  // We have dictionaries over LazyVector. We load for some indices in
  // the top dictionary. The intermediate dictionaries refer to
  // non-loaded items in the base of the LazyVector, including indices
  // past its end. We check that we end up with one level of
  // dictionary and no dictionaries that are invalid by through
  // referring to uninitialized/nonexistent positions.
  auto base = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  auto lazy = std::make_shared<LazyVector>(
      execCtx_->pool(),
      INTEGER(),
      1000,
      std::make_unique<test::SimpleVectorLoader>(
          [base](auto /*size*/) { return base; }));
  auto row = makeRowVector({BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(100, [](auto row) { return row; }),
      100,

      BaseVector::wrapInDictionary(
          nullptr,
          makeIndices(1000, [](auto row) { return row; }),
          1000,
          lazy))});

  // We expect a single level of dictionary.
  auto result = evaluate("c0", row);
  EXPECT_EQ(result->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(result->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  assertEqualVectors(result, base);
}
}
