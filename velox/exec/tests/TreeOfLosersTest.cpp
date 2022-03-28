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
#include "velox/exec/tests/utils/MergeTestUtils.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class TreeOfLosersTest : public testing::Test, public MergeTestUtils {
 protected:
  void SetUp() override {
    seed(1);
  }
};

TEST_F(TreeOfLosersTest, merge) {
  constexpr int32_t kNumValues = 5000000;
  constexpr int32_t kNumRuns = 35;

  TestData testData = makeTestData(kNumValues, kNumRuns);
  test<TreeOfLosers<TestingStream>>(testData, true);
  test<MergeArray<TestingStream>>(testData, true);
}
