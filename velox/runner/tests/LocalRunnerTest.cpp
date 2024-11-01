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

#include "velox/exec/tests/utils/DistributedPlanBuilder.h"
#include "velox/exec/tests/utils/LocalRunnerTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::runner;
using namespace facebook::velox::exec::test;

class LocalRunnerTest : public LocalRunnerTestBase {
 protected:
  static constexpr int32_t kNumFiles = 5;
  static constexpr int32_t kNumVectors = 5;
  static constexpr int32_t kRowsPerVector = 10000;
  static constexpr int32_t kNumRows = kNumFiles * kNumVectors * kRowsPerVector;

  void SetUp() override {
    LocalRunnerTestBase::SetUp();
    ensureDataset();
  }

  void ensureDataset() {
    if (files_) {
      return;
    }

    int32_t counter1 = 0;
    auto patch1 = [&](const RowVectorPtr& rows) {
      makeAscending(rows, counter1);
    };

    int32_t counter2 = 0;
    auto patch2 = [&](const RowVectorPtr& rows) {
      makeAscending(rows, counter2);
    };

    rowType_ = ROW({"c0"}, {BIGINT()});
    std::vector<TableSpec> specs = {
        TableSpec{
            .name = "T",
            .columns = rowType_,
            .rowsPerVector = kRowsPerVector,
            .numVectorsPerFile = kNumVectors,
            .numFiles = kNumFiles,
            .patch = patch1},
        TableSpec{
            .name = "U",
            .columns = rowType_,
            .rowsPerVector = kRowsPerVector,
            .numVectorsPerFile = kNumVectors,
            .numFiles = kNumFiles,
            .patch = patch2}};

    schema_ = makeTables(specs, files_);
    sourceFactory_ = std::make_shared<LocalSplitSourceFactory>(schema_, 2);
  }

  void TearDown() override {
    schema_.reset();
    files_.reset();
    sourceFactory_.reset();
    LocalRunnerTestBase::TearDown();
  }

  MultiFragmentPlanPtr makeScan() {
    MultiFragmentPlan::Options options = {
        .queryId = "test.", .numWorkers = 4, .numDrivers = 2};
    const int32_t width = 3;

    DistributedPlanBuilder rootBuilder(options, idGenerator_, pool_.get());
    rootBuilder.tableScan("T", rowType_);
    return std::make_shared<MultiFragmentPlan>(
        rootBuilder.fragments(), std::move(options));
  }

  MultiFragmentPlanPtr makeJoin(std::string project = "c0") {
    MultiFragmentPlan::Options options = {
        .queryId = "test.", .numWorkers = 4, .numDrivers = 2};
    const int32_t width = 3;

    DistributedPlanBuilder rootBuilder(options, idGenerator_, pool_.get());
    rootBuilder.tableScan("T", rowType_)
        .project({project})
        .shuffle({"c0"}, 3, false)
        .hashJoin(
            {"c0"},
            {"b0"},
            DistributedPlanBuilder(rootBuilder)
                .tableScan("U", rowType_)
                .project({"c0 as b0"})
                .shuffleResult({"b0"}, width, false),
            "",
            {"c0", "b0"})
        .shuffle({}, 1, false)
        .finalAggregation({}, {"count(1)"}, {{BIGINT()}});
    return std::make_shared<MultiFragmentPlan>(
        rootBuilder.fragments(), std::move(options));
  }

  void makeAscending(const RowVectorPtr& rows, int32_t& counter) {
    auto ints = rows->childAt(0)->as<FlatVector<int64_t>>();
    for (auto i = 0; i < ints->size(); ++i) {
      ints->set(i, counter + i);
    }
    counter += ints->size();
  }

  std::shared_ptr<core::PlanNodeIdGenerator> idGenerator_{
      std::make_shared<core::PlanNodeIdGenerator>()};
  // The below are declared static to be scoped to TestCase so as to reuse the
  // dataset between tests.

  inline static RowTypePtr rowType_;
  inline static std::shared_ptr<LocalSchema> schema_;
  inline static std::shared_ptr<TempDirectoryPath> files_;
  inline static std::shared_ptr<SplitSourceFactory> sourceFactory_;
};

TEST_F(LocalRunnerTest, count) {
  auto join = makeJoin();
  auto localRunner = std::make_shared<LocalRunner>(
      std::move(join), makeQueryCtx("q1"), sourceFactory_);
  auto results = readCursor(localRunner);
  auto stats = localRunner->stats();
  EXPECT_EQ(1, results.size());
  EXPECT_EQ(1, results[0]->size());
  EXPECT_EQ(
      kNumRows, results[0]->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0));
  results.clear();
  localRunner->waitForCompletion(5000);
}

TEST_F(LocalRunnerTest, count) {
  auto join = makeJoin("if (c0 = 111, c0 / 0, c0 + 1) as c0");
  auto localRunner = std::make_shared<LocalRunner>(
      std::move(join), makeQueryCtx("q1"), sourceFactory_);
  auto results = readCursor(localRunner);
  auto stats = localRunner->stats();
  EXPECT_EQ(1, results.size());
  EXPECT_EQ(1, results[0]->size());
  EXPECT_EQ(
      kNumRows, results[0]->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0));
  results.clear();
  localRunner->waitForCompletion(5000);
}
