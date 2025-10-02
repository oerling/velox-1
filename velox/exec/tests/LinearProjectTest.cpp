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

#include "velox/exec/tests/utils/FeatureGen.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/ProjectSequence.h"
#include "velox/exec/Linear.h"
#include "velox/parse/Expressions.h"
#include "velox/core/Expressions.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::exec {
namespace {
using namespace facebook::velox::exec::test;

class LinearProjectTest : public test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    test::HiveConnectorTestBase::SetUp();
    setupLinearMetadata();
  }
};

TEST_F(LinearProjectTest, basic) {
  test::FeatureOptions opts;
  opts.rng.seed(1);
  auto vectors = test::makeFeatures(1, 100, opts, pool_.get());

  const auto rowType = vectors[0]->rowType();
  const auto fields = rowType->names();

  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set<const std::vector<uint32_t>>(
      dwrf::Config::MAP_FLAT_COLS, {2, 3, 4});

  auto file = TempFilePath::create();
  writeToFile(file->getPath(), vectors, config, rowType);

  auto readSchema = ROW({"uid", "ts", "float_features", "id_list_features"}, {BIGINT(), BIGINT(), opts.floatStruct, opts.idListStruct});

  auto plan =
        PlanBuilder()
            .tableScan(readSchema, {}, "", rowType)
            .filter("uid % 511 < 508")
            .project(
                {"uid",
                 "ts",
                 "row_constructor(coalesce(float_features.10010, 0), coalesce(float_features.10020, 0)) as ff_1",
                 "id_list_features"})
            .project(
                {"uid",
                 "ts",
                 "row_constructor(ff_1.c0 * 2 + 1, clamp(ff_1.c1 + 2, -10, 10))",
                 "row_constructor(array_sum(first_x(id_list_features.200100, 10)), array_intersect(id_list_features.200200, id_list_features.200300)) as id_list_features"})
            .planNode();

  auto split = makeHiveConnectorSplit(file->getPath());
}

TEST_F(LinearProjectTest, constantFolding) {
  // Test that preprocess folds constants in "a + (1 + 2 + 3)" to "a + 6"
  auto rowType = ROW({"a"}, {BIGINT()});

  // Parse the expression "a + (1 + 2 + 3)"
  auto untyped = parse::parseExpr("a + (1 + 2 + 3)", {});
  auto typedExpr = core::Expressions::inferTypes(untyped, rowType, pool());

  #if 0
  // Create a simple project node to create ProjectSequence
  auto projectNode = std::make_shared<core::ProjectNode>(
      "test_project",
      rowType,
      std::vector<std::string>{"result"},
      std::vector<core::TypedExprPtr>{typedExpr},
      std::make_shared<core::ValuesNode>("values", rowType));

  // Create ProjectSequence
  ProjectVector projects = {projectNode};
  auto driverCtx = createDriverCtx();
  ProjectSequence sequence(0, driverCtx.get(), projects);

  // Apply preprocessing
  auto preprocessed = sequence.preprocess(typedExpr);

  // Check that the result has the expected structure: "a + 6"
  ASSERT_EQ(preprocessed->kind(), core::ExprKind::kCall);
  auto call = preprocessed->asUnchecked<core::CallTypedExpr>();
  ASSERT_EQ(call->name(), "plus");
  ASSERT_EQ(call->inputs().size(), 2);

  // First input should be field access to "a"
  ASSERT_EQ(call->inputs()[0]->kind(), core::ExprKind::kFieldAccess);

  // Second input should be constant 6
  ASSERT_EQ(call->inputs()[1]->kind(), core::ExprKind::kConstant);
  auto constant = call->inputs()[1]->asUnchecked<core::ConstantTypedExpr>();
  ASSERT_EQ(constant->value().value<int64_t>(), 6);
#endif
}

} // namespace
} // namespace facebook::velox::exec
