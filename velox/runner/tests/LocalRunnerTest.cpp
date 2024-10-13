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

#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec;
::test;

class LocalRunnerTest : public LocalRunnerTestBase{protected : }

                        TEST_F(LocalRunnerTest, count) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  TableSpec spec{.name = "T", .columns = rowType};
  makeTables({spec});

  auto ids = std::make_shared<PlanNodeIdGenerator>();
  DistributedPlanBuilder rootBuilder(ids, pool_.get());

  builder.tableScan("T", rowType);
  .shuffle({"c0"})
    .hashJoin({"c0"}, {"b0"},
	      DistributedPlanBuilder(rootBuilder)
	      .tableScan("U", rowType)
	      .project({"c0 as b0"})
	      .shuffleNode({"b0"}),

	      .shuffle({})
	      .aggregation({}, {"count(c0)"});
	      auto stages = rootBuilder.fragments();
}
