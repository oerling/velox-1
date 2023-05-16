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
#include <folly/Unit.h>
#include <folly/init/Init.h>
#include <velox/exec/Driver.h>
#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

DECLARE_bool(fuzz_plans);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

using namespace facebook::velox::common::testutil;
using facebook::velox::test::BatchMaker;


class PlanFuzzerTest : public OperatorTestBase {
 protected:

};


TEST_F(PlanFuzzerTest, basic) {
  // Make a set of batches and see that they come through fuzzing.
  FLAGS_fuzz_plans = true;
}

