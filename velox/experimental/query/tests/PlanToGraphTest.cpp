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

#include "velox/experimental/query/PlanToGraph.h"
#include <gtest/gtest.h>
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/experimental/query/tests/Tpch.h"

using namespace facebook::velox;
using namespace facebook::velox::query;

class PlanToGraphTest : public testing::Test {
 protected:
  void SetUp() override {
    allocator_ = std::make_unique<HashStringAllocator>(
        memory::MappedMemory::getInstance());
    context_ = std::make_unique<QueryGraphContext>(*allocator_);
    queryCtx() = context_.get();
    builder_ = std::make_unique<exec::test::TpchQueryBuilder>(
        dwio::common::FileFormat::PARQUET);
  }

  std::unique_ptr<HashStringAllocator> allocator_;

  std::unique_ptr<QueryGraphContext> context_;
  std::unique_ptr<exec::test::TpchQueryBuilder> builder_;
};

TEST_F(PlanToGraphTest, q3) {
  auto q3 = builder_->getQueryPlan(3);
  auto schema = tpchSchema(100, false, true, false);
  Optimization opt(*q3.plan, *schema);
  auto result = opt.bestPlan();
  LOG(INFO) << result->toString(true, true);
}
