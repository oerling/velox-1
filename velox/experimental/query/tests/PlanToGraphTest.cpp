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

#include "velox/experimental/query/Plan.h"
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/experimental/query/tests/Tpch.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

DEFINE_string(
    data_path,
    "/home/oerling/tpch100pqsnlinks",
    "Path to directory for TPC-H files");

using namespace facebook::velox;
using namespace facebook::verax;

class PlanToGraphTest : public testing::Test {
 protected:
  void SetUp() override {
    allocator_ = std::make_unique<HashStringAllocator>(
        memory::MappedMemory::getInstance());
    context_ = std::make_unique<QueryGraphContext>(*allocator_);
    queryCtx() = context_.get();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    parquet::registerParquetReaderFactory(parquet::ParquetReaderType::NATIVE);
    builder_ = std::make_unique<exec::test::TpchQueryBuilder>(
        dwio::common::FileFormat::PARQUET);
    builder_->initialize(FLAGS_data_path);
    makeCheats();
  }

  void makeCheats() {
    baseSelectivities()
        ["table: lineitem, range filters: [(l_shipdate, BigintRange: [9205, 9223372036854775807] no nulls)]"] =
            0.5;
    baseSelectivities()
        ["table: orders, range filters: [(o_orderdate, BigintRange: [-9223372036854775808, 9203] no nulls)]"] =
            0.5;
    baseSelectivities()
        ["table: customer, range filters: [(c_mktsegment, Filter(BytesValues, deterministic, null not allowed))]"] =
            0.2;
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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
