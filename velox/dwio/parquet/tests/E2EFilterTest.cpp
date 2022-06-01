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

#include "velox/dwio/dwrf/test/E2EFilterTestBase.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/dwio/parquet/reader/NativeParquetReader.h"

using namespace facebook::velox::dwio::dwrf;
using namespace facebook::velox::dwrf;
using namespace facebook::velox::parquet;
using namespace facebook::velox;
using namespace facebook::velox::common;

using dwio::common::MemoryInputStream;
using dwio::common::MemorySink;

class E2EFilterTest : public E2EFilterTestBase {
 protected:
  void writeToMemory(
      const TypePtr& type,
      const std::vector<RowVectorPtr>& batches,
      bool forRowGroupSkip) override {
    auto sink = std::make_unique<MemorySink>(*pool_, 200 * 1024 * 1024);
    sinkPtr_ = sink.get();

    writer_ = std::make_unique<facebook::velox::parquet::Writer>(std::move(sink), *pool_, 10000);
    for (auto& batch : batches) {
      writer_->write(batch);
    }
    writer_->close();
  }

  std::unique_ptr<dwio::common::Reader> makeReader(
      const dwio::common::ReaderOptions& opts,
      std::unique_ptr<dwio::common::InputStream> input) override {
    return std::make_unique<NativeParquetReader>(std::move(input), opts);
  }

  std::unique_ptr<facebook::velox::parquet::Writer> writer_;
};

TEST_F(E2EFilterTest, integerDirect) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint,"
      "long_null:bigint",
      [&]() { makeAllNulls("long_null"); },
      true,
      {"short_val", "int_val", "long_val"},
      20,
      true,
      true);
}
