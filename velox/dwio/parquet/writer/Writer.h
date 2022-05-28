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
#pragma once

#include <parquet/api/writer.h>

namespace facebook::velox::parquet {
class Writer {
 public:
  Writer(
	 dwio::common::DataSink* FOLLY_NONNULL sink,
      int32_t rowsInRowGroup,
      int64_t bytesInRowGroup)
    : rowsInRowGroup_(rowsInRowGroup),
    bytesInRowGroup_(bytesInRowGroup),
    sink_(sink) {}

      void append(const RowVectorPtr& data);
  void close();

 private:
  const int32_t rowsInRowGroup_;
  const int32_t bytesInRowGroup_{0};
  int32_t rowsInCurrentGroup_{0};
  dwio::common::DataSink* sink_;
  std::shared_ptr<arrow::io::OutputStream> stream_;
  std::shared_ptr<::parquet::schema::GroupNode> root_;
  std::vector<::parquet::ColumnWriter*> columnWriters_;
  ::parquet::RowGroupWriter* rowGroupWriter_{nullptr};
  std::unique_ptr<::parquet::ParquetFileWriter> fileWriter_;
};

} // namespace facebook::velox::parquet
