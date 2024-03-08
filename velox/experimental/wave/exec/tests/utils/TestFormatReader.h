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

#include "velox/experimental/wave/dwio/ColumnReader.h"

namespace facebook::velox::wave {
class TestFormatParams {
 public:
  TestFormatParams(const test::Stripe* stripe) : stripe_(stripe) {}

  const test::Stripe* stripe_;
};

class TestFormatData : public wave::FormatData {
 public:
  TestFormatData(const test::Column* column) : column_(column) {}

 private:
  const test::Column* nulls_{nullptr};
  const Column* column_{nullptr};
};

class TestFormatParams : public wave::FormatParams {
 public:
  TestFormatDataParams(
      memory::MemoryPool& pool,
      ColumnReaderStatistics& stats,
      const test::Stripe* stripe)
      : FormatParams(pool, stats), stripe_(stripe) {}

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec) override;

  const test::Stripe* stripe_;
};

class TestFormatReader {
  static std::unique_ptr<ColumnReader> build(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec,
      bool isRoot = false);
};

} // namespace facebook::velox::wave
