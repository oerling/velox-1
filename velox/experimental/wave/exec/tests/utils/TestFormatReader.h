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
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"
#include "velox/type/Subfield.h"

namespace facebook::velox::wave::test {

class TestFormatData : public wave::FormatData {
 public:
  TestFormatData(OperandId operand, const test::Column* column)
      : operand_(operand), column_(column) {}

  /// Adds the next read of the column. If the column is a filter depending on
  /// another filter, the previous filter is given on the first call. Returns an
  /// OR of flags describing the action. See kStaged, kQueued, kAllQueued.
  virtual int32_t startRead(
      int32_t offset,
      RowSet rows,
      FormatData* previousFilter,
      SplitStaging& staging,
      DecodePrograms& program) = 0;

 private:
  const OperandId operand_;
  const test::Column* column_;
  bool staged_{false};
  bool queued_{false};
};

class TestFormatParams : public wave::FormatParams {
 public:
  TestFormatParams(
      memory::MemoryPool& pool,
      dwio::common::ColumnReaderStatistics& stats,
      const test::Stripe* stripe)
      : FormatParams(pool, stats), stripe_(stripe) {}

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec,
      OperandId operand) override;

 private:
  const test::Stripe* stripe_;
};

class TestFormatReader {
 public:
  static std::unique_ptr<ColumnReader> build(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      TestFormatParams& params,
      common::ScanSpec& scanSpec,
      std::vector<std::unique_ptr<common::Subfield::PathElement>>& path,
      const DefinesMap& defines,
      bool isRoot = false);
};

} // namespace facebook::velox::wave::test
