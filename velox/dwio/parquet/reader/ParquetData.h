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
#include <thrift/protocol/TCompactProtocol.h>
#include "Decoder.h"
#include "ParquetThriftTypes.h"
#include "ThriftTransport.h"
#include "dwio/common/BufferedInput.h"
#include "velox/common/base/BitSet.h"
#include "velox/common/base/RawVector.h"
#include "velox/dwio/dwrf/reader/SelectiveStructColumnReader.h"
#include "velox/dwio/parquet/reader/PageDecoder.h"

namespace facebook::velox::parquet {

class ParquetTypeWithId : public dwio::common::TypeWithId {
 public:
  ParquetTypeWithId(
      TypePtr type,
      const std::vector<std::shared_ptr<const TypeWithId>>&& children,
      uint32_t id,
      uint32_t maxId,
      uint32_t column,
      std::string name,
      uint32_t maxRepeat,
      uint32_t maxDefine)
      : TypeWithId(type, std::move(children), id, maxId, column),
        name_(name),
        maxRepeat_(maxRepeat),
        maxDefine_(maxDefine) {}

  std::string name_;
  uint32_t maxRepeat_;
  uint32_t maxDefine_;
};

class ParquetParams : public dwio::common::FormatParams {
 public:
  ParquetParams(memory::MemoryPool& pool, const FileMetaData& metaData)
      : FormatParams(pool), metaData_(metaData) {}
  std::unique_ptr<dwio::common::FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type) override;

 private:
  const FileMetaData& metaData_;
};

// Format-specific data created for each leaf column of a Parquet rowgroup.
class ParquetData : public dwio::common::FormatData {
 public:
  ParquetData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const std::vector<RowGroup>& rowGroups,
      memory::MemoryPool& pool)
      : pool_(pool),
        type_(std::static_pointer_cast<const ParquetTypeWithId>(type)),
        rowGroups_(rowGroups),
        maxDefine_(type_->maxDefine_),
        maxRepeat_(type_->maxRepeat_),
        rowsInRowGroup_(-1) {}

  // true if no nulls in the current row group
  bool isNonNull() {
    VELOX_CHECK_NE(kNoRowGroup, rowGroupIndex_);
    if (maxDefine_ == 0) {
      return true;
    }
    auto& columnChunk = rowGroups_[rowGroupIndex_].columns[type_->column];
    VELOX_CHECK(columnChunk.__isset.meta_data);
    auto& metaData = columnChunk.meta_data;
    if (metaData.__isset.statistics) {
      return false;
    }
    auto& stats = metaData.statistics;
    return (stats.__isset.null_count && stats.null_count == 0);
  }

  // Prepares to read data for 'index'th row group.
  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  // Positions 'this' at 'index'th row group. enqueueRowGroup must be called
  // first.
  void seekToRowGroup(uint32_t index);

  bool filterMatches(const RowGroup& rowGroup, common::Filter& filter);

  std::vector<uint32_t> filterRowGroups(
      uint64_t rowsPerRowGroup,
      const dwio::common::StatsWriterInfo& writerInfo) override {
    std::vector<uint32_t> stridesToSkip;
    return stridesToSkip;
  }

  void skip(int32_t numRows) {
    decoder_->skip(numRows);
  }

  template <typename Visitor>
  void readWithVisitor(Visitor visitor) {
    decoder_->readWithVisitor(visitor);
  }

 protected:
  static constexpr int32_t kNoRowGroup = -1;

  memory::MemoryPool& pool_;
  std::shared_ptr<const ParquetTypeWithId> type_;
  const std::vector<RowGroup>& rowGroups_;
  // Streams for this column in each of 'rowGroups_'. Will be created on or
  // ahead of first use, not at construction.
  std::vector<std::unique_ptr<dwio::common::SeekableInputStream>> streams_;

  int32_t rowGroupIndex_{kNoRowGroup};

  const uint32_t maxDefine_;
  const uint32_t maxRepeat_;
  int64_t rowsInRowGroup_;
  std::unique_ptr<PageDecoder> decoder_;
};

} // namespace facebook::velox::parquet
