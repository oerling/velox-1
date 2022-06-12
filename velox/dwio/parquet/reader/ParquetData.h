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
#include "dwio/dwrf/common/BufferedInput.h"
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
  ParquetParams(memory::MemoryPool& pool, FileMetaData& metaData)
      : FormatParams(pool), metaData_(metaData) {}
  std::unique_ptr<dwio::common::FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type) override;

 private:
  FileMetaData metaData_;
};

// Format-specific data created for each leaf column of a Parquet rowgroup.
class ParquetData : public dwio::common::FormatData {
 public:
  ParquetData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      std::vector<RowGroup>& rowGroups,
      memory::MemoryPool& pool)
    : pool_(pool),
      type_(std::static_pointer_cast<ParquetTypeWithId>(type)),
        rowGroups_(rowGroups),
        maxDefine_(type_->maxDefine_),
        maxRepeat_(type_->maxRepeat_),
        rowsInRowGroup_(-1) {}

  void enqueueRowGroup(uint32_t index, dwrf::BufferedInput& input);

  void seekToRowGroup(uint32_t index);

  bool filterMatches(uint32_t index, common::Filter& filter) {
    return true;
  }

  std::vector<uint32_t> filterRowGroups(const common::Filter& filter) const {
    std::vector<uint32_t> stridesToSkip;
    return stridesToSkip;
  }

  template <typename Visitor>
  void readWithVisitor(Visitor visitor) {
    decoder_->readWithVisitor(visitor);
  }

 protected:
  memory::MemoryPool& pool_;
  std::shared_ptr<const ParquetTypeWithId> type_;
  std::vector<RowGroup>& rowGroups_;
  // Streams for this column in each of 'rowGroups_'. Will be created on or
  // ahead of first use, not at construction.
  std::vector<std::unique_ptr<dwrf::SeekableInputStream>> streams_;

  int32_t rowGroupIndex_{0};

  const uint32_t maxDefine_;
  const uint32_t maxRepeat_;

  int64_t rowsInRowGroup_;
  std::unique_ptr<PageDecoder> decoder_;
};

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type) {
  return std::make_unique<ParquetData>(
      std::static_pointer_cast<ParquetTypeWithId>(type),
      metaData_.row_groups,
      pool());
}

} // namespace facebook::velox::parquet
