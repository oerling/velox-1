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

//
// Created by Ying Su on 2/14/22.
//

#pragma once

#include <thrift/protocol/TCompactProtocol.h>
#include "Decoder.h"
#include "ParquetThriftTypes.h"
#include "ThriftTransport.h"
#include "dwio/dwrf/common/BufferedInput.h"
#include "velox/common/base/BitSet.h"
#include "velox/common/base/RawVector.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/dwrf/common/DirectDecoder.h"
#include "velox/dwio/dwrf/reader/ColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveColumnReader.h"

namespace facebook::velox::parquet {

//-----------------------ParquetColumnInfo-----------------------------

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


  class ParquetParams : public dwrf::FormapParams {
 public:
  ParquetParams(memory::MemoryPool& pool, FileMetaData& metaData)
      : FormatParams(pool), metaData_(metaData) {}
  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<TypeWithId>& type);

 private:
  FileMetaData metaData_;
};

class Dictionary {
 public:
  Dictionary(const void* dict, uint32_t size) : dict_(dict), size_(size) {}

 private:
  const void* dict_;
  uint32_t size_;
};

class ParquetData {
 public:
  ParquetData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      std::vector<RowGroup* FOLLY_NULLABLE> rowGroups)
    : type_(static_pointer_cast<ParquetTypeWithId>(type)),
      rowGroups_(std::move(rowGroups)),
        maxDefine_(std::dynamic_pointer_cast<const ParquetTypeWithId>(dataType)
                       ->maxDefine_),
        maxRepeat_(std::dynamic_pointer_cast<const ParquetTypeWithId>(dataType)
                       ->maxRepeat_),
        rowsInRowGroup_(-1) {}

  void enqueueRowGroup(uint32_t index, dwrf::BufferedInput& input);

  seekToRowGroup(uint32_t index);

  bool filterMatches(const RowGroup& rowGroup, common::Filter& filter);

 protected:
  std::shared_ptr<ParquetTypeWithId> type_;
  std::vector<const RowGroup * FOLLY_NULLABLE> rowGroups_;
  // Streams for this column in each of 'rowGroups_'. Will be created on or ahead of first
  // use, not at construction.
  std::vector<std::unique_ptr<SeekableInputStream>> > streams_;
  const ColumnChunk* columnChunk_;

  uint32_t maxDefine_;
  uint32_t maxRepeat_;

  int64_t rowsInRowGroup_;
  std::unique_ptr<PageDecoder> decoder_;
};

// Wrapper for static functions for Parquet columns.
class ParquetColumnReader {
 public:
  static std::unique_ptr<ParquetColumnReader> build(
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      ParquetParams& params,
      common::ScanSpec* scanSpec);
};

class ParquetStructColumnReader : public dwrf::SelectiveStructColumnReader {
 public:
  ParquetStructColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      ParquetParams& params,
      common::ScanSpec* scanSpec,
      memory::MemoryPool& pool)
      : SelectiveStructColumnReader(dataType, params, scanSpec, type->type), {
    auto& childSpecs = scanSpec->children();
    for (auto i = 0; i < childSpecs.size(); ++i) {
      if (childSpecs[i]->isConstant()) {
        continue;
      }
      auto childDataType = nodeType_->childByName(childSpecs[i]->fieldName());

      children_.push_back(ParquetColumnReader::build(
          childDataType, params, childSpecs[i].get(), input_, memoryPool_));
      childSpecs[i]->setSubscript(children_.size() - 1);
    }
  }

  bool filterMatches(const RowGroup& rowGroup) override;
  void seekToRowGroup(uint32_t index index) override;
};

class IntegerColumnReader : public dwrf::SelectiveIntegerColumnReader {
 public:
  IntegerColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      ParquetParams& params,
      uint32_t numBytes,
      common::ScanSpec* scanSpec)
      : SelectiveIntegerColumnReader(
            std::move(requestedType),
            params,
            scanSpec,
            dataType->type) {}

  bool hasBulkPath() const override {
    return true;
  }

  void seekToRowGroup(uint32_t index) override {}

  uint64_t skip(uint64_t numValues) override {
    auto& parquet = formatData_->as<ParquetData>();
    parquet->skip(numValues);
  }

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override {
    auto& data = formatData_->as<ParquetData>();
    VELOX_WIDTH_DISPATCH(
        sizeOfIntKind(type_->kind()), prepareRead, offset, rows, nullptr);

    readCommon<IntegerColumnReader>(rows);
  }

  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor) {
    auto& data = formatData<ParquetData>();
    pageDecoder_->readWithVisitor(nullptr, visitor);
  }
}

} // namespace facebook::velox::parquet
