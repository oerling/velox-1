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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/FormatData.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/dwrf/common/ByteRLE.h"
#include "velox/dwio/dwrf/common/Compression.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/reader/EncodingContext.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwrf {

// DWRF specific functions shared between all readers.
  class DwrfData : public dwio::common::FormatData {
  public:
    DwrfData(StripeStreams& stripe,
	     FlatMapContext flatMapContext,
	     dwio::common::ScanSpec& scanSpec);

    void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      VectorPtr* result,
      BufferPtr& nulls) override;

 	     
    virtual std::vector<uint32_t> filterRowGroups(
      uint64_t rowsPerRowGroup,
      vonst StatsWriterInfo& context) {
    VELOX_NYI();
  }

    bool hasNulls() const override {
      return notNullDecoder_ != nullptr;
    }
    
  // seeks possible nulls to the row group and returns a PositionsProvider for the other streams.
  PositionsProvider seektoRowGroup(uint32_t index) {
    positions_ = toPositions(index_->entry(index));
      ensureRowGroupIndex();
      tempPositions_ = toPositionsInner(index_->entry(index));
      dwio::common::PositionProvider positionsProvider(tempPositions);
      
    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }
    return positionsProvider;
  }

  
 private:
  void ensureRowGroupIndex();
  
    static std::vector<uint64_t> toPositionsInner(const proto::RowIndexEntry& entry) {
  return std::vector<uint64_t>(
      entry.positions().begin(), entry.positions().end());
    }
    std::unique_ptr<ByteRleDecoder> notNullDecoder_;
  const std::shared_ptr<const dwio::common::TypeWithId> nodeType_;
  memory::MemoryPool& memoryPool_;
  FlatMapContext flatMapContext_;
  std::unique_ptr<dwio::common::SeekableInputStream> indexStream_;
  std::unique_ptr<proto::RowIndex> index_;
  // Number of rows in a row group. Last row group may have fewer rows.
  uint32_t rowsPerRowGroup_;

  std::vector<uint64_t> positions_;
}  


// DWRF specific initialization.
class DwrfParams : public FormatParams {
 public:
  DwrfParams(
      memory::MemoryPool& pool,
      StripeStreams& stripe,
      FlatMapContext context = FlatMapContext::nonFlatMapContext())
      : FormatParams(pool), stripe_(stripe), flatMapContext_(context) {}

  std::unique_ptr<FormatData> toFormatData(
					   const std::shared_ptr<const dwio::common::TypeWithId>& type, dwio::common::ScanSpec& scanSpec) override {
    return std::make_unique<DwrfData>(type, stripe_, flatMapContext_);
  }

  StripeStreams& stripeStreams() {
    return stripeStreams_;
  }

  FlatMapContext flatMapContext() {
    return flatMapContext_;
  }
  
 private:
  StripeStreams& stripe_;
  FlatMapContext flatMapContext_;
};

} // namespace facebook::velox::dwrf


