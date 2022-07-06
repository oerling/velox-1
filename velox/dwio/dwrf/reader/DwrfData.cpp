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


#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {


DwrfData::DwrfData(
		   std::shared_ptr<const dwio::common::TypeWithId> nodeType,
		   StripeStreams& stripe,
		   FlatMapContext flatMapContext)
  : pool_(stripe.getMemoryPool()),
    nodeType_(std::move(nodeType)),
    flatMapContext_(std::move(flatMapContext)),
      rowsPerRowGroup_{stripe.rowsPerRowGroup()}{
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  std::unique_ptr<dwio::common::SeekableInputStream> stream =
      stripe.getStream(encodingKey.forKind(proto::Stream_Kind_PRESENT), false);
  if (stream) {
    notNullDecoder_ = createBooleanRleDecoder(std::move(stream), encodingKey);
  }


  // We always initialize indexStream_ because indices are needed as
  // soon as there is a single filter that can trigger row group skips
  // anywhere in the reader tree. This is not known at construct time
  // because the first filter can come from a hash join or other run
  // time pushdown.
  indexStream_ = stripe.getStream(
      encodingKey.forKind(proto::Stream_Kind_ROW_INDEX), false);

      
}

uint64_t DwrfData::skip(uint64_t numValues) {
  if (notNullDecoder_) {
    // page through the values that we want to skip
    // and count how many are non-null
    std::array<char, BUFFER_SIZE> buffer;
    constexpr auto bitCount = BUFFER_SIZE * 8;
    uint64_t remaining = numValues;
    while (remaining > 0) {
      uint64_t chunkSize = std::min(remaining, bitCount);
      notNullDecoder_->next(buffer.data(), chunkSize, nullptr);
      remaining -= chunkSize;
      numValues -= bits::countNulls(
          reinterpret_cast<uint64_t*>(buffer.data()), 0, chunkSize);
    }
  }
  return numValues;
}

  
  void DwrfData::ensureRowGroupIndex() const {
    VELOX_CHECK(index_ || indexStream_, "Reader needs to have an index stream");
    if (indexStream_) {
      index_ = ProtoUtils::readProto<proto::RowIndex>(std::move(indexStream_));
    }
  }


  void DwrfData::readNulls(
    vector_size_t numValues,
    const uint64_t* incomingNulls,
    BufferPtr& nulls) {
  if (!notNullDecoder_ && !incomingNulls) {
    nulls = nullptr;
    return;
  }
  auto numBytes = bits::nbytes(numValues);
  if (!nulls || nulls->capacity() < numBytes + simd::kPadding) {
    nulls =
        AlignedBuffer::allocate<char>(numBytes + simd::kPadding, pool__);
  }
  nulls->setSize(numBytes);
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  if (!notNullDecoder_) {
    memcpy(nullsPtr, incomingNulls, numBytes);
    return;
  }
  memset(nullsPtr, bits::kNotNullByte, numBytes);
  notNullDecoder_->next(
      reinterpret_cast<char*>(nullsPtr), numValues, incomingNulls);
}

uint64_t DwrfData::skipNulls(uint64_t numValues) {
  if (notNullDecoder_) {
    // page through the values that we want to skip
    // and count how many are non-null
    std::array<char, BUFFER_SIZE> buffer;
    constexpr auto bitCount = BUFFER_SIZE * 8;
    uint64_t remaining = numValues;
    while (remaining > 0) {
      uint64_t chunkSize = std::min(remaining, bitCount);
      notNullDecoder_->next(buffer.data(), chunkSize, nullptr);
      remaining -= chunkSize;
      numValues -= bits::countNulls(
          reinterpret_cast<uint64_t*>(buffer.data()), 0, chunkSize);
    }
  }
  return numValues;
}
  
  

  std::vector<uint32_t> DwrfData::filterRowGroups(
    uint64_t rowGroupSize,
    const dwio::common::StatsContext& context) const {
  formatData_->filterRowGroups(rowGroupSize, context);
  if ((!index_ && !indexStream_) || !scanSpec_->filter()) {
    return ColumnReader::filterRowGroups(rowGroupSize, context);
  }

  ensureRowGroupIndex();
  auto filter = scanSpec_->filter();

  std::vector<uint32_t> stridesToSkip;
  for (auto i = 0; i < index_->entry_size(); i++) {
    const auto& entry = index_->entry(i);
    auto columnStats =
        buildColumnStatisticsFromProto(entry.statistics(), context);
    if (!testFilter(filter, columnStats.get(), rowGroupSize, type_)) {
      VLOG(1) << "Drop stride " << i << " on " << scanSpec_->toString();
      stridesToSkip.push_back(i); // Skipping stride based on column stats.
    }
  }
  return stridesToSkip;
}
}

  
