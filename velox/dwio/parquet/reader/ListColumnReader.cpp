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

#include "velox/dwio/parquet/reader/ListColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"

namespace facebook::velox::parquet {

ListColumnReader::ListColumnReader(
    std::shared_ptr<const dwio::common::TypeWithId> requestedType,
    ParquetParams& params,
    common::ScanSpec& scanSpec,
    common::ScanSpec& topLevelScanSpec)
    : ParquetRepeatedColumnReader(
          requestedType,
          params,
          scanSpec,
          topLevelScanSpec) {
  auto& childType = requestedType->childAt(0);

  elementReader_ = ParquetColumnReader::build(
      childType, params, *scanSpec.children()[0], scanSpec, true);
}
uint64_t ListColumnReader::skip(uint64_t numRows) {
  VELOX_CHECK(lengths_);
}

void ListColumnReader::seekToRowGroup(uint32_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  scanState().clear();
  readOffset_ = 0;
  elementReader_->seekToRowGroup(index);
}

void ListColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* /*incomingNulls*/) {
  if (!offsets_) {
    // The topmost list reader reads the repdefs for the left subtree.
    int numTop = (offset - readOffset_) + rows.back() + 1;
    ParquetRepeatedColumnReader::readLeafRepDefs(this, numTop);
  }
  seekTo(offset, false);

  readOffset_ = rows.back() + 1;
}

void ListColumnReader::getValues(RowSet rows, VectorPtr* result) {
  VectorPtr elements;
  if (elementReader_) {
    elementReader_->getValues(rows, &elements);
  }
  *result = std::make_shared<ArrayVector>(
      &memoryPool_,
      nodeType_->type,
      resultNulls_,
      rows.size(),
      offsets_,
      lengths_,
      elements);
}

} // namespace facebook::velox::parquet
