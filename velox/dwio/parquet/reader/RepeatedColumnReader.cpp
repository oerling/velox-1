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

  PageReader* readLeafRepDefs(
      SelectiveColumnReader* FOLLY_NONNULL reader,
      int32_t numTop) {
    auto children = reader->children();
    if (children.empty()) {
      auto pageReader = reader->formatData().as<ParquetData>().reader();
      pageReader->decodeRepDefs(numTop);
      return pageReader;
    }
    PageReader* pageReader;
    for (auto i = 0; i < children.size(); ++i) {
      auto child = children[i];

      if (i == 0) {
        pageReader = readLeafRepDefs(child, numTop);
        auto& type =
            *reinterpret_cast<const ParquetTypeWithId*>(&reader->nodeType());
        if (auto structChild = dynamic_cast<StructColumnReader*>(reader)) {
          VELOX_NYI();
        }
        auto nested = dynamic_cast<ParquetRepeatedColumnReader*>(reader);
        VELOX_CHECK(nested);
        pageReader->getOffsetsAndNulls(
            type.maxRepeat_,
            type.maxDefine_,
            nested->offsets_,
            nested->lengths_,
            nested->nullsInReadRange());
      } else {
        readLeafRepDefs(child, numTop);
      }
    }
    return pageReader;
  }

    static void enqueueChildren(
      SelectiveColumnReader* reader,
      uint32_t index,
      dwio::common::BufferedInput& input) {
    auto children = reader->children();
    if (children.empty()) {
      reader->formatData().as<ParquetData>().enqueueRowGroup(index, input);
      return;
    }
    for (auto* child : children) {
      enqueueChildren(child, index, input);
    }
  }


  
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

  void ListColumnReader::enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input) {
    enqueueChildren(this, index, input);
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
    const uint64_t* incomingNulls) {
    // The topmost list reader reads the repdefs for the left subtree.
  if (!offsets    int numTop = (offset - readOffset_) + rows.back() + 1;
    ParquetRepeatedColumnReader::readLeafRepDefs(this, numTop);
  }
  SelectiveListColumnReader(offset, rows, incomingNulls);
}


} // namespace facebook::velox::parquet
