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


#include "velox/dwio/parquet/reader/ParquetData.h"

namespace facebook::velox::parquet {

  void ParquetData::enqueueRowGroup(uint32_t index, dwrf::BufferedInput& input) {
    streams_.resize(rowGroups_.size());
    auto& columnChunk = rowGroups_[index].columns[type->column];
    auto columnData = columnChunk.meta_data;
    auto size = columnData.total_compressed_size;
    auto start = columnData.data_page_offset;
      streams_[index] = input->enqueue(dwio::common::Region(start, size), type_->column());
  }

  void ParquetData::seekToRowGroup(uint32_t index) {
    VELOX_CHECK_LT(index, streams_.size());
    VELOX_CHECK(streams_[index], "Stream not enqueued for column");
    decoder_ = std::make_unique<PageDecoder>(std::move(streams_[index], pool_);
  }

  
  
}


