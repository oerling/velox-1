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

#include "velox/dwio/common/SelectiveFloatingPointColumnReader.h"
#include "velox/dwio/dwrf/common/FloatingPointDecoder.h"

namespace facebook::velox::parquet {

template <typename TData, typename TRequested>
class FloatingPointColumnReader
    : public dwio::common::
          SelectiveFloatingPointColumnReader<TData, TRequested> {
 public:
  using ValueType = TRequested;

  using root = dwio::common::SelectiveColumnReader;
  using base =
      dwio::common::SelectiveFloatingPointColumnReader<TData, TRequested>;

  FloatingPointColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> nodeType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  void seekToRowGroup(uint32_t index) override {
    scanState().clear();
    readOffset_ = 0;
    formatData_->as<ParquetData>().seekToRowGroup(index);
  }
  
  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override {
    using T = SelectiveFloatingPointColumnReader<TData, TRequested>;
    base::template readCommon<T>(offset, rows, incomingNulls);
  }

  template <typename TVisitor>
  void readWithVisitor(RowSet rows, TVisitor visitor);

  FloatingPointDecoder<TData, TRequested> decoder_;
};

template <typename TData, typename TRequested>
SelectiveFloatingPointColumnReader<TData, TRequested>::
    SelectiveFloatingPointColumnReader(
        std::shared_ptr<const dwio::common::TypeWithId> requestedType,
        ParquetParams& params,
        common::ScanSpec& scanSpec)
    : dwio::common::SelectiveFloatingPointColumnReader<TData, TRequested>(
          std::move(requestedType),
          params,
          scanSpec), {}

template <typename TData, typename TRequested>
uint64_t FloatingPointColumnReader<TData, TRequested>::skip(
    uint64_t numValues) {
  return root::formatData_->skip(numValues);
}

template <typename TData, typename TRequested>
template <typename TVisitor>
void SelectiveFloatingPointColumnReader<TData, TRequested>::readWithVisitor(
    RowSet rows,
    TVisitor visitor) {
    formatData_->as<ParquetData>().readWithVisitor(visitor);
    root::readOffset_ += rows.back() + 1;
}

} // namespace facebook::velox::dwrf
