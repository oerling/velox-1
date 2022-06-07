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
#include "velox/dwio/dwrf/common/DirectDecoder.h"

namespace facebook::dwio::parquet {

class PageDecoder {
 public:
  PageDecoder(std::unique_ptr<dwio::dwrf::SeekableInputStream> stream)
      : inputStream_(std::move(stream)),

        chunkReadOffset_(0),
        remainingRowsInPage_(0),
        dictionary_(nullptr) {}

  template <typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    VELOX_CHECK(!nulls, "Parquet does not accept incoming nulls");
  }

 protected:
  virtual int loadDataPage(
      const PageHeader& pageHeader,
      const Encoding::type& pageEncoding) = 0;

  void readNextPage();
  PageHeader readPageHeader();
  void prepareDataPageV1(const PageHeader& pageHeader);
  void prepareDataPageV2(const PageHeader& pageHeader);
  void prepareDictionary(const PageHeader& pageHeader);
  bool canNotHaveNull();

 protected:
  BufferPtr defineOutBuffer_;
  BufferPtr repeatOutBuffer_;
  std::unique_ptr<RleBpFilterAwareDecoder<uint8_t>> repeatDecoder_;
  std::unique_ptr<RleBpFilterAwareDecoder<uint8_t>> defineDecoder_;

  // in bytes
  uint64_t chunkReadOffset_;
  int64_t remainingRowsInPage_;
  BufferPtr pageBuffer_;

  std::unique_ptr<Dictionary> dictionary_;
  const char* dict_ = nullptr;
};

template <typename Visitor>
void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
  VELOX_CHECK(!nulls);
}

} // namespace facebook::dwio::parquet
