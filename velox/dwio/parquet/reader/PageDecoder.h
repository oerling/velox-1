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

namespace facebook::velox::parquet {

  class Dictionary {
 public:
  Dictionary(const void* dict, uint32_t size) : dict_(dict), size_(size) {}

 private:
  const void* dict_;
  uint32_t size_;
}; 

class PageDecoder {
 public:
  PageDecoder(std::unique_ptr<dwrf::SeekableInputStream> stream)
      : inputStream_(std::move(stream)),
        chunkReadOffset_(0),
        remainingRowsInPage_(0),
        dictionary_(nullptr) {}

  template <typename Visitor>
  void readWithVisitor(Visitor& visitor) {

  }

 protected:
  void setupDecoderForPage(
      const PageHeader& pageHeader,
      const Encoding::type& pageEncoding);

  void readNextPage();
  PageHeader readPageHeader();
  void prepareDataPageV1(const PageHeader& pageHeader);
  void prepareDataPageV2(const PageHeader& pageHeader);
  void prepareDictionary(const PageHeader& pageHeader);

  template <typename T>
  T readNumber() {
    T number;
    dwrf::readBytes(sizeof(T), inputStream_.get(), &number, bufferStart_, bufferEnd_);
    offsetInChunk_ += sizeof(T);
    return number;
  }

  // Returns a pointer to contiguous space for the next 'size' bytes
  // from current position. Copies data into 'copy' if the range
  // straddles buffers. Allocates or resizes 'copy' as needed.
  const char* FOLLY_NONNULL readBytes(int32_t size, BufferPtr& copy);
  
  bool canNotHaveNull();

  std::unique_ptr<dwrf::SeekableInputStream> inputStream_;
  const char* bufferStart_;
  char* bufferEnd_;


  BufferPtr defineOutBuffer_;
  BufferPtr repeatOutBuffer_;
  const char* definitionStart_{nullptr};
  const char* repetitionStart_{nullptr};
  BufferPtr definitionCopy_;
  BufferPtr repetitionCopy_;
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
