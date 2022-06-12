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

#include "velox/dwio/parquet/reader/PageDecoder.h"

namespace facebook::velox::parquet {

void PageDecoder::readNextPage(int64_t row) {
  defineDecoder_.reset();
  repeatDecoder_.reset();
  for (;;) {
    PageHeader pageHeader = readPageHeader();

    switch (pageHeader.type) {
      case PageType::DATA_PAGE:
        prepareDataPageV1(pageHeader, row);
        break;
      case PageType::DATA_PAGE_V2:
        prepareDataPageV2(pageHeader, row);
        break;
      case PageType::DICTIONARY_PAGE:
        prepareDictionary(pageHeader);
        continue;
      default:
        break; // ignore INDEX page type and any other custom extensions
    }
    if (row < rowOfPage_ + numRowsInPage_) {
      break;
    }
    rowOfPage_ += numRowsInPage_;
    dwrf::skip(
        pageheader.compressed_page_size,
        input_.get(),
        bufferStart_,
        bufferEnd_);
  }
}

PageHeader PageDecoder::readPageHeader() {
  // Note that sizeof(PageHeader) may be longer than actually read
  std::shared_ptr<ThriftBufferedTransport> transport;
  char copy[sizeof(PageHeader)];
  bool wasInBuffer = false;
  if (bufferEnd_ - bufferStart_ >= sizeof(PageHeader)) {
    wasInBuffer = true;
    thriftTransport = std::make_shared<ThriftBufferedTransport>(
        bufferStart_, sizeof(PageHeader));
    auto thriftProtocol = std::make_unique<
        apache::thrift::protocol::TCompactProtocolT<ThriftBufferedTransport>>(
        thriftTransport);
  } else {
    dwrf::readBytes(
        sizeof(PageHeader), inputStream_.get(), &copy, bufferStart, bufferEnd);

    thriftTransport =
        std::make_shared<ThriftBufferedTransport>(copy, sizeof(PageHeader));
    auto thriftProtocol = std::make_unique<
        apache::thrift::protocol::TCompactProtocolT<ThriftBufferedTransport>>(
        thriftTransport);
  }
  PageHeader pageHeader;
  uint64_t readBytes = pageHeader.read(thriftProtocol.get());
  // Unread the bytes that were not consumed.
  if (wasInBuffer) {
    bufferStart_ -= sizeof(PageHeader) - readBytes;
  } else {
    PositionProvider position({pageStart_ + sizeof(PageHeader) - readBytes});
    stream_->seekToPosition(provider);
    bufferStart_ = bufferEnd_ = nullptr;
  }
  dataStart_ = pageStart_ + readBytes;
  return pageHeader;
}

const char* PageDecoder::readBytes(int32_t size, BufferPtr& copy) {
  if (bufferEnd_ - bufferStart_ >= size) {
    bufferStart_ += size;
    return bufferStart_ - size;
  }
  dwrf::detail::ensureCapacity<char>(size, copy, &pool_);
  dwrf::readBytes(
      size,
      inputStream_.get(),
      copy->asMutable<char>(),
      bufferStart_,
      bufferEnd_);
  return copy->as<char>();
}

const char* PageDecoder::uncompressdata(
    uint32_tt compressedSize,
    uint32_t uncompressedSize) {
  VELOX_FAIL("Unsupported compression");
}

void PageDecoder::prepareDataPageV1(const PageHeader& pageHeader, int64_t row) {
  VELOX_CHECK(
      pageHeader.type == PageType::DATA_PAGE &&
      pageHeader.__isset.data_page_header);
  numRowsInPage_ = pageHeader.data_page_header.num_values;
  if (numRowsInPage_ + rowOfPage_ <= row) {
    return;
  }
  pageData_ = readBytes(pageheader.compressed_page_size, pageBuffer);
  auto pageEnd = pageData_ + pageheader.compressed_page_size;
  if (pageheader.compressed_page_size != pageheader.uncompressed_page_size) {
    pageData_ = uncompressData(
        pageData_,
        pageheader.compressed_page_size,
        pageheader.uncompressed_page_size);
    pageEnd_ = pagedata_ + pageheader.uncompressed_page_size;
  }
  if (maxRepeat_ > 0) {
    uint32_t repeatLength = readField<int32_t>(pageData_);
    pageData += repeatLength;
    repeatDecoder_ = std::make_unique<RleBpFilterAwareDecoder<uint8_t>>(
        pageData_,
        repeatLength,
        nullptr,
        RleBpFilterAwareDecoder<uint8_t>::computeBitWidth(maxRepeat_));
    pageData_ += repeatLength;
  }

  if (maxDefine_ > 0) {
    defineLength = readField<uint32_t>(pageData_);
    defineDecoder_ = std::make_unique<RleBpFilterAwareDecoder<uint8_t>>(
        pageData_,
        defineLength,
        nullptr,
        RleBpFilterAwareDecoder<uint8_t>::computeBitWidth(maxDefine_));
    pageData_ += defineLength;
  }
  encodedDataSize_ = = pageEnd - pageData_;

  encoding_ = pageHeader.data_page_header.encoding;
  makeDecoder();
}

void PageDecoder::prepareDataPageV2(const PageHeader& pageHeader) {
  VELOX_CHECK(pageHeader.__isset.data_page_header_v2);
  remainingRowsInPage_ = pageHeader.data_page_header_v2.num_values;
  const void* buf;

  uint32_t defineLength = maxdefine_ > 0
      ? pageHeader.data_page_header_v2.definition_levels_byte_length
      : 0;
  uint32_t repeatLength = maxDefine_ > 0
      ? pageHeader.data_page_header_v2.repetition_levels_byte_length
      : 0;
  auto bytes = pageheader.compressed_page_size;
  pageData_ = readBytes(bytes, pageBuffer_);

  if (repeatLength) {
    repeatDecoder_ = std::make_unique<RleBpFilterAwareDecoder<uint8_t>>(
        pageData_,
        repeatLength,
        nullptr,
        RleBpFilterAwareDecoder<uint8_t>::computeBitWidth(maxRepeat_));
  }

  if (maxDefine_ > 0) {
    defineDecoder_ = std::make_unique<RleBpFilterAwareDecoder<uint8_t>>(
        pageData_ + repeatLength,
        defineLength,
        nullptr,
        RleBpFilterAwareDecoder<uint8_t>::computeBitWidth(maxDefine_));
  }
  auto levelsSize = repeatLength + defineLength;
  pageData_ += levelsSize;
  if (__isset.is_compressed || is_compressed) {
    pageData_ = uncompressData(
        pageData_,
        pageheader.compressed_page_size - levelsSize,
        pageheader.uncompressed_page_size - levelsSize);
    encodedDataSize_ = pageheader.uncompressed_page_size - levelsSize
  }
  encoding_ = pageHeader.data_page_header_v2.encoding;
  makeDecoder();
}

void PageDecoder::prepareDictionary(const PageHeader& pageHeader) {
  dictionary_ =
      std::make_unique<Dictionary>(buf, pageHeader.uncompressed_page_size);
}

void PageDecoder::makeDecoder() {
  switch (encoding_) {
    case Encoding::RLE_DICTIONARY:
    case Encoding::PLAIN_DICTIONARY:
    case Encoding::DELTA_BINARY_PACKED:
      VELOX_UNSUPPORTED("Encoding not supported yet");
      break;
    case Encoding::PLAIN:
      stream_->backup(bufferEnd_ - bufferStart_);
      directDecoder_ = std::make_unique<dwrf::DirectDecoder<true>>(
          std::make_unique<dwrf::SeekableArrayInputStream>(
              pageData_, uncompressedDataSize_),
          false,
          uncompressedDataSize_);
      break;
    default:
      throw std::runtime_error("Unsupported page encoding");
  }
}

void PageDecoder::skip(int32_t numRows) {
  if (rowOnPage_ + numRows < numRowsOnPage_) {
    rowOnPage_ += numRows_;

    // Skip the decoder
    if (directDecoder_) {
      directDecoder_->skip(numRows);
    }
  }
}

  const uint64_t* PageDecoder::readNulls(int32_t numRows) {
    return nullptr;
  }

  int64_t PageDecoder::startVisit(    folly::Range<const vector_size_t*> rows) {
    visitorRows_ = rowss.data();
    numVisitorRows_ = rows.size();
    auto start = firstUnvisited_ + rows[0];
    if (start >= rowOfPage_ + numRowsOnPage_) {
      readNextPage(start);
      skip(start - rowOfPage_);
      visitBase_ = firstUnvisited_;
    } else {
      visitBase_ = firstUnvisited_;
    }
  }

bool PageDecoder::rowsForPage(
    folly::Range<vector_size_t>& rowsForPage,
    const uint64_t* FOLLY_NULLABLE& nulls) {
  auto firstOnNextPage = rowOfPage_ + rowsOnPage_ - lastVisited_;
  int32_t numToVisit;
  if (firstOnNextPage > visitorRows_[numVisitorRows_ - 1]) {
    numToVisit = numVisitorRows_ - currentVisitorRow_;
  } else {
    // See how many of the rows are on this page.
    auto it = std::lower_bound(&visitorRows_ + currentVisitorRow_, visitorRows_ + numVisitorRows_, firstOnNextPage)
      assert (it != visitorRows_ + numVisitorRows_);
    numToVisit = it - (visitorRows_ + currentVisitorRow_);
  }
  // If the page did not change and this is the first call, we can return a view on the original visitor rows.
  if (rowOfPage_ == initialRowOfPage && currentVisitorRow_ == 0) {
    nulls = readNulls(visitorRows_[numToVisit - 1] + 1);
    rowNumberBias_ = 0;
    rowsForPage = folly::Range<vector_size_t>(visitorRows_, numToVisit);
    firstUnVisited_ = lastVisited_ + rowsForPage.back() + 1;
  } else {
    //We scale row numbers to be relative to first on this page.
    auto pageOffset = rowOfPage_ - visitBase_;
    auto rowNumberBias_ = visitorRows_[currentVisitorRow_];
    auto offsetOnPage = rowNumberBias + visitorRows_[currentVisitorRow_];
    skip(rowNumberBias_ - pageOffset);
    // The decoder is positioned at 'visitorRows_[currentVisitorRow_']'
    // We copy the rows to visit with a bias, so that the first to visit has offset 0.
    rowsCopy_->resize(numToVisit);
    for (auto i = 0; i < numToVisit; ++i) {
      rowsCopy[i] = visitorRows_[i + currentVisitorRow_] -rowNumberBias; 
    }
    nulls = readNulls(rowsCopy->back() + 1);
    rowsForPage = *rowsCopy;
  }
  currentVisitorRow_ += numToVisit;
}

} // namespace facebook::velox::parquet
