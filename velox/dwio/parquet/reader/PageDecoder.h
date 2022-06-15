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
#include "velox/dwio/parquet/reader/Decoder.h"
#include "velox/dwio/parquet/reader/ParquetThriftTypes.h"
#include "velox/vector/BaseVector.h"
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
  PageDecoder(
      std::unique_ptr<dwio::common::SeekableInputStream> stream,
      memory::MemoryPool& pool,
      int32_t maxDefine,
      int32_t maxRepeat,
      CompressionCodec::type codec)
      : pool_(pool),
        inputStream_(std::move(stream)),
        maxDefine_(maxDefine),
        maxRepeat_(maxRepeat),
        codec_(codec) {}

  // Advances 'numRows' top level rows.
  void skip(int64_t numRows);

  template <typename Visitor>
  void readWithVisitor(Visitor& visitor);

 private:
  // If the current page has nulls, returns a nulls bitmap owned by 'this'. This
  // is filled for 'numRows' bits.
  const uint64_t* readNulls(int32_t numRows);

  // Makes a decoder based on 'encoding_' for bytes from ''pageData_' to
  // 'pageData_' + 'encodedDataSize_'.
  void makedecoder();

  // Reads and skips pages until finding a data page that contains 'row'. Reads
  // and sets 'rowOfPage_' and 'numRowsInPage_' and initializes a decoder for
  // the found page.
  void readNextPage(int64_t row);

  PageHeader readPageHeader();
  void prepareDataPageV1(const PageHeader& pageHeader, int64_t row);
  void prepareDataPageV2(const PageHeader& pageHeader, int64_t row);
  void prepareDictionary(const PageHeader& pageHeader);
  void makeDecoder();

  // Returns a pointer to contiguous space for the next 'size' bytes
  // from current position. Copies data into 'copy' if the range
  // straddles buffers. Allocates or resizes 'copy' as needed.
  const char* FOLLY_NONNULL readBytes(int32_t size, BufferPtr& copy);

  // Decompresses data starting at 'pageData_', consuming 'compressedsize' and
  // producing up to 'uncompressedSize' bytes. The The start of the decoding
  // result is returned. an intermediate copy may be made in 'uncompresseddata_'
  const char* FOLLY_NONNULL uncompressData(
      const char* FOLLY_NONNULL pageData,
      uint32_t compressedSize,
      uint32_t uncompressedSize);

  template <typename T>
  T readField(const char*& ptr) {
    T data = *reinterpret_cast<const T*>(ptr);
    ptr += sizeof(T);
    return data;
  }

  // Starts iterating over 'rows', which may span multiple pages. 'rows' are
  // relative to current position, with 0 meaning the first
  // unprocessed value in the current page, i.e. the row after the
  // last row touched on a previous call to skip() or
  // readWithVisitor(). This is the first row of the first data page
  // if first call.
  void startVisit(folly::Range<const vector_size_t*> rows);

  // Seeks to the next page in a range given by startVisit().
  // Returns true if there are unprocessed rows in the set given to
  // startVisit(). Seeks 'this' to
  // the appropriate page and sets 'rowsForPage' to refer to the subset
  // of 'rows' that are on the current page. The numbers in rowsForPage
  // are relative to the first unprocessed value on the page, for a
  // new page 0 means the first value. Sets 'nulls' to nullptr if all
  // rows between the first and last accessed row on the page are
  // non-null, otherwise sets 'nulls' to a nulls bitmap where bit 0 is
  // the null flag for rowsOnPage[0]. there are rowsOnPage.back() -
  // rowsOnPage.front() + 1 bits in 'nulls'.
  bool rowsForPage(
      folly::Range<const vector_size_t*>& rows,
      const uint64_t* FOLLY_NULLABLE& nulls);

  memory::MemoryPool& pool_;

  std::unique_ptr<dwio::common::SeekableInputStream> inputStream_;
  const int32_t maxDefine_;
  const int32_t maxRepeat_;
  const CompressionCodec::type codec_;
  const char* bufferStart_{nullptr};
  const char* bufferEnd_{nullptr};

  BufferPtr defineOutBuffer_;
  BufferPtr repeatOutBuffer_;
  std::unique_ptr<RleBpFilterAwareDecoder<uint8_t>> repeatDecoder_;
  std::unique_ptr<RleBpFilterAwareDecoder<uint8_t>> defineDecoder_;

  // Encoding of current page.
  Encoding::type encoding_;

  // Row number of first value in current page from start of ColumnChunk.
  int64_t rowOfPage_{0};

  // Number of rows in current page.
  int32_t numRowsInPage_{0};

  // Copy of data if data straddles buffer boundary.
  BufferPtr pageBuffer_;

  // Uncompressed data for the page. Rep-def-data in V1, data alone in V2.
  BufferPtr uncompressedData_;

  // First byte of uncompressed encoded data. Contains the encoded data as a
  // contiguous run of bytes.
  const char* pageData_{nullptr};
  std::unique_ptr<Dictionary> dictionary_;
  const char* dict_ = nullptr;

  // Offset of current page's header from start of ColumnChunk.
  uint64_t pageStart_{0};

  // Offset of first byte after current page' header.
  uint64_t pageDataStart_{0};

  // Number of bytes starting at pageData_ for current encoded data.
  int32_t encodedDataSize_{0};

  // Below members Keep state between calls to readWithVisitor().

  // Original rows in Visitor.
  const vector_size_t* FOLLY_NULLABLE visitorRows_{nullptr};
  int32_t numVisitorRows_{0};

  // 'rowOfPage_' at the start of readWithVisitor().
  int64_t initialRowOfPage_{0};

  // Index in 'visitorRows_' for the first row that is beyond the
  // current page. Equals 'numVisitorRows_' if all are on current page.
  int32_t currentVisitorRow_{0};

  // Row relative to ColumnChunk for first unvisited row. 0 if nothing
  // visited.  The rows passed to readWithVisitor from rowsForPage()
  // are relative to this.
  int64_t firstUnvisited_{0};

  // Offset of 'visitorRows_[0]' relative too start of ColumnChunk.
  int64_t visitBase_{0};

  //  Temporary for rewriting rows to access in readWithVisitor when moving
  //  between pages. Initialized from the visitor.
  raw_vector<vector_size_t>* FOLLY_NULLABLE rowsCopy_{nullptr};

  // If 'rowsCopy_' is used, this is the difference between the rows in
  // 'rowsCopy_' and the row numbers in 'rows' given to readWithVisitor().
  int32_t rowNumberBias_{0};

  // Decoders. Only one will be set at a time.
  std::unique_ptr<dwrf::DirectDecoder<true>> directDecoder_;

  // Add decoders for other encodings here.
};

template <typename Visitor>
void PageDecoder::readWithVisitor(Visitor& visitor) {
  constexpr bool hasFilter =
      !std::is_same<typename Visitor::FilterType, common::AlwaysTrue>::value;

  auto rows = visitor.rows();
  auto numRows = visitor.numRows();
  startVisit(folly::Range<const vector_size_t*>(rows, numRows));
  rowsCopy_ = &visitor.rowsCopy();
  folly::Range<const vector_size_t*> pageRows;
  const uint64_t* nulls = nullptr;
  while (rowsForPage(pageRows, nulls)) {
    visitor.setRows(pageRows);
    auto firstResult = hasFilter ? visitor.numRows() : 0;
    if (nulls) {
      directDecoder_->readWithVisitor<true>(nulls, visitor);
    } else {
      directDecoder_->readWithVisitor<false>(nulls, visitor);
    }
    if (hasFilter && rowNumberBias_) {
      visitor.offsetOutputRows(firstResult, rowNumberBias_);
    }
  }
}

} // namespace facebook::velox::parquet
