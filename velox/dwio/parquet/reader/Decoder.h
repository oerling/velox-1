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

//
// Created by Ying Su on 3/25/22.
//

#pragma once

#include <cstdint>
#include "dwio/dwrf/common/BufferedInput.h"
#include "velox/common/base/BitSet.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/type/Filter.h"
//#include "velox/common/base/SimdUtil.h"

namespace facebook::velox::parquet {

// For Parquet, Filter will be pushed to decoding process

  class FilterAwareDecoder {
 public:

  FilterAwareDecoder(
      const void* inputBuffer, // note we have loaded the full page
      uint64_t inputBytes,
      std::optional<common::Filter*> filter)
      : inputBuffer_(reinterpret_cast<const uint8_t*>(inputBuffer)),
        inputBytes_(inputBytes),
        filter_(filter),
        kHasBulkPath_(true) {
    DWIO_ENSURE(filter);
  }

  virtual ~FilterAwareDecoder() = default;
  /**
   * Seek over a given number of values.
   */
  virtual void skip(uint64_t numRows) = 0;

  /**
   * Read a number of values into the batch.
   * @param data the array to read into
   * @param numRows the number of values to read
   * @param nulls If the pointer is null, all values are read. If the
   *    pointer is not null, positions that are true are skipped.
   */
  //  virtual void
  //  next(void*& data, uint64_t numRows, BitSet& selectivityVec) = 0;

  virtual void next(BufferPtr outputBuffer, RowSet& rows, uint64_t numRows) = 0;
  void next(int64_t* data, uint64_t numValues, const uint64_t* nulls);

 protected:
  bool useFastPath(bool hasNulls) {
    return process::hasAvx2() &&
        (!filter_.has_value() || filter_.value()->isDeterministic()) &&
        kHasBulkPath_ &&
        (!hasNulls || (filter_.has_value() && !filter_.value()->testNull()));
  }

  const uint8_t* inputBuffer_;
  const uint64_t inputBytes_;
  //  const uint8_t* bufferEnd_;
  //  const uint32_t numInputValues_;
  std::optional<common::Filter*> filter_;

  bool kHasBulkPath_;
};


template <typename T>
class RleBpFilterAwareDecoder : FilterAwareDecoder {
 public:
  RleBpFilterAwareDecoder(
      const void* inputBuffer,
      uint64_t inputBytes,
      std::optional<common::Filter*> filter,
      uint8_t bitWidth)
      : FilterAwareDecoder(inputBuffer, inputBytes, filter),
        bitWidth_(bitWidth),
        currentValue_(0),
        repeatCount_(0),
        literalCount_(0) {
    DWIO_ENSURE(bitWidth_ < 64, "Decode bit width too large");
    byteEncodedLen_ = ((bitWidth_ + 7) / 8);
    maxVal_ = (1 << bitWidth_) - 1;
  }


  virtual void next(BufferPtr outputBuffer, RowSet& rows, uint64_t numRows)
      override {
    DWIO_ENSURE(
        outputBuffer->capacity() >= numRows * sizeof(T) + outputBuffer->size());

    if (filter_.has_value() && filter_.value() != nullptr) {
      readWithFilter(outputBuffer, numRows, rows);
    } else {
      readNoFilter(outputBuffer, numRows);
    }
  }

  virtual void skip(uint64_t numRows) override {
    uint64_t valuesSkipped = 0;
    while (valuesSkipped < numRows) {
      if (repeatCount_ > 0) {
        int repeatBatch = std::min(
            numRows - valuesSkipped, static_cast<uint64_t>(repeatCount_));
        repeatCount_ -= repeatBatch;
        valuesSkipped += repeatBatch;
      } else if (literalCount_ > 0) {
        uint32_t literalBatch = std::min(
            numRows - valuesSkipped, static_cast<uint64_t>(literalCount_));
        literalCount_ -= literalBatch;
        valuesSkipped += literalBatch;
      } else {
        if (!nextCounts()) {
          break;
        }
      }
    }
    DWIO_ENSURE(
        valuesSkipped == numRows,
        "RLE/BP decoder did not find enough values to skip");
  }

  static uint8_t computeBitWidth(uint32_t val) {
    if (val == 0) {
      return 0;
    }
    uint8_t ret = 1;
    while (((uint32_t)(1 << ret) - 1) < val) {
      ret++;
    }
    return ret;
  }

 private:
  void readNoFilter(BufferPtr outputBuffer, uint64_t numRows) {
    uint64_t offset = outputBuffer->size();
    auto outputBufferPtr =
        (T*)outputBuffer->template asMutable<T>() + offset * sizeof(T);
    uint64_t valuesRead = 0;

    while (valuesRead < numRows) {
      if (repeatCount_ > 0) {
        int repeatBatch =
            std::min(numRows - valuesRead, static_cast<uint64_t>(repeatCount_));
        std::fill(
            outputBufferPtr + valuesRead,
            outputBufferPtr + valuesRead + repeatBatch,
            static_cast<T>(currentValue_));
        repeatCount_ -= repeatBatch;
        valuesRead += repeatBatch;
      } else if (literalCount_ > 0) {
        uint32_t literalBatch = std::min(
            numRows - valuesRead, static_cast<uint64_t>(literalCount_));
        uint32_t actual_read =
            bitUnpack(outputBufferPtr + valuesRead, literalBatch);
        if (literalBatch != actual_read) {
          throw std::runtime_error("Did not find enough outputBufferPtr");
        }
        literalCount_ -= literalBatch;
        valuesRead += literalBatch;
      } else {
        if (!nextCounts()) {
          break;
        }
      }
    }
    outputBuffer->setSize(offset + valuesRead * sizeof(T));
    DWIO_ENSURE(
        valuesRead == numRows,
        "RLE/BP decoder did not find enough values to read");
  }

  void readWithFilter(
      BufferPtr outputBuffer,
      uint64_t numRows,
      BitSet& selectivityVec) {}

  void readWithFilter(BufferPtr outputBuffer, uint64_t numRows, RowSet& rows) {}

  uint32_t varintDecode() {
    uint32_t result = 0;
    uint8_t shift = 0;
    uint8_t len = 0;
    while (true) {
      auto byte = *inputBuffer_++;
      len++;
      result |= (byte & 127) << shift;
      if ((byte & 128) == 0)
        break;
      shift += 7;
      if (shift > 32) {
        throw std::runtime_error("Varint-decoding found too large number");
      }
    }
    return result;
  }

  bool nextCounts() {
    // Read the next run's indicator int, it could be a literal or repeated run.
    // The int is encoded as a vlq-encoded value.
    if (bitpackPosition_ != 0) {
      //    buffer_.inc(1);
      inputBuffer_++;
      bitpackPosition_ = 0;
    }
    auto indicator_value = varintDecode();

    // lsb indicates if it is a literal run or repeated run
    bool is_literal = indicator_value & 1;
    if (is_literal) {
      literalCount_ = (indicator_value >> 1) * 8;
    } else {
      repeatCount_ = indicator_value >> 1;
      // (ARROW-4018) this is not big-endian compatible, lol
      currentValue_ = 0;
      for (auto i = 0; i < byteEncodedLen_; i++) {
        currentValue_ |= (*inputBuffer_++ << (i * 8));
      }
      // sanity check
      if (repeatCount_ > 0 && currentValue_ > maxVal_) {
        throw std::runtime_error(
            "Payload value bigger than allowed. Corrupted file?");
      }
    }
    // TODO complain if we run out of buffer
    return true;
  }

  uint32_t bitUnpack(T* dest, uint32_t count) {
    auto mask = BITPACK_MASKS[bitWidth_];

    //    auto inputBuffer = reinterpret_cast<const uint8_t*>(inputBuffer_);

    for (uint32_t i = 0; i < count; i++) {
      //    T val = (buffer_.get<uint8_t>() >> bitpackPosition_) & mask;
      T val = (*inputBuffer_ >> bitpackPosition_) & mask;
      bitpackPosition_ += bitWidth_;
      while (bitpackPosition_ > BITPACK_DLEN) {
        inputBuffer_++;
        val |=
            (*inputBuffer_ << (BITPACK_DLEN - (bitpackPosition_ - bitWidth_))) &
            mask;
        bitpackPosition_ -= BITPACK_DLEN;
      }
      dest[i] = val;
    }
    return count;
  }

  // void readNoFilter(BufferPtr outputBuffer, uint64_t numRows);
  // void readWithFilter(
  //     BufferPtr outputBuffer,
  //     uint64_t numRows,
  //     BitSet& selectivityVec_);
  // uint32_t varintDecode();
  // bool nextCounts();
  // uint32_t bitUnpack(T* dest, uint32_t count);

  constexpr static const uint32_t BITPACK_MASKS[] = {
      0,          1,         3,        7,         15,        31,
      63,         127,       255,      511,       1023,      2047,
      4095,       8191,      16383,    32767,     65535,     131071,
      262143,     524287,    1048575,  2097151,   4194303,   8388607,
      16777215,   33554431,  67108863, 134217727, 268435455, 536870911,
      1073741823, 2147483647};
  static const uint8_t BITPACK_DLEN = 8;

  uint8_t bitWidth_;
  uint8_t bitpackPosition_ = 0;
  uint64_t currentValue_;
  uint32_t repeatCount_;
  uint32_t literalCount_;
  uint8_t byteEncodedLen_;
  uint32_t maxVal_;
};

} // namespace facebook::velox::parquet
