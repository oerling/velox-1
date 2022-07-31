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

#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/common/IntDecoder.h"


namespace facebook::velox::parquet {

struct DropValues;

template <bool isSigned>
class RleDecoder : public dwio::common::IntDecoder<isSigned> {
 public:
  using super = dwio::common::IntDecoder<isSigned>;

  RleDecoder(
      std::unique_ptr<dwio::common::SeekableInputStream> input,
      uint8_t bitWidth,
      uint32_t numBytes)
      : IntDecoder<isSigned>{std::move(input), useVInts, numBytes},
	bitWidth_(bitWidth),
        remainingValues(0),
        value(0),
        repeating(false) {}

  void skip(uint64_t numValues) override;

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    while (numValues > 0) {
      if (remainingValues == 0) {
        readHeader();
      }
      uint64_t count = std::min<int>(numValues, remainingValues);
      remainingValues -= count;
      numValues -= count;
      if (repeating) {
        value += delta * static_cast<int64_t>(count);
      } else {
        IntDecoder<isSigned>::skipLongsFast(count);
      }
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    if (dwio::common::useFastPath<Visitor, hasNulls>(visitor)) {
      fastPath<hasNulls>(nulls, visitor);
      return;
    }
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    int32_t toSkip;
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();
    for (;;) {
      if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
        toSkip = visitor.processNull(atEnd);
      } else {
        if (hasNulls && !allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) {
            return;
          }
        }

        // We are at a non-null value on a row to visit.
        if (!remainingValues) {
          readHeader();
        }
        if (repeating) {
          toSkip = visitor.process(value, atEnd);
        } else {
          value = IntDecoder<isSigned>::readLong();
          toSkip = visitor.process(value, atEnd);
        }
        --remainingValues;
      }
      ++current;
      if (toSkip) {
        skip<hasNulls>(toSkip, current, nulls);
        current += toSkip;
      }
      if (atEnd) {
        return;
      }
    }
  }

  template <typename T>
  static extractBits(uint8_t bitWidth, int32_t bitOffset, const  uint64_t* base, const int32_t* indices, int32_t numIndices, int32_t indexBias, T* out) {
    if (bitWidth < 32 
  }
  
 private:
  template <bool hasNulls, typename Visitor>
  void fastPath(const uint64_t* nulls, Visitor& visitor) {
    constexpr bool hasFilter =
        !std::is_same<typename Visitor::FilterType, common::AlwaysTrue>::value;
    constexpr bool hasHook =
        !std::is_same<typename Visitor::HookType, dwio::common::NoHook>::value;
    auto rows = visitor.rows();
    auto numRows = visitor.numRows();
    auto rowsAsRange = folly::Range<const int32_t*>(rows, numRows);
    if (hasNulls) {
      raw_vector<int32_t>* innerVector = nullptr;
      auto outerVector = &visitor.outerNonNullRows();
      if (Visitor::dense) {
        dwio::common::nonNullRowsFromDense(nulls, numRows, *outerVector);
        if (outerVector->empty()) {
          visitor.setAllNull(hasFilter ? 0 : numRows);
          return;
        }
        bulkScan<hasFilter, hasHook, true>(
            folly::Range<const int32_t*>(rows, outerVector->size()),
            outerVector->data(),
            visitor);
      } else {
        innerVector = &visitor.innerNonNullRows();
        int32_t tailSkip = -1;
        auto anyNulls = dwio::common::nonNullRowsFromSparse < hasFilter,
             !hasFilter &&
            !hasHook >
                (nulls,
                 rowsAsRange,
                 *innerVector,
                 *outerVector,
                 (hasFilter || hasHook) ? nullptr : visitor.rawNulls(numRows),
                 tailSkip);
        if (anyNulls) {
          visitor.setHasNulls();
        }
        if (innerVector->empty()) {
          skip<false>(tailSkip, 0, nullptr);
          visitor.setAllNull(hasFilter ? 0 : numRows);
          return;
        }
        bulkScan<hasFilter, hasHook, true>(
            *innerVector, outerVector->data(), visitor);
        skip<false>(tailSkip, 0, nullptr);
      }
    } else {
      bulkScan<hasFilter, hasHook, false>(rowsAsRange, nullptr, visitor);
    }
  }

  template <bool hasFilter, bool hasHook, bool scatter, typename Visitor>
  void processRun(
      const int32_t* rows,
      int32_t rowIndex,
      int32_t currentRow,
      int32_t numRows,
      const int32_t* scatterRows,
      int32_t* filterHits,
      typename Visitor::DataType* values,
      int32_t& numValues,
      Visitor& visitor) {
    if (Visitor::dense) {
      super::bulkRead(numRows, values + numValues);
    } else {
      super::bulkReadRows(
          folly::Range<const int32_t*>(rows + rowIndex, numRows),
          values + numValues,
          currentRow);
    }
    visitor.template processRun<hasFilter, hasHook, scatter>(
        values + numValues,
        numRows,
        scatterRows,
        filterHits,
        values,
        numValues);
  }

  // Returns 1. how many of 'rows' are in the current run 2. the
  // distance in rows from the current row to the first row after the
  // last in rows that falls in the current run.
  template <bool dense>
  std::pair<int32_t, std::int32_t> findNumInRun(
      const int32_t* rows,
      int32_t rowIndex,
      int32_t numRows,
      int32_t currentRow) {
    DCHECK_LT(rowIndex, numRows);
    if (dense) {
      auto left = std::min<int32_t>(remainingValues, numRows - rowIndex);
      return std::make_pair(left, left);
    }
    if (rows[rowIndex] - currentRow >= remainingValues) {
      return std::make_pair(0, 0);
    }
    if (rows[numRows - 1] - currentRow < remainingValues) {
      return std::pair(numRows - rowIndex, rows[numRows - 1] - currentRow + 1);
    }
    auto range = folly::Range<const int32_t*>(
        rows + rowIndex,
        std::min<int32_t>(remainingValues, numRows - rowIndex));
    auto endOfRun = currentRow + remainingValues;
    auto bound = std::lower_bound(range.begin(), range.end(), endOfRun);
    return std::make_pair(bound - range.begin(), bound[-1] - currentRow + 1);
  }

  template <bool hasFilter, bool hasHook, bool scatter, typename Visitor>
  void bulkScan(
      folly::Range<const int32_t*> nonNullRows,
      const int32_t* scatterRows,
      Visitor& visitor) {
    auto numAllRows = visitor.numRows();
    visitor.setRows(nonNullRows);
    auto rows = visitor.rows();
    auto numRows = visitor.numRows();
    auto rowIndex = 0;
    int32_t currentRow = 0;
    auto values = visitor.rawValues(numRows);
    auto filterHits = hasFilter ? visitor.outputRows(numRows) : nullptr;
    int32_t numValues = 0;
    for (;;) {
      if (remainingValues) {
        auto [numInRun, numAdvanced] =
            findNumInRun<Visitor::dense>(rows, rowIndex, numRows, currentRow);
        if (!numInRun) {
          // We are not at end and the next row of interest is after this run.
          VELOX_CHECK(!numAdvanced, "Would advance past end of RLEv1 run");
        } else if (repeating) {
          visitor.template processRle<hasFilter, hasHook, scatter>(
              value,
              0,
              numInRun,
              currentRow,
              scatterRows,
              filterHits,
              values,
              numValues);
        } else {
          processRun<hasFilter, hasHook, scatter>(
              rows,
              rowIndex,
              currentRow,
              numInRun,
              scatterRows,
              filterHits,
              values,
              numValues,
              visitor);
        }
        remainingValues -= numAdvanced;
        currentRow += numAdvanced;
        rowIndex += numInRun;
        if (visitor.atEnd()) {
          visitor.setNumValues(hasFilter ? numValues : numAllRows);
          return;
        }
        if (remainingValues) {
          currentRow += remainingValues;
          skip<false>(remainingValues, -1, nullptr);
        }
      }
      readHeader();
    }
  }

  inline void readHeader() {
    signed char ch = IntDecoder<isSigned>::readByte();
    if (ch < 0) {
      remainingValues = static_cast<uint64_t>(-ch);
      repeating = false;
    } else {
      remainingValues = static_cast<uint64_t>(ch) + RLE_MINIMUM_REPEAT;
      repeating = true;
      value = IntDecoder<isSigned>::readLong();
    }
  }

  int8_t bitWidth_;
  uint64_t remainingValues_;
  int64_t value_;
  int8_t bitOffset_{0};
  bool repeating;
};

} // namespace facebook::velox::dwrf
