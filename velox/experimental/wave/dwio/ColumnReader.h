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

#include "velox/experimental/wave/exec/Wave.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/experimental/wave/dwio/FormatData.h"

namespace facebook::velox::wave {

class   ReadStream;
  
/// dwio::SelectiveColumnReader for Wave
class ColumnReader {
 public:
  ColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : memoryPool_(params.pool()),
        requestedType_(requestedType),
        fileType_(fileType),
        formatData_(params.toFormatData(fileType, scanSpec)),
        scanSpec_(&scanSpec) {}

  /// True if 'this' has a position from which a new ReadStream can be
  /// started.  This may be false if processing a previous ReadStream
  /// needs to advance before 'this' knows the starting position of
  /// the next read. For example, lengths of a previous ReadStream may
  /// have to be added up before a next read can start.
  virtual bool mayStartReadStream() const;
  
  /// Returns how many rows are left for which no read has been
  /// initiated. A read is initiated by making a ReadStream, which
  /// schedules a range of rows for reading. This may be called if
  /// mayStartReadStream() is true.
  int32_t numRowsRemaining()const;

  
 
protected:
  std::unique_ptr<FormatData> formatData_;
  // Specification of filters, value extraction, pruning etc. The
  // spec is assigned at construction and the contents may change at
  // run time based on adaptation. Owned by caller.
  velox::common::ScanSpec* FOLLY_NONNULL scanSpec_;

  // Row number after last read row, relative to the ORC stripe or Parquet
  // Rowgroup start.
  vector_size_t readOffset_ = 0;
};

  class ReadStream {
  public:
    ReadStream(ColumnReader* columnReader, vector_size_t offset, RowSet rows, OperandSet firstColumns);

    
    // Returns the vectors for Operands.
    void getVectors(folly::Range<Operand*> operands, WaveVectorPtr* vectors);
    
  };
  
} // namespace facebook::velox::wave
