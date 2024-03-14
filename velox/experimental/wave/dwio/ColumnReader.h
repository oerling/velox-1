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

#include "velox/dwio/common/TypeWithId.h"
#include "velox/experimental/wave/dwio/FormatData.h"
#include "velox/experimental/wave/exec/Wave.h"

namespace facebook::velox::wave {

class ReadStream;
class StructColumnReader;

/// dwio::SelectiveColumnReader for Wave
class ColumnReader {
 public:
  ColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      OperandId operand,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : requestedType_(requestedType),
        fileType_(fileType),
        operand_(operand),
        formatData_(params.toFormatData(fileType_, scanSpec, operand)),
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
  int32_t numRowsRemaining() const;

  const common::ScanSpec& scanSpec() const {
    return *scanSpec_;
  }
  const std::vector<ColumnReader*> children() const {
    return children_;
  }
  OperandId operand() const {
    return operand_;
  }

 protected:
  TypePtr requestedType_;
  std::shared_ptr<const dwio::common::TypeWithId> fileType_;
  const OperandId operand_;
  std::unique_ptr<FormatData> formatData_;
  // Specification of filters, value extraction, pruning etc. The
  // spec is assigned at construction and the contents may change at
  // run time based on adaptation. Owned by caller.
  velox::common::ScanSpec* scanSpec_;

  std::vector<ColumnReader*> children_;

  // Row number after last read row, relative to the ORC stripe or Parquet
  // Rowgroup start.
  vector_size_t readOffset_ = 0;
};

// Specifies an action on a column. A column is not indivisible. It
// has parts and another column's decode may depend on one part of
// another column but not another., e.g. a child of a nullable struct
// needs the nulls of the struct but no other parts to decode.
enum class ColumnAction { kNulls, kFilter, kLengths, kValues };

/// A generic description of a decode step. The actual steps are
/// provided by FormatData specializations but this captures
/// dependences, e.g. filters before non-filters, nulls and lengths
/// of repeated containers before decoding the values. A dependency
/// can be device side only or may need host decision. Items that
/// depend device side can be made into consecutive decode ops in
/// one kernel launch or can be in consecutively queued
/// kernels. dependences which need host require the prerequisite
/// kernel to ship data to host, which will sync on the stream and
/// only then may schedule the dependents in another kernel.
struct ColumnOp {
  static constexpr int32_t kNoPrerequisite = -1;
  static constexpr int32_t kNoOperand = -1;
  // Is the column fully decoded after this? If so, any dependent action can be
  // queued as soon as this is set.
  bool isFinal;
  // True if has a host side result. A dependent cannot start until the kernel
  // of this arrives and the host processes the result.
  bool hasResult;
  OperandId producesOperand{kNoOperand};
  // Index of another op in column ops array in ReadStream.
  int32_t prerequisite{kNoPrerequisite};
  ColumnAction action;
  ColumnReader* reader;
  // Vector completed by arrival of this. nullptr if no vector.
  WaveVectorPtr waveVector_;
  // Host side result size. 0 for unconditional decoding. Can be buffer size for
  // passing rows, length/offset array etc.
  int32_t resultSize_{0};

  // Device side non-vector result, like set of passing rows, array of
  // lengths/starts etc.
  int32_t* deviceResult{nullptr};
  int32_t* hostResult{nullptr};
};

class ReadStream : Executable {
 public:
  ReadStream(
      StructColumnReader* columnReader,
      vector_size_t offset,
      RowSet rows,
      WaveStream& waveStream,
      const OperandSet* firstColumns = nullptr);

 private:
  StructColumnReader* reader_;
  std::vector<ColumnOp> ops_;
};

} // namespace facebook::velox::wave
