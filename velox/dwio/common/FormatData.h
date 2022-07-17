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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeWithId.h"

namespace facebook::velox::dwio::common {

// Interface base class for format-specific state in common between all types of
// readers.
class FormatData {
 public:
  virtual ~FormatData() = default;

  template <typename T>
  T& as() {
    return *reinterpret_cast<T*>(this);
  }

  virtual void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      BufferPtr& nulls) = 0;

  virtual uint64_t skipNulls(uint64_t numValues) = 0;

  virtual uint64_t skip(uint64_t numValues) = 0;

  // True if 'this' may produce a null. true does not imply the
  // existence of an actual null, though. For example, if the format
  // is ORC and there is a nulls decoder for the column this returns
  // true. False if it is certain there is no null in the range of
  // 'this'. False also if nulls are not managed by this, e.g. in
  // Parquet.
  virtual bool hasNulls() const = 0;

  // Seeks the position to the 'index'th row group for the streams
  // managed by 'this'. Returns a PositionProvider for streams not
  // managed by 'this'. In a format like Parquet where all the reading
  // is in FormatData the provider is at end. For ORC/DWRF the type
  // dependent stream positions are accessed via the provider. The
  // provider is valid until next call of this.
  virtual dwio::common::PositionProvider seekToRowGroup(uint32_t index) = 0;

  virtual std::vector<uint32_t> filterRowGroups(
      const velox::common::ScanSpec& scanSpec,
      uint64_t rowsPerRowGroup,
      const StatsContext& context) = 0;
};

// Base class for format-specific reader initialization arguments.
class FormatParams {
 public:
  explicit FormatParams(memory::MemoryPool& pool) : pool_(pool) {}

  virtual ~FormatParams() = default;

  // Makes format-specific structures for the column of 'type'.
  virtual std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec) = 0;

  memory::MemoryPool& pool() {
    return pool_;
  }

 private:
  memory::MemoryPool& pool_;
};

} // namespace facebook::velox::dwio::common
