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

namespace facebook::velox::wave {

// Describes how a column is staged on GPU, for example, copy from host RAM,
// direct read, already on device etc.
struct Staging {
  // Pointer to data in pageable host memory, if applicable.
  const void* hostData{nullptr};

  // Pointer to data in pinned host memory if applicable.
  const void* pinnedHostData{nullptr};
  //  Size in bytes.
  size_t size;

  // Add members here to describe locations in storage for GPU direct transfer.
};

/// Describes how columns to be read together are staged on device. This is
/// anything from a set of host to device copies, GPU direct IO, or no-op if
/// data already on device.
class SplitStaging {
  add(Staging& staging);

  // Starts the transfers registered with add(). 'stream' is set to a stream
  // where operations depending on the transfer may be queued.
  void transfer(WaveStream& stream, Stream*& stream);

  // Pinned host memory for transfer to device. May be nullptr if using unified
  // memory.
  WaveBufferPtr hostData_;

  // Device accessible memory (device or unified) with the data to read.
  WaveBufferPtr deviceData_;
  // Pointers to starts of streams inside deviceBuffer_.
  std::vector<char*> devicePtrs;
};

/// Operations on leaf columns. This is specialized for each file format.
class FormatData {
 public:
  virtual ~FormatData() = default;

  void addStaging(int32_t numRows, SplitStaging& staging) = 0;

  void schedule(int32_t numRows, FormatData* previousFilter = nullptr);
};

class FormatParams {
  explicit FormatParams(memory::MemoryPool& pool, ColumnReaderStatistics& stats)
      : pool_(pool), stats_(stats) {}

  virtual ~FormatParams() = default;

  /// Makes format-specific structures for the column given by  'type'.
  /// 'scanSpec' is given as extra context.
  virtual std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec) = 0;

  memory::MemoryPool& pool() {
    return pool_;
  }

  ColumnReaderStatistics& runtimeStatistics() {
    return stats_;
  }

 private:
  memory::MemoryPool& pool_;
  ColumnReaderStatistics& stats_;
};

}; // namespace facebook::velox::wave
}
