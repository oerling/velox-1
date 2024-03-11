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

  //  Size in bytes.
  size_t size;

  // Add members here to describe locations in storage for GPU direct transfer.
};

/// Describes how columns to be read together are staged on device. This is
/// anything from a set of host to device copies, GPU direct IO, or no-op if
/// data already on device.
class SplitStaging {
public:
  /// Adds a transfer described by 'staging'. Returns an id of the
  /// device side buffer. The id will be mapped to an actual buffer
  /// when the transfers are queud. At this time, pointers that
  /// are registered to the id are patched to the actual device side
  /// address.
  int32_t add(Staging& staging);

  /// Registers '*ptr' to be patched to the device side address of the transfer identified by 'id'. The *ptr is an offset into the buffer identified by id, so that the actual start of the area is added to the offset at *ptr.
  void registerPointer(int32_t id, int64_t* ptr);
  
  // Starts the transfers registered with add(). 'stream' is set to a stream
  // where operations depending on the transfer may be queued.
  void transfer(WaveStream& stream, Stream*& stream);

private:
  // Pinned host memory for transfer to device. May be nullptr if using unified
  // memory.
  WaveBufferPtr hostData_;

  // Device accessible memory (device or unified) with the data to read.
  WaveBufferPtr deviceData_;

  std::vector<Staging> staging_;
  // Offsets into device buffer for each id returned by add().
  std::vector<int64_t> offsets_;

  // List of pointers to patch to places inside deviceBuffer once this is allocated.
  std::vector<std::pair<int32_t>, int64_t*>> patch_;

  // Total device side space reserved so farr.
  int64_t fill_{0};
};

/// Operations on leaf columns. This is specialized for each file format.
class FormatData {
 public:
  static constexpr int32_t kStaged = 1;
  /// Mask indicating that one or more data transfers were added to staging. If set, whatever was added to program can must be enqueued on the same stream after the staging.
  static constexpr int32_t kStaged = 1;
  /// Mask indicating that work was added to a DecodeProgram. 
  static constexpr kQueued = 2;
  // Mask indicating that the column or RowSet from  'this' is ready after the queued work is ready. Dependent work can be enqueued on the stream of the program as soon as the program is launched.
  static constexpr int32_t kAllQueued = 4;
  
  virtual ~FormatData() = default;


  
  /// Adds the next read of the column. If the column is a filter depending on another filter, the previous filter is given on the first call. Returns an OR of flags describing the action. See kStaged, kQueued, kAllQueued. Allocates device and host buffers. These are owned by 'waveStream'.
  virtual int32_t  startRead(int32_t offset, RowSet rows, FormatData* previousFilter, SplitStaging& staging, DecodePrograms& program, WaveStream& waveStream) = 0;
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
