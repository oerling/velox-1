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

#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/SeekableInputStream.h"

namespace facebook::velox::dwio::common {

class SelectiveBufferedInput;

class CoalescedInputStream : public SeekableInputStream {
 public:
  CoalescedInputStream(
      SelectiveBufferedInput* bufferedInput,
      IoStatistics* ioStats,
      const velox::common::Region& region,
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t fileNum,
      std::shared_ptr<cache::ScanTracker> tracker,
      cache::TrackingId trackingId,
      uint64_t groupId,
      int32_t loadQuantum);

  bool Next(const void** data, int* size) override;
  void BackUp(int count) override;
  bool SkipInt64(int64_t count) override;
  google::protobuf::int64 ByteCount() const override;

  void seekToPosition(PositionProvider& position) override;
  std::string getName() const override;
  size_t positionSize() override;

  // Testing function to access loaded state.
  void testingData(
      velox::common::Region& loadedRegion,
      memory::Allocation*& data,
      std::string*& tinyData) {
    loadedRegion = loadedRegion_;
    data = &data_;
    tinyData = &tinyData_;
  }

 private:
  // Ensures that the current position is covered by 'data_'.
  void loadPosition();

  // Synchronously sets 'data_' to cover loadedRegion_'.
  void loadSync();

  SelectiveBufferedInput* const bufferedInput_;
  IoStatistics* ioStats_;
  std::shared_ptr<ReadFileInputStream> input_;
  // The region of 'input' 'this' ranges over.
  const velox::common::Region region_;
  const uint64_t fileNum_;
  std::shared_ptr<cache::ScanTracker> tracker_;
  const cache::TrackingId trackingId_;
  const uint64_t groupId_;

  velox::common::Region loadedRegion_;

  // Maximum number of bytes read from 'input' at a time.
  const int32_t loadQuantum_;

  // Allocation with loaded data. Has space for region.length or loadQuantum_
  // bytes, whichevr is less.
  memory::Allocation data_;

  // Contains the data if the range is too small for Allocation.
  std::string tinyData_;

  // Offset of current run from start of 'data_'
  uint64_t offsetOfRun_;

  // Pointer  to start of  current run in 'entry->data()' or
  // 'entry->tinyData()'.
  uint8_t* run_{nullptr};
  // Position of stream relative to 'run_'.
  int offsetInRun_{0};
  // Index of run in 'data_'
  int runIndex_ = -1;
  // Number of valid bytes above 'run_'.
  uint32_t runSize_ = 0;
  // Position relative to 'region_.offset'.
  uint64_t position_ = 0;

  //
  bool isLoaded_{false};
};

} // namespace facebook::velox::dwio::common
