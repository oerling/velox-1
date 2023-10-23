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

#include <folly/executors/QueuedImmediateExecutor.h>

#include "velox/common/process/TraceContext.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/CoalescedInputStream.h"
#include "velox/dwio/common/SelectiveBufferedInput.h"

using ::facebook::velox::common::Region;

namespace facebook::velox::dwio::common {

using velox::cache::ScanTracker;
using velox::cache::TrackingId;
using velox::memory::MemoryAllocator;

CoalescedInputStream::CoalescedInputStream(
    SelectiveBufferedInput* bufferedInput,
    IoStatistics* ioStats,
    const Region& region,
    std::shared_ptr<ReadFileInputStream> input,
    uint64_t fileNum,
    std::shared_ptr<ScanTracker> tracker,
    TrackingId trackingId,
    uint64_t groupId,
    int32_t loadQuantum)
    : bufferedInput_(bufferedInput),
      ioStats_(ioStats),
      input_(std::move(input)),
      region_(region),
      fileNum_(fileNum),
      tracker_(std::move(tracker)),
      trackingId_(trackingId),
      groupId_(groupId),
      loadQuantum_(loadQuantum) {}

bool CoalescedInputStream::Next(const void** buffer, int32_t* size) {
  if (position_ >= region_.length) {
    *size = 0;
    return false;
  }
  loadPosition();

  *buffer = reinterpret_cast<const void**>(run_ + offsetInRun_);
  *size = runSize_ - offsetInRun_;
  if (position_ + *size > region_.length) {
    *size = region_.length - position_;
  }
  offsetInRun_ += *size;
  position_ += *size;

  if (tracker_) {
    tracker_->recordRead(trackingId_, *size, fileNum_, groupId_);
  }
  return true;
}

void CoalescedInputStream::BackUp(int32_t count) {
  DWIO_ENSURE_GE(count, 0, "can't backup negative distances");

  uint64_t unsignedCount = static_cast<uint64_t>(count);
  DWIO_ENSURE(unsignedCount <= offsetInRun_, "Can't backup that much!");
  position_ -= unsignedCount;
}

bool CoalescedInputStream::Skip(int32_t count) {
  if (count < 0) {
    return false;
  }
  uint64_t unsignedCount = static_cast<uint64_t>(count);
  if (unsignedCount + position_ <= region_.length) {
    position_ += unsignedCount;
    return true;
  }
  position_ = region_.length;
  return false;
}

google::protobuf::int64 CoalescedInputStream::ByteCount() const {
  return static_cast<google::protobuf::int64>(position_);
}

void CoalescedInputStream::seekToPosition(PositionProvider& seekPosition) {
  position_ = seekPosition.next();
}

std::string CoalescedInputStream::getName() const {
  return fmt::format(
      "CoalescedInputStream {} of {}", position_, region_.length);
}

size_t CoalescedInputStream::positionSize() {
  // not compressed, so only need 1 position (uncompressed position)
  return 1;
}

namespace {
std::vector<folly::Range<char*>>
makeRanges(size_t size, memory::Allocation& data, std::string& tinyData) {
  std::vector<folly::Range<char*>> buffers;
  if (size > SelectiveBufferedInput::kTinySize) {
    buffers.reserve(data.numRuns());
    uint64_t offsetInRuns = 0;
    for (int i = 0; i < data.numRuns(); ++i) {
      auto run = data.runAt(i);
      uint64_t bytes = run.numPages() * memory::AllocationTraits::kPageSize;
      uint64_t readSize = std::min(bytes, size - offsetInRuns);
      buffers.push_back(folly::Range<char*>(run.data<char>(), readSize));
      offsetInRuns += readSize;
    }
  } else {
    buffers.push_back(folly::Range<char*>(tinyData.data(), size));
  }
  return buffers;
}
} // namespace

void CoalescedInputStream::loadSync() {
  if (region_.length < SelectiveBufferedInput::kTinySize) {
    tinyData_.resize(region_.length);
  } else {
    auto numPages = memory::AllocationTraits::numPages(loadedRegion_.length);
    if (numPages > data_.numPages()) {
      bufferedInput_->pool()->allocateNonContiguous(numPages, data_);
    }
  }

  process::TraceContext trace("CoalescedInputStream::loadSync");

  ioStats_->incRawBytesRead(loadedRegion_.length);
  auto ranges = makeRanges(loadedRegion_.length, data_, tinyData_);
  uint64_t usec = 0;
  {
    MicrosecondTimer timer(&usec);
    input_->read(ranges, loadedRegion_.offset, LogType::FILE);
  }
  ioStats_->read().increment(loadedRegion_.length);
  ioStats_->queryThreadIoLatency().increment(usec);
}

void CoalescedInputStream::loadPosition() {
  auto offset = region_.offset;
  if (!isLoaded_) {
    auto load = bufferedInput_->coalescedLoad(this);
    if (load) {
      folly::SemiFuture<bool> waitFuture(false);
      uint64_t usec = 0;
      {
        MicrosecondTimer timer(&usec);
        if (!load->loadOrFuture(&waitFuture)) {
          auto& exec = folly::QueuedImmediateExecutor::instance();
          std::move(waitFuture).via(&exec).wait();
        }
        loadedRegion_.offset = region_.offset;
        loadedRegion_.length = load->getData(region_.offset, data_, tinyData_);
        isLoaded_ = true;
      }
      ioStats_->queryThreadIoLatency().increment(usec);
    }
  }
  // Check if position outside of loaded bounds.
  if (region_.offset + position_ < loadedRegion_.offset ||
      region_.offset + position_ >=
          loadedRegion_.offset + loadedRegion_.length) {
    loadedRegion_.offset = region_.offset + position_;
    loadedRegion_.length = position_ + loadQuantum_ <= region_.length
        ? loadQuantum_
        : region_.length - position_;
    loadSync();
  }

  auto offsetInEntry = position_ - (loadedRegion_.offset - region_.offset);
  if (data_.numPages() == 0) {
    run_ = reinterpret_cast<uint8_t*>(tinyData_.data());
    runSize_ = tinyData_.size();
    offsetInRun_ = offsetInEntry;
    offsetOfRun_ = 0;
  } else {
    data_.findRun(offsetInEntry, &runIndex_, &offsetInRun_);
    offsetOfRun_ = offsetInEntry - offsetInRun_;
    auto run = data_.runAt(runIndex_);
    run_ = run.data();
    runSize_ = run.numPages() * memory::AllocationTraits::kPageSize;
    if (offsetOfRun_ + runSize_ > loadedRegion_.length) {
      runSize_ = loadedRegion_.length - offsetOfRun_;
    }
  }
}

} // namespace facebook::velox::dwio::common
