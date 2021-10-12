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

#include "velox/dwio/dwrf/common/CachedBufferedInput.h"
#include "velox/dwio/dwrf/common/CacheInputStream.h"

namespace facebook::velox::dwrf {

using cache::CachePin;
using cache::LoadState;
using cache::RawFileCacheKey;
using cache::ScanTracker;
using cache::SsdFile;
using cache::SsdPin;
using cache::TrackingId;
using memory::MappedMemory;

std::unique_ptr<SeekableInputStream> CachedBufferedInput::enqueue(
    dwio::common::Region region,
    const StreamIdentifier* si = nullptr) {
  if (region.length == 0) {
    return std::make_unique<SeekableArrayInputStream>(
        static_cast<const char*>(nullptr), 0);
  }

  TrackingId id;
  if (si) {
    id = TrackingId(si->node, si->kind);
  }
  requests_.emplace_back(
      RawFileCacheKey{fileNum_, region.offset}, region.length, id);
  tracker_->recordReference(id, region.length, fileNum_, groupId_);
  auto stream = std::make_unique<CacheInputStream>(
						   this, ioStats_.get(), region, input_, fileNum_, tracker_, id, groupId_);
  requests_.back().stream = stream.get();
  return stream;
}

bool CachedBufferedInput::isBuffered(uint64_t /*offset*/, uint64_t /*length*/)
    const {
  return false;
}

bool CachedBufferedInput::shouldPreload() {
  // True if after scheduling this for preload, half the capacity
  // would be in a loading but not yet accessed state.
  if (requests_.empty()) {
    return false;
  }
  int32_t numPages = 0;
  for (auto& request : requests_) {
    numPages += bits::roundUp(
                    std::min<int32_t>(
                        request.size, CacheInputStream::kDefaultLoadQuantum),
                    MappedMemory::kPageSize) /
        MappedMemory::kPageSize;
  }
  auto cachePages = cache_->incrementCachedPages(0);
  auto maxPages = cache_->maxBytes() / MappedMemory::kPageSize;
  auto allocatedPages = cache_->numAllocated();
  if (numPages < maxPages - allocatedPages) {
    return true;
  }
  auto prefetchPages = cache_->incrementPrefetchPages(0);
  if (numPages + prefetchPages < cachePages / 2) {
    return true;
  }
  return false;
}

void CachedBufferedInput::load(const dwio::common::LogType) {
  static std::vector<int32_t> readPctBuckets = {80, 50, 10, 0};
  std::vector<CacheRequest*> toLoad;
  // 'requests_ is cleared on exit.
  int32_t numNewLoads = 0;
  auto requests = std::move(requests_);
  cache::SsdFile* ssdFile =
      cache_->ssdCache() ? &cache_->ssdCache()->file(fileNum_) : nullptr;
  for (auto readPct : readPctBuckets) {
    std::vector<CacheRequest*> storageLoad;
    std::vector<CacheRequest*> ssdLoad;
    for (auto& request : requests) {
      if (!request.processed && request.trackingId.empty() ||
          tracker_->readPct(request.trackingId) >= readPct) {
        request.processed = true;
        if (cache_->exists(request.key)) {
          continue;
        }
        if (ssdFile) {
          request.ssdPin = ssdFile->find(request.key);
          if (!request.ssdPin.empty()) {
            ssdLoad.push_back(&request);
            continue;
          }
        }
        storageLoad.push_back(&request);
      }
    }
    makeLoads(std::move(storageLoad), readPct == readPctBuckets[0]);
    makeLoads(std::move(ssdLoad), readPct == readPctBuckets[0]);
  }
}

void CachedBufferedInput::makeLoads(
    std::vector<CacheRequest*> requests,
    bool schedule) {
  if (requests.empty()) {
    return;
  }

  int32_t maxDistance = requests[0]->ssdPin.empty() ? kMaxMergeDistance : 10000;
  std::sort(
      requests.begin(),
      requests.end(),
      [&](const CacheRequest* left, const CacheRequest* right) {
        if (left->ssdPin.empty()) {
          return left->key.offset < right->key.offset;
        }
        return left->ssdPin.run().offset() < right->ssdPin.run().offset();
      });
  // Combine adjacent short reads.
  dwio::common::Region last = {0, 0};

  std::vector<CacheRequest*> readBatch;
  int32_t numNewLoads = 0;
  for (const auto& request : requests) {
    dwio::common::Region entryRegion{
        static_cast<uint64_t>(
            request->ssdPin.empty() ? request->key.offset
                                    : request->ssdPin.run().offset()),
        static_cast<uint64_t>(request->size)};

    VELOX_CHECK_LT(0, entryRegion.length);
    if (last.length == 0) {
      // first region
      last = entryRegion;
    } else if (!tryMerge(last, entryRegion, maxDistance)) {
      ++numNewLoads;
      readRegion(std::move(readBatch));
      last = entryRegion;
    }
    readBatch.push_back(request);
  }
  ++numNewLoads;
  readRegion(std::move(readBatch));
  if (executor_ && numNewLoads > 1) {
    for (auto& load : allFusedLoads_) {
      if (load->state() == LoadState::kPlanned) {
        executor_->add(
            [pendingLoad = load]() { pendingLoad->loadOrFuture(nullptr); });
      }
    }
  }
}

bool CachedBufferedInput::tryMerge(
    dwio::common::Region& first,
    const dwio::common::Region& second,
    int32_t maxDistance) {
  DWIO_ENSURE_GE(second.offset, first.offset, "regions should be sorted.");
  int64_t gap = second.offset - first.offset - first.length;
  if (gap < 0) {
    // We do not support one region going to two target buffers.
    return false;
  }
  // compare with 0 since it's comparison in different types
  if (gap <= maxDistance) {
    int64_t extension = gap + second.length;

    if (extension > 0) {
      first.length += extension;
      if (gap > 0) {
        ioStats_->incRawOverreadBytes(gap);
        if (input_.getStats() != nullptr) {
          input_.getStats()->incRawOverreadBytes(gap);
        }
      }
    }

    return true;
  }

  return false;
}

// namespace {
class DwrfFusedLoadBase : public cache::FusedLoad {
 public:
  DwrfFusedLoadBase(
      cache::AsyncDataCache* cache,
      std::shared_ptr<dwio::common::IoStatistics> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : cache_(cache), ioStats_(std::move(ioStats)), groupId_(groupId) {
    for (auto& request : requests) {
      requests_.push_back(std::move(*request));
    }
  }

  const std::vector<CacheRequest>& requests() {
    return requests_;
  }

  bool makePins() override {
    for (auto& request : requests_) {
      request.pin = cache_->findOrCreate(request.key, request.size, nullptr);
      if (request.pin.empty()) {
        // Already loading for another thread.
        continue;
      }
      if (request.pin.entry()->isExclusive()) {
        // A new entry to be filled.
        if (false /*prefetch_*/) {
          request.pin.entry()->setPrefetch();
        }
        request.pin.entry()->setTrackingId(request.trackingId);
        request.pin.entry()->setGroupId(groupId_);
        toLoad_.push_back(&request);
      } else {
        // Already in cache, access time is refreshed.
        request.pin.clear();
      }
    }
    std::vector<CachePin> pins;
    for (auto* request : toLoad_) {
      pins.push_back(std::move(request->pin));
    }
    initialize(std::move(pins));
    return !toLoad_.empty();
  }

 protected:
  cache::AsyncDataCache* const cache_;
  std::vector<CacheRequest> requests_;
  std::vector<CacheRequest*> toLoad_;
  std::shared_ptr<dwio::common::IoStatistics> ioStats_;
  const uint64_t groupId_;
};

class DwrfFusedLoad : public DwrfFusedLoadBase {
 public:
  DwrfFusedLoad(
      cache::AsyncDataCache* cache,
      std::unique_ptr<AbstractInputStreamHolder> input,
      std::shared_ptr<dwio::common::IoStatistics> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : DwrfFusedLoadBase(cache, ioStats, groupId, std::move(requests)),
        input_(std::move(input)) {}

  void loadData(bool isPrefetch) override {
    auto& stream = input_->get();
    std::vector<folly::Range<char*>> buffers;
    uint64_t start = pins_[0].entry()->offset();
    uint64_t lastOffset = start;
    uint64_t totalRead = 0;
    for (auto& pin : pins_) {
      auto& buffer = pin.entry()->data();
      uint64_t startOffset = pin.entry()->offset();
      totalRead += pin.entry()->size();
      if (lastOffset < startOffset) {
        buffers.push_back(
            folly::Range<char*>(nullptr, startOffset - lastOffset));
      }

      auto size = pin.entry()->size();
      uint64_t offsetInRuns = 0;
      if (buffer.numPages() == 0) {
        buffers.push_back(folly::Range<char*>(pin.entry()->tinyData(), size));
        offsetInRuns = size;
      } else {
        for (int i = 0; i < buffer.numRuns(); ++i) {
          velox::memory::MappedMemory::PageRun run = buffer.runAt(i);
          uint64_t bytes = run.numBytes();
          uint64_t readSize = std::min(bytes, size - offsetInRuns);
          buffers.push_back(folly::Range<char*>(run.data<char>(), readSize));
          offsetInRuns += readSize;
        }
      }
      DWIO_ENSURE(offsetInRuns == size);
      lastOffset = startOffset + size;
    }
    if (isPrefetch) {
      ioStats_->prefetch().increment(totalRead);
    }
    ioStats_->read().increment(totalRead);

    stream.read(buffers, start, dwio::common::LogType::FILE);
  }

  std::unique_ptr<AbstractInputStreamHolder> input_;
};

class SsdLoad : public DwrfFusedLoadBase {
 public:
  SsdLoad(
      cache::AsyncDataCache* cache,
      std::shared_ptr<dwio::common::IoStatistics> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : DwrfFusedLoadBase(cache, ioStats, groupId, std::move(requests)) {}

  void loadData(bool isPrefetch) override {
    uint64_t start = toLoad_[0]->ssdPin.run().offset();
    uint64_t lastOffset = start;
    uint64_t totalRead = 0;
    std::vector<folly::Range<char*>> buffers;
    for (auto i = 0; i < pins_.size(); ++i) {
      auto& pin = pins_[i];
      auto& buffer = pin.entry()->data();
      uint64_t startOffset = toLoad_[i]->ssdPin.run().offset();
      totalRead += pin.entry()->size();
      if (lastOffset < startOffset) {
        buffers.push_back(
            folly::Range<char*>(nullptr, startOffset - lastOffset));
      }

      auto size = pin.entry()->size();
      uint64_t offsetInRuns = 0;
      if (buffer.numPages() == 0) {
        buffers.push_back(folly::Range<char*>(pin.entry()->tinyData(), size));
        offsetInRuns = size;
      } else {
        for (int i = 0; i < buffer.numRuns(); ++i) {
          velox::memory::MappedMemory::PageRun run = buffer.runAt(i);
          uint64_t bytes = run.numBytes();
          uint64_t readSize = std::min(bytes, size - offsetInRuns);
          buffers.push_back(folly::Range<char*>(run.data<char>(), readSize));
          offsetInRuns += readSize;
        }
      }
      DWIO_ENSURE(offsetInRuns == size);
      lastOffset = startOffset + size;
    }
    if (isPrefetch) {
      ioStats_->prefetch().increment(totalRead);
    }
    ioStats_->ssdRead().increment(totalRead);

    toLoad_[0]->ssdPin.file()->read(start, buffers);
  }
};

//} // namespace

void CachedBufferedInput::readRegion(std::vector<CacheRequest*> requests) {
  std::shared_ptr<cache::FusedLoad> load;
  if (!requests[0]->ssdPin.empty()) {
    load = std::make_shared<SsdLoad>(cache_, ioStats_, groupId_, requests);
  } else {
    load = std::make_shared<DwrfFusedLoad>(
					   cache_, streamSource_(), ioStats_, groupId_, requests);
  }
  allFusedLoads_.push_back(load);
  fusedLoads_.withWLock([&](auto& loads) {
    for (auto& request : requests) {
      loads[request->stream] = load;
    }
  });
}

std::shared_ptr<cache::FusedLoad> CachedBufferedInput::fusedLoad(
    const SeekableInputStream* stream) {
  return fusedLoads_.withWLock(
      [&](auto& loads) -> std::shared_ptr<cache::FusedLoad> {
        auto it = loads.find(stream);
        if (it == loads.end()) {
          return nullptr;
        }
        auto load = std::move(it->second);
        auto dwrfLoad = dynamic_cast<DwrfFusedLoadBase*>(load.get());
        for (auto& request : dwrfLoad->requests()) {
          loads.erase(request.stream);
        }
        return load;
      });
}

std::unique_ptr<SeekableInputStream> CachedBufferedInput::read(
    uint64_t offset,
    uint64_t length,
    dwio::common::LogType /*logType*/) const {
  return std::make_unique<CacheInputStream>(
					    const_cast<CachedBufferedInput*>(this),
      ioStats_.get(),
      dwio::common::Region{offset, length},
      input_,
      fileNum_,
      nullptr,
      TrackingId(),
      0);
}

} // namespace facebook::velox::dwrf
