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

DEFINE_int32(cache_load_quantum, 8 << 20, "Max size of single IO to cache");
DEFINE_int32(
    cache_prefetch_min_pct,
    80,
    "Minimum percentage of actual uses over references to a column for prefetching. No prefetch if > 100");
DEFINE_int32(
    storage_max_coalesce_distance,
    1 << 20,
    "Max gap across wich IOs are coalesced for storage");
DEFINE_int32(
    ssd_max_coalesce_distance,
    (50 << 10),
    "Max gap across wich IOs are coalesced for SSD");
DEFINE_int32(
    coalesce_trace_count,
    0,
    "Number of upcoming coalesces to b traced to LOG(INFO)");
DEFINE_int32(
    max_coalesced_io_size,
    (16 << 20),
    "Maximum size of a single load for coalesced streams");

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
                    std::min<int32_t>(request.size, FLAGS_cache_load_quantum),
                    MappedMemory::kPageSize) /
        MappedMemory::kPageSize;
  }
  auto cachePages = cache_->incrementCachedPages(0);
  auto maxPages = cache_->maxBytes() / MappedMemory::kPageSize;
  auto allocatedPages = cache_->numAllocated();
  if (numPages < maxPages - allocatedPages) {
    // There is free space for the read-ahead.
    return true;
  }
  auto prefetchPages = cache_->incrementPrefetchPages(0);
  if (numPages + prefetchPages < cachePages / 2) {
    // The planned prefetch plus other prefetches are under half the cache.
    return true;
  }
  return false;
}

namespace {

std::vector<int32_t> makeReadPctBuckets() {
  return {80, 50, 20, 0};
}

const static std::vector<int32_t>& readPctBuckets() {
  static std::vector<int32_t> buckets = makeReadPctBuckets();
  return buckets;
}

bool isPrefetchPct(int32_t pct) {
  return pct >= FLAGS_cache_prefetch_min_pct;
}

std::vector<CacheRequest*> makeRequestParts(
    CacheRequest& request,
    const cache::TrackingData& trackingData,
    int32_t loadQuantum,
    std::vector<std::unique_ptr<CacheRequest>>& extraRequests) {
  if (request.size <= loadQuantum) {
    return {&request};
  }

  // Large columns will be part of coalesced reads if the access frequency
  // qualifies for read ahead and if over 80% of the column gets accessed. Large
  // metadata columns (empty no trackingData) always coalesce.
  auto readPct =
      (100 * trackingData.numReads) / (1 + trackingData.numReferences);
  auto readDensity =
      (100 * trackingData.readBytes) / (1 + trackingData.referencedBytes);
  bool prefetch = trackingData.referencedBytes > 0 &&
      (isPrefetchPct(readPct) && readDensity >= 80);
  std::vector<CacheRequest*> parts;
  for (uint64_t offset = 0; offset < request.size; offset += loadQuantum) {
    int32_t size = std::min<int32_t>(loadQuantum, request.size - offset);
    extraRequests.push_back(std::make_unique<CacheRequest>(
        RawFileCacheKey{request.key.fileNum, request.key.offset + offset},
        size,
        request.trackingId));
    parts.push_back(extraRequests.back().get());
    parts.back()->coalesces = prefetch;
  }
  return parts;
}
} // namespace

void CachedBufferedInput::load(const dwio::common::LogType) {
  std::vector<CacheRequest*> toLoad;
  // 'requests_ is cleared on exit.
  int32_t numNewLoads = 0;
  auto requests = std::move(requests_);
  cache::SsdFile* ssdFile =
      cache_->ssdCache() ? &cache_->ssdCache()->file(fileNum_) : nullptr;
  // Extra requests made for preloadable regions that are larger then
  // 'loadQuantum'.
  std::vector<std::unique_ptr<CacheRequest>> extraRequests;
  for (auto readPct : readPctBuckets()) {
    std::vector<CacheRequest*> storageLoad;
    std::vector<CacheRequest*> ssdLoad;
    for (auto& request : requests) {
      if (request.processed) {
        continue;
      }
      cache::TrackingData trackingData;
      if (!request.trackingId.empty()) {
        trackingData = tracker_->trackingData(request.trackingId);
      }
      if (request.trackingId.empty() ||
          (100 * trackingData.numReads) / (1 + trackingData.numReferences) >=
              readPct) {
        request.processed = true;
        auto parts = makeRequestParts(
            request, trackingData, FLAGS_cache_load_quantum, extraRequests);
        for (auto part : parts) {
          if (cache_->exists(part->key)) {
            continue;
          }
          if (ssdFile) {
            part->ssdPin = ssdFile->find(part->key);
            if (!part->ssdPin.empty()) {
              ssdLoad.push_back(part);
              continue;
            }
          }
          storageLoad.push_back(part);
        }
      }
    }
    makeLoads(std::move(storageLoad), isPrefetchPct(readPct));
    makeLoads(std::move(ssdLoad), isPrefetchPct(readPct));
  }
  if (!allFusedLoads_.empty() && FLAGS_coalesce_trace_count) {
    traceFusedLoads();
  }
}

void CachedBufferedInput::makeLoads(
    std::vector<CacheRequest*> requests,
    bool prefetch) {
  if (requests.size() < 2) {
    return;
  }
  bool isSsd = !requests[0]->ssdPin.empty();
  int32_t maxDistance = isSsd ? FLAGS_ssd_max_coalesce_distance
                              : FLAGS_storage_max_coalesce_distance;
  std::sort(
      requests.begin(),
      requests.end(),
      [&](const CacheRequest* left, const CacheRequest* right) {
        if (isSsd) {
          return left->ssdPin.run().offset() < right->ssdPin.run().offset();
        } else {
          return left->key.offset < right->key.offset;
        }
      });
  // Combine adjacent short reads.

  int32_t numNewLoads = 0;

  coalesceIo<CacheRequest*, CacheRequest*>(
      requests,
      maxDistance,
      // Break batches up. Better load more short ones i parallel.
      40,
      [&](int32_t index) {
        return isSsd ? requests[index]->ssdPin.run().offset()
                     : requests[index]->key.offset;
      },
      [&](int32_t index) { return requests[index]->size; },
      [&](int32_t index) {
        return requests[index]->coalesces ? 1 : kNoCoalesce;
      },
      [&](CacheRequest* request, std::vector<CacheRequest*>& ranges) {
        ranges.push_back(request);
      },
      [&](int32_t /*gap*/, std::vector<CacheRequest*> /*ranges*/) { /*no op*/ },
      [&](const std::vector<CacheRequest*>& /*requests*/,
          int32_t /*begin*/,
          int32_t /*end*/,
          uint64_t /*offset*/,
          const std::vector<CacheRequest*>& ranges) {
        ++numNewLoads;
        readRegion(ranges, prefetch);
      });
  if (prefetch && executor_ && (isSpeculative_ || numNewLoads > 1)) {
    for (auto& load : allFusedLoads_) {
      if (load->state() == LoadState::kPlanned) {
        executor_->add(
            [pendingLoad = load]() { pendingLoad->loadOrFuture(nullptr); });
      }
    }
  }
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
    VELOX_CHECK(toLoad_.empty());
    for (auto& request : requests_) {
      request.pin = cache_->findOrCreate(request.key, request.size, nullptr);
      if (request.pin.empty()) {
        // Already loading for another thread.
        continue;
      }
      if (request.pin.entry()->isExclusive()) {
        // A new entry to be filled.
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

  std::string toString() const override {
    int32_t payload = 0;
    int32_t total = requests_.back().key.offset + requests_.back().size -
        requests_[0].key.offset;
    for (auto& request : requests_) {
      payload += request.size;
    }
    return fmt::format(
        "<FusedLoad: {} entries, {} total {} extra>",
        requests_.size(),
        total,
        total - payload);
  }

 protected:
  void updateStats(const CoalesceIoStats& stats, bool isPrefetch, bool isSsd) {
    if (ioStats_) {
      ioStats_->incRawOverreadBytes(stats.extraBytes);
      if (isSsd) {
        ioStats_->ssdRead().increment(stats.payloadBytes);
      } else {
        // Reading the file increments rawReadBytes. Reverse this
        // increment here because actually accessing the data via
        // CacheInputStream will do the increment.
        ioStats_->incRawBytesRead(-stats.payloadBytes);

        ioStats_->read().increment(stats.payloadBytes);
      }
      if (isPrefetch) {
        ioStats_->prefetch().increment(stats.payloadBytes);
      }
    }
  }

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
    auto stats = cache::readPins(
        pins_,
        100000,
        1000,
        [&](int32_t i) { return pins_[i].entry()->offset(); },
        [&](const std::vector<CachePin>& pins,
            int32_t begin,
            int32_t end,
            uint64_t offset,
            const std::vector<folly::Range<char*>>& buffers) {
          stream.read(buffers, offset, dwio::common::LogType::FILE);
        });
    updateStats(stats, isPrefetch, false);
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
    std::vector<SsdPin> ssdPins;
    for (auto& request : toLoad_) {
      ssdPins.push_back(std::move(request->ssdPin));
    }
    auto stats = ssdPins[0].file()->load(ssdPins, pins_);
    updateStats(stats, isPrefetch, true);
  }
};

//} // namespace

void CachedBufferedInput::readRegion(
    std::vector<CacheRequest*> requests,
    bool prefetch) {
  if (requests.empty() || (requests.size() == 1 && !prefetch)) {
    return;
  }
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

void CachedBufferedInput::traceFusedLoads() {
  auto counter = FLAGS_coalesce_trace_count;
  FLAGS_coalesce_trace_count = std::min(counter - 1, 0);
  std::stringstream out;
  for (auto& load : allFusedLoads_) {
    out << load->toString() << " ";
  }
  LOG(INFO) << "Fused Loads: " << out.str();
}

} // namespace facebook::velox::dwrf
