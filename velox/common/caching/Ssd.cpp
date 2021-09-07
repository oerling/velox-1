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

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/portability/SysUio.h>
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace facebook::velox::cache {

SsdPin::SsdPin(SsdFile& file, SsdRun run) : file_(&file), run_(run) {
  file_->pinRegion(run_.offset());
}

SsdPin::~SsdPin() {
  if (file_) {
    file_->unpinRegion(run_.offset());
  }
}

void SsdPin::operator=(SsdPin&& other) {
  if (file_) {
    file_->unpinRegion(run_.offset());
  }
  file_ = other.file_;
  other.file_ = nullptr;
  run_ = other.run_;
}

SsdFile::SsdFile(
    const std::string& filename,
    SsdCache& cache,
    int32_t ordinal,
    int32_t maxRegions)
    : cache_(cache), ordinal_(ordinal), maxRegions_(maxRegions) {
  fd_ = open(filename.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd_ < 0) {
    LOG(ERROR) << "Cannot open or create " << filename << " error " << errno
               << std::endl;
    exit(1);
  }
  uint64_t size = lseek(fd_, 0, SEEK_END);
  numRegions_ = size / kRegionSize;
  if (size % kRegionSize > 0) {
    ftruncate(fd_, numRegions_ * kRegionSize);
  }
  fileSize_ = numRegions_ * kRegionSize;
  regionScore_.resize(maxRegions_);
  regionSize_.resize(maxRegions_);
  regionPins_.resize(maxRegions_);
}

void SsdFile::pinRegion(uint64_t offset) {
  std::lock_guard<std::mutex> l(mutex_);
  ++regionPins_[regionIndex(offset)];
}

void SsdFile::unpinRegion(uint64_t offset) {
  std::lock_guard<std::mutex> l(mutex_);
  auto count = --regionPins_[regionIndex(offset)];
  if (suspended_ && count == 0) {
    evictLocked();
  }
}

namespace {
void addEntryToIovecs(AsyncDataCacheEntry& entry, std::vector<iovec>& iovecs) {
  if (entry.tinyData()) {
    iovecs.push_back({entry.tinyData(), entry.size()});
    return;
  }
  auto& data = entry.data();
  iovecs.reserve(iovecs.size() + data.numRuns());
  int64_t bytesLeft = entry.size();
  for (auto i = 0; i < data.numPages(); ++i) {
    auto run = data.runAt(i);
    iovecs.push_back(
        {run.data<char>(), std::min<int64_t>(bytesLeft, run.numBytes())});
    bytesLeft -= run.numBytes();
    if (bytesLeft <= 0) {
      break;
    };
  }
}
} // namespace

SsdPin SsdFile::find(RawFileCacheKey key) {
  SsdKey ssdKey{StringIdLease(fileIds(), key.fileNum), key.offset};
  std::lock_guard<std::mutex> l(mutex_);
  if (suspended_) {
    return SsdPin();
  }
  newEventLocked();
  auto it = entries_.find(ssdKey);
  if (it == entries_.end()) {
    return SsdPin();
  }
  return SsdPin(*this, it->second);
}

void SsdFile::newEventLocked() {
  ++numEvents_;
  if (numEvents_ > kDecayInterval && numEvents_ > entries_.size() / 2) {
    for (auto i = 0; i < numRegions_; ++i) {
      int64_t score = regionScore_[i];
      regionScore_[i] = (score * 15) / 16;
    }
  }
}

void SsdFile::load(SsdRun run, AsyncDataCacheEntry& entry) {
  VELOX_CHECK_EQ(run.size(), entry.size());
  regionScore_[regionIndex(run.offset())] += run.size();
  if (entry.tinyData()) {
    load(run, entry.tinyData());
  } else {
    std::vector<struct iovec> iovecs;
    addEntryToIovecs(entry, iovecs);

    auto rc = folly::preadv(fd_, iovecs.data(), iovecs.size(), run.offset());
    VELOX_CHECK_EQ(rc, run.size());
  }
  entry.setSsdFile(this, regionIndex(run.offset()));
}

void SsdFile::load(SsdRun run, char* data) {
  regionScore_[regionIndex(run.offset())] += run.size();
  auto rc = pread(fd_, data, run.size(), run.offset());
  VELOX_CHECK_EQ(rc, run.size());
}

std::pair<uint64_t, int32_t> SsdFile::getSpace(
    const std::vector<CachePin>& pins,
    int32_t begin) {
  std::lock_guard<std::mutex> l(mutex_);
  for (;;) {
    if (writableRegions_.empty()) {
      if (!evictLocked()) {
        return {0, 0};
      }
    }
    auto region = writableRegions_[0];
    auto offset = regionSize_[region];
    auto available = kRegionSize - regionSize_[region];
    int64_t toWrite = 0;
    for (; begin < pins.size(); ++begin) {
      auto entry = pins[begin].entry();
      if (entry->size() > available) {
        break;
      }
      available -= entry->size();
      toWrite += entry->size();
    }
    if (toWrite) {
      regionSize_[region] += toWrite;
      return {offset, toWrite};
    }
    writableRegions_.erase(writableRegions_.begin());
  }
}

bool SsdFile::evictLocked() {
  if (numRegions_ < maxRegions_) {
    writableRegions_.push_back(numRegions_);
    regionSize_[numRegions_ - 1] = 0;
    ++numRegions_;
    return true;
  }
  std::vector<int32_t> candidates;
  int64_t scoreSum = 0;
  for (int i = 0; i < numRegions_; ++i) {
    if (regionPins_[i]) {
      continue;
    }
    if (candidates.empty() || regionScore_[i] < scoreSum / candidates.size()) {
      scoreSum += regionScore_[i];
      candidates.push_back(i);
    }
  }
  if (candidates.empty()) {
    suspended_ = true;
    return false;
  }
  std::sort(
      candidates.begin(), candidates.end(), [&](int32_t left, int32_t right) {
        return regionScore_[left] < regionScore_[right];
      });
  if (candidates.size() > 3) {
    candidates.resize(3);
  }
  clearRegionEntriesLocked(candidates);
  suspended_ = false;
  return true;
}

void SsdFile::clearRegionEntriesLocked(const std::vector<int32_t>& toErase) {
  auto it = entries_.begin();
  while (it != entries_.end()) {
    auto region = regionIndex(it->second.offset());
    if (std::find(toErase.begin(), toErase.begin(), region) != toErase.end()) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void SsdFile::store(std::vector<CachePin> pins) {
  std::sort(pins.begin(), pins.end());
  uint64_t total = 0;
  for (auto& pin : pins) {
    total += pin.entry()->size();
  }
  int32_t storeIndex = 0;
  while (storeIndex < pins.size()) {
    auto [offset, available] = getSpace(pins, storeIndex);
    if (!available) {
      // No space can be reclaimed. Free the pins.
      for (int32_t i = storeIndex; i < pins.size(); ++i) {
        pins[i].clear();
      }
      return;
    }
    int32_t numWritten = 0;
    int32_t bytes = 0;
    std::vector<iovec> iovecs;
    for (auto i = storeIndex; i < pins.size(); ++i) {
      auto pin = std::move(pins[i]);
      pin.entry()->setSsdFile(this, offset);
      addEntryToIovecs(*pin.entry(), iovecs);
      bytes += pin.entry()->size();
      ++numWritten;
      if (bytes >= available) {
        break;
      }
    }
    if (offset > fileSize_) {
      ftruncate(fd_, offset);
      fileSize_ = offset;
    }
    int32_t rc = folly::pwritev(fd_, iovecs.data(), iovecs.size(), offset);
    if (rc != bytes) {
      LOG(ERROR) << "Failed to write to SSD " << errno;
      // Unpin without marking as resident in SSD.
      for (auto i = storeIndex; i < pins.size(); ++i) {
        pins[i].clear();
      }
      return;
    }
    if (offset == fileSize_) {
      fileSize_ += bytes;
    }
    {
      std::lock_guard<std::mutex> l(mutex_);
      for (auto i = storeIndex; i < storeIndex + numWritten; ++i) {
        auto entry = pins[i].entry();
        entry->setSsdFile(this, offset);
        auto size = entry->size();
        SsdKey key = {entry->key().fileNum, offset};
        entries_[std::move(key)] = SsdRun(offset, size);
        offset += size;
      }
    }
    // Unpin the written entries.
    for (auto i = storeIndex; i < storeIndex + numWritten; ++i) {
      pins[i].clear();
    }
    storeIndex += numWritten;
  }
}

SsdCache::SsdCache(std::string_view filePrefix, uint64_t maxBytes)
    : filePrefix_(filePrefix) {
  files_.reserve(kNumShards);
  constexpr uint64_t kSizeQuantum = kNumShards * SsdFile::kRegionSize;
  int32_t fileMaxRegions =
      bits::roundUp(maxBytes, kSizeQuantum) / kSizeQuantum / kNumShards;
  for (auto i = 0; i < kNumShards; ++i) {
    files_.push_back(std::make_unique<SsdFile>(
        fmt::format("{}{}", filePrefix_, i), *this, i, fileMaxRegions));
  }
}

SsdFile& SsdCache::file(uint64_t fileId) {
  auto index = fileId % kNumShards;
  return *files_[index];
}

namespace {
folly::IOThreadPoolExecutor* ssdStoreExecutor() {
  static auto executor = std::make_unique<folly::IOThreadPoolExecutor>(4);
  return executor.get();
}
} // namespace

bool SsdCache::startStore() {
  if (0 == storesInProgress_.fetch_add(kNumShards)) {
    return true;
  }
  storesInProgress_.fetch_sub(kNumShards);
  return false;
}

void SsdCache::store(std::vector<CachePin> pins) {
  std::vector<std::vector<CachePin>> shards(kNumShards);
  for (auto& pin : pins) {
    auto& target = file(pin.entry()->key().fileNum.id());
    shards[target.ordinal()].push_back(std::move(pin));
  }
  int32_t numNoStore = 0;
  for (auto i = 0; i < kNumShards; ++i) {
    if (shards[i].empty()) {
      ++numNoStore;
      continue;
    }
    ssdStoreExecutor()->add([this, i, pinsForShard = std::move(shards[i])]() {
      files_[i]->store(std::move(pinsForShard));
    });
  }
  storesInProgress_.fetch_sub(numNoStore);
}

} // namespace facebook::velox::cache
