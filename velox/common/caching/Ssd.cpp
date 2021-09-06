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
  file_->unpinRegion(run_.offset());
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
  if (entry.tinyData()) {
    load(run, entry.tinyData());
  } else {
    load(run, entry.data());
  }
  entry.setSsdFile(this, regionIndex(run.offset()));
}

void SsdFile::load(SsdRun run, char* data) {
  regionScore_[regionIndex(run.offset())] += run.size();
  auto rc = pread(fd_, data, run.size(), run.offset());
  VELOX_CHECK_EQ(rc, run.size());
}

void SsdFile::load(SsdRun ssdRun, memory::MappedMemory::Allocation& data) {
  regionScore_[regionIndex(ssdRun.offset())] += ssdRun.size();
  std::vector<struct iovec> iovecs;
  iovecs.reserve(data.numRuns());
  int64_t bytesLeft = ssdRun.size();
  for (auto i = 0; i < data.numPages(); ++i) {
    auto run = data.runAt(i);
    iovecs.push_back(
        {run.data<char>(), std::min<int64_t>(bytesLeft, run.numBytes())});
    bytesLeft -= run.numBytes();
    if (bytesLeft <= 0) {
      break;
    };
  }

  auto rc = folly::preadv(fd_, iovecs.data(), iovecs.size(), ssdRun.offset());
  VELOX_CHECK_EQ(rc, ssdRun.size());
}

  std::pair<uint64_t, int32_t> SsdFile::getSpace(
      const std::vector<CachePin>& pins,
      int32_t begin) {
  std::lock_guard<std::mutex> l(mutex_);
  evictLocked();
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
