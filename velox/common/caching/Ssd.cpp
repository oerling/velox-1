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

#include "velox/common/caching/AsyncDataCache.h"

#include <folly/portability/SysUio.h>

       #include <sys/types.h>
       #include <sys/stat.h>
       #include <fcntl.h>

namespace facebook::velox::cache {

  SsdPin::SsdPin(SsdFile& file, SsdRun run)
    : file_(&file),
      run_(run) {
    file_->pinRegion(run_.offset());
  }

  SsdPin::~SsdPin() {
    file_->unpinRegion(run_.offset());
  }
  
  SsdFile::SsdFile(const std::string& filename, int32_t maxRegions)
    : maxRegions_(maxRegions) {

    fd_ = open(filename.c_str(), O_CREATE | O_RDWR,  S_IRUSR | S_IWUSR);
    if (fd_ < 0) {
      std::c_err << "Cannot open or create " << filename << " error " << errno << std::endl;
      exit(1);
    }
    uint64_t size = lseek(fd_, 0, SEEK_END);
    numRegions_ = size / kRegionSize;
    if (size % kRegionSize > 0) {
      ftruncate(fd_, numRegions_ * kRegionSize);
    }
    regionScore_.resize(kMaxRegions);
    regionSize_.resize(kMaxRegions);
    regionPins_.resize(kMaxRegions);

  }

  void SsdFile::pinRegion(uint64_t offset) {
    std::lock_guard<std::mutex> l(mutex_);
    ++regionPins_[regionIndex(offset)];

  }
  
  void unpinRegion(uint64_t offset) {
    std::lock_guard<std::mutex> l(mutex_);
    --regionPins_[regionIndex(offset)];
  }

  SsdPin SsdFile::find(RawFileCacheKey key) {
    SsdKey ssdKey{StringIdLease(fileIds(), key.fileId), key.offset};
    std::lock_guard<std::mutex> l(mutex_);
    newEvent();
    auto it = entries_.find(ssdKey);
    if (it == entries_.end()) {
      return SsdPin();
    }
    return SsdPin(this, it->second);
  }

  void SsdFile::newEvent() {
    ++numActions_;
    if (numActions > kDecayInterval && numActions_ > entries_.size() /2) {
      for (auto i = 0; i < numRegions_; ++i) {
	int64_t score = regionScore_[i];
	regionScore[i] = (score * 15) / 16;
      }
    }
  }
  
void SsdFile::load(SsdRun run, char* data) {
  regionScore_[regionIndex(run.offset())] += run.size();
  auto rc = pread(fd_, data, run.size(), run.offset()); 
  VELOX_CHECK(rc, run.size());
}

void SsdFile::load(SsdRun ssdRun, memory::MappedMemory::Allocation data) {
  regionScore_[regionIndex(run.offset())] += run.size();
  std::vector<struct iovec> iovecs;
  iovecs.reserve(data.numRuns());
  int64_t bytesLeft = ssdRun.size();
  for (auto i = 0; i < data.numPages(); ++i) {

    auto run = data.runAt(i);
    iovecs.push_back({run.data<char>(), std::min(run.numBytes)});
    bytesLeft -= run.numBytes();
    if (bytesLeft <= 0) {
      break;
    };
  }


  auto rc = folly::preadv(fd_, iovects.data(), iovecs.size(), ssdRun.offset());
  VELOX_CHECK_EQ(rc, ssdRun.size());
}

void SsdFile::store(std::vector<CachePin> pins) {
  uint64_t total = 0;
  for (auto& pin : pins) {
    total += pin.entry()->size();
  }
  
}
  
SsdCache::SsdCache(std::string_view filePrefix, uint64_t maxSize)
    : filePrefix_(filePrefix),
      maxSize_(bits::roundUp(maxSize, kNumShards * SsdFile::kRegionSize * 4)) {
  files_.reserve(kNumShards);
  for (auto i = 0; i < kNumShards; ++i) {
    files_.push_back(
        std::make_unique<SsdFile>(std::format("{}{}", filePrefix_, i)));
  }
}

SsdFile* SsdCache::file(uint64_t fileId) {
  auto index = fileId % kNumShards;
  return files_[index];
}

} // namespace facebook::velox::cache

