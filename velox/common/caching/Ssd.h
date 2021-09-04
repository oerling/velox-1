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
#include "velox/common/caching/ScanTracker.h"

namespace facebook::velox::cache {

// A 64 bit word describing a SSD cache entry. 32 files x 64G per file, up to
// 8MB in entry size for a total of 2TB. Larger capacities need more bits.
class SsdRun {
 public:
  SsdRun() : bits_(0) {}

  SsdRun(uint8_t file, uint64_t offset, uint32_t size)
      : bits_((offset << 23) | ((size - 1))) {
    VELOX_CHECK_LT(offset, 1L << 36); // 64G
    VELOX_CHECK_LT(size - 1, 1 << 23); // 8MB
  }
  uint64_t offset() const {
    return (bits_ >> 23);
  }

  uint32_t size() const {
    return bits_ & ((1 << 23) - 1);
  }

 private:
  const uint64_t bits_;
};

struct SsdKey {
  StringIdLease file;
  uint64_t offset;

  bool operator=(const SsdKey& other) const {
    return offset == other.offset && file.id() == other.file.id();
  }
};

} // namespace facebook::velox::cache

namespace std {
template <>
struct hash<::facebook::velox::cache::SsdKey> {
  size_t operator()(const ::facebook::velox::cache::SsdKey& key) const {
    return facebook::velox::bits::hashMix(key.file.id(), key.offset);
  }
};

} // namespace std

namespace facebook::velox::cache {

  class SsdFile;

// Represents an SsdFile entry that is planned for load or being loaded. This
// is destroyed after load.
class SsdPin {
 public:
  SsdPin() : file_(nullptr) {}

  SsdPin(SsdFile& file, SsdRun run);

  ~SsdPin();
  bool empty() {
    return file_ == nullptr;
  }

 private:
  SsdFile* file_;
  SsdRun run_;
};

class SsdFile {
 public:
  static constexpr uint64_t kMaxSize = 1UL << 36; // 64G
  static constexpr uint64_t kRegionSize = 1 << 26; // 64MB
  static constexpr int32_t kNumRegions = kMaxSize / kRegionSize;

  // Constructs a cache backed by filename. Discards any previous
  // contents of filename.
  SsdFile(std::string_view filename);

  // Adds entries of  'pins'  to this file. 'pins' must be in read mode and
  // those pins that are successfully added to SSD are marked as being on SSD.
  // The file of the entries must be a file that is backed by 'this'.
  void store(std::vector<CachePin> pins);

  SsdPin find(RawFileCacheKey key);

  void load(SsdRun run, char* data);

  void load(SsdRun run, memory::MappedMemory::Allocation data);

  void pinRegion(uint64_t offset);
  void unpinRegion(uint64_t offset);

  int32_t regionIndex(uint64_t offset) {
    return offset / kRegionSize;
  }

  void regionUsed(int32_t region, int32_t size) {
    regionScore_[region] += size;
  }

 private:
  static constexpr int32_t kDecayInterval = 1000;
  std::mutex mutex_;
  // A bitmap where a 1 indicates a region that is in use. Entries that
  // refer to an in use region are readable.
  std::vector<uint64_t> activeRegions_;
  // Number of 64MB regions in the file.
  int32_t numRegions_{0};

  // Maximum size of the backing file in kRegionSize units.
  const int32_t maxRegions_;

  // Number of used bytes in in each region. A new entry must fit
  // between the offset and the end of the region.
  std::vector<uint32_t> regionSize_;

  // Count of KB used from the corresponding region. Decays with time.
  std::vector<uint64_t> regionScore_;

  std::vector<int32_t> regionPins_;

  // Map of file number and offset to location in file.
  folly::F14FastMap<SsdKey, SsdRun> entries_;

  // Count of reads and writes. The scores are decayed every time e count goes
  // over kDecayInterval or half 'entries_' size, wichever comes first.
  uint64_t numActions_{0};

  // File descriptor.
  int32_t fd_;
};

class SsdCache {
 public:
  static constexpr int32_t kNumShards = 32;
  SsdCache(std::string_view filePrefix, uint64_t maxSize);

  SsdFile& file(uint64_t fileId);

 private:
  std::string filePrefix_;
  std::vector<std::unique_ptr<SsdFile>> files_;
};

} // namespace facebook::velox::cache
