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

#include "velox/common/base/Bloom.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/StringIdMap.h"
#include "velox/common/caching/FileIds.h"

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

namespace facebook::velox::cache {

class ApproxCounter {
 public:
  ApproxCounter(int32_t range) : bits_(std::max(2, range / 8)) {}

  void add(uint64_t value) {
    auto hash = folly::hasher<uint64_t>()(value);
    if (!Bloom::test(bits_.data(), bits_.size(), hash)) {
      Bloom::set(bits_.data(), bits_.size(), hash);
      ++count_;
    }
  }

  int32_t count() const {
    return count_;
  }

 private:
  std::vector<uint64_t> bits_;
  int32_t count_;
};

struct ReadCounts {
  uint64_t referenceBytes;
  uint64_t readBytes;
  uint32_t readCount;
  uint32_t referenceCount;
};

// Represents a groupId, column and its size and score.
struct SsdScore {
  //
  float score;

  // Expected size in bytes for caching to SSD
  float size;

  // Represents the groupId and TrackingId of the group, column pair.
  uint64_t hash;
};

class GroupTracker {
 public:
  static constexpr int32_t kExpectedNumFiles = 100;

  GroupTracker(const StringIdLease& name)
      : name_(name), numFiles_(kExpectedNumFiles) {}

  void recordFile(uint64_t fileId, int32_t numStripes);

  void recordReference(uint64_t fileId, TrackingId trackingId, int32_t bytes);
  void recordRead(uint64_t fileId, TrackingId trackingId, int32_t bytes);

  void addColumnScores(std::vector<SsdScore>& scores) const;
  
 private:
  StringIdLease name_;
  std::mutex mutex_;

  //
  folly::F14FastMap<TrackingId, TrackingData> columns_;
  ApproxCounter numFiles_;

  uint64_t numOpens_{0};
  uint64_t numOpenStripes_{0};

};

// Singleton for  keeping track of file groups.
class GroupStats {
 public:
  void recordFile(uint64_t fileId, uint64_t groupId, int32_t numStripes);

  void recordReference(
      uint64_t fileId,
      uint64_t groupId,
      TrackingId id,
      int32_t bytes);

  void recordRead(
      uint64_t fileId,
      uint64_t groupId,
      TrackingId trackingId,
      int32_t bytes);

  static GroupStats& instance();

  // Returns true if groupId, trackingId qualify the data to be cached to SSD.
  bool shouldSaveToSsd(uint64_t groupId, TrackingId trackingId) const;

  // Updates the SSD selection criteria. The group. trackingId pairs
  // that account for the top 'ssdSize' bytes of reported IO are
  // selected.
  void updateSsdFilter(uint64_t ssdSize);

private:
  void decay();
  GroupTracker& group(uint64_t id) {
    auto it = groups_.find(id);
    if (it == groups_.end()) {
      groups_[id] = std::make_unique<GroupTracker>(StringIdLease(fileIds(), id));
      return *groups_[id];
    }
    return *it->second;
  }
  
  std::mutex mutex_;
  folly::F14FastMap<uint64_t, std::unique_ptr<GroupTracker>> groups_;
  // Bloom filter of groupId, trackingId hashes for streams that should be saved
  // to SSD.
  std::vector<uint64_t> saveToSsd_;
};

} // namespace facebook::velox::cache
