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
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/StringIdMap.h"

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
  int32_t count_{0};
};

// Represents a groupId, column and its size and score. These are
// sorted and as many are selected from the top as will fit on SSD.
struct SsdScore {
  //
  float score;

  // Expected size in bytes for caching to SSD
  float size;
  // Recorded read activity, with older reads decayed.
  float readBytes;
  uint64_t groupId;
  int32_t columnId;
};

class GroupTracker {
 public:
  static constexpr int32_t kExpectedNumFiles = 100;

  GroupTracker(const StringIdLease& name)
      : name_(name), numFiles_(kExpectedNumFiles) {}

  void recordFile(uint64_t fileId, int32_t numStripes);

  void recordReference(uint64_t fileId, int32_t columnId, int32_t bytes);
  void recordRead(uint64_t fileId, int32_t columnId, int32_t bytes);

  // Adds the column scores to 'scores'. If 'decayPct' is non-0,
  // decays the recorded accesses by 'decayPct'% but at least by one
  // whole access.
  void addColumnScores(int32_t decayPct, std::vector<SsdScore>& scores);

  bool eraseColumn(int32_t columnId) {
    columns_.erase(columnId);
    return columns_.empty();
  }
    
 private:
  StringIdLease name_;
  std::mutex mutex_;

  //Map of column to access data.
  folly::F14FastMap<int32_t, TrackingData> columns_;

  // Count of distinct files seen in recordFile().
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
  // selected. If 'decayPct' is non-0, old stats are decayed and
  // removed if counts go to zero.
  void updateSsdFilter(uint64_t ssdSize, int32_t decayPct = 0);

  // Returns an estimate of the total size of the dataset based of the
  // groups, files and columns referenced to data.
  float dataSize() const {
    return dataSize_;
  }

  // Returns the percentage of historical reads that hit the currently SSD
  // cachable fraction of the data.
  float cachableReadPct() const {
    return cachableReadPct_;
  }

  // Returns percent of all seen data that fits in the SSD cachable fraction.
  float cachableDataPct() const {
    return cachableDataPct_;
  }

  // Clears the state to be as after default construction.
  void clear();

  // Recalculates the best groups and makes a human readable
  // summary. 'cacheBytes' is used to compute what fraction of the tracked
  // working set can be cached in 'cacheBytes'.
  std::string toString(uint64_t cacheBytes);

 private:
  GroupTracker& group(uint64_t id) {
    auto it = groups_.find(id);
    if (it == groups_.end()) {
      groups_[id] =
          std::make_unique<GroupTracker>(StringIdLease(fileIds(), id));
      return *groups_[id];
    }
    return *it->second;
  }

  // Returns the tracked group/column pairs best score first. Sets the
  // 'dataSize_', 'cachableReadPct_' and 'cachableDataPct_' according
  // to 'cacheBytes'. access counts by decayPct if decayPct% is
  // non-0. Trims away scores that fall to zero accesses by decay or
  // fall outside of the top FLAGS_max_group_stats top scores.
  std::vector<SsdScore> ssdScoresLocked(uint64_t cacheBytes, int32_t decayPct = 0);

  //  Removes the information on groupId/id.
  void eraseStatLocked(uint64_t groupId, int32_t columnId);

  std::mutex mutex_;
  folly::F14FastMap<uint64_t, std::unique_ptr<GroupTracker>> groups_;
  // Bloom filter of groupId, columnId hashes for streams that should be saved
  // to SSD.
  std::vector<uint64_t> saveToSsd_;
  bool allFitOnSsd_{false};
  double dataSize_{0};
  double totalRead_{0};
  float cachableDataPct_{0};
  float cachableReadPct_{0};
  
};

} // namespace facebook::velox::cache
