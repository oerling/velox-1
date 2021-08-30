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

#include "velox/common/caching/ScanTracker.h"

namespace facebook::velox::cache {

class ApproxCounter {
 public:
  ApproxCounter(int32_t range) : bits_(std::max(2, range / 8)) {}

  void add(uint64_t value) {
    auto hash = folly::hasher<uint64_t>()(value);
    if (!bloom::test(bits, bits.size(), hash)) {
      Bloom::set(bits, bits.size(), hash);
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

class GroupTracker {
 public:
  GroupTracker(const StringIdLease& name) : name_(name){};

 private:
  StringIdLease name_;
  std::mutex mutex_;
  folly::F14FastMap<TrackingId, ColumnData> columns_;
  ApproxCounter numFiles_;
  folly::F14FastSet<uint64_t> sampleFiles_;
  uint64_t numStripes_{0};
};

void recordStripes(uint64_t fileId, int32_t numStripes, uint64_t fileSize);

void recordReference(TrackingId id, uint64_t fileId, uint64_t groupId);

void recordRead(TrackingId id, uint64_t fileId, uint64_t groupId);

// Singleton for  keeping track of file groups.
class GroupStats {
 public:
  static GroupStats* instance();

 private:
  F14FastMap<uint64_t, std::unique_ptr<GroupTracker>> groups_;
};

} // namespace facebook::velox::cache
