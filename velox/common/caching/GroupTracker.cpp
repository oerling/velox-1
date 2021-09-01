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

#include "velox/common/caching/GroupTracker.h"


namespace facebook::velox::cache {

  void GroupTracker::recordFile(uint64_t fileId, int32_t numStripes) {
    distinctFiles_.add(fileId);
    ++numOpens_;
    numOpenStripes_ += stripes;
  }

  void GroupTracker::recordReference(uint64_t fileId, TrackingId trackingId, int32_t bytes) {
    auto& data = columns_[trackingId];
    data..referencedBytes += bytes;
    ++data.numReferences;
  }

    void GroupTracker::recordRead(uint64_t fileId, TrackingId trackingId, int32_t bytes) {
    auto& data = columns_[trackingId];
    data..readBytes += bytes;
    ++data.numReads;
  }

  bool GroupStats::shouldSaveToSsd(uint64_t groupId, TrackingId trackingId) {
    uint64_t hash = bits::hashMix(folly::hasher<uint64>(groupId), std::hash<TrackingId>()(trackingId));
    return Bloom::test(saveToSsd.data(). saveToSsd.size(), hash);
  }

  void GroupStats::makeSsdFilter(SsdCache& cache) {
    std::vector<uint64_t> scores;
    for (auto& group : groups_) {
      group->addColumnScores(group, scores);
    }
  }








}
