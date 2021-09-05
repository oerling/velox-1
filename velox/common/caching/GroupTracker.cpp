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

#include "velox/common/base/BitUtil.h"

namespace facebook::velox::cache {

void GroupTracker::recordFile(uint64_t fileId, int32_t numStripes) {
  numFiles_.add(fileId);
  ++numOpens_;
  numOpenStripes_ += numStripes;
}

void GroupTracker::recordReference(
    uint64_t fileId,
    TrackingId trackingId,
    int32_t bytes) {
  auto& data = columns_[trackingId];
  data.referencedBytes += bytes;
  ++data.numReferences;
}

void GroupTracker::recordRead(
    uint64_t fileId,
    TrackingId trackingId,
    int32_t bytes) {
  auto& data = columns_[trackingId];
  data.readBytes += bytes;
  ++data.numReads;
}

void GroupTracker::addColumnScores(std::vector<SsdScore>& scores) const {}

void GroupStats::recordFile(
    uint64_t fileId,
    uint64_t groupId,
    int32_t numStripes) {
  std::lock_guard<std::mutex> l(mutex_);
  group(groupId).recordFile(fileId, numStripes);
}

void GroupStats::recordReference(
    uint64_t fileId,
    uint64_t groupId,
    TrackingId trackingId,
    int32_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  group(groupId).recordReference(fileId, trackingId, bytes);
}

void GroupStats::recordRead(
    uint64_t fileId,
    uint64_t groupId,
    TrackingId trackingId,
    int32_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  group(groupId).recordRead(fileId, trackingId, bytes);
}

bool GroupStats::shouldSaveToSsd(uint64_t groupId, TrackingId trackingId)
    const {
  uint64_t hash = bits::hashMix(
      folly::hasher<uint64_t>()(groupId), std::hash<TrackingId>()(trackingId));
  return Bloom::test(saveToSsd_.data(), saveToSsd_.size(), hash);
}

void GroupStats::updateSsdFilter(uint64_t ssdSize) {
  std::vector<SsdScore> scores;
  for (auto& pair : groups_) {
    pair.second->addColumnScores(scores);
  }
}

GroupStats& GroupStats::instance() {
  static GroupStats* stats = new GroupStats();
  return *stats;
}
} // namespace facebook::velox::cache
