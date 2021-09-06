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
namespace {
uint64_t ssdFilterHash(uint64_t groupId, TrackingId trackingId) {
  return bits::hashMix(
      folly::hasher<uint64_t>()(groupId), std::hash<TrackingId>()(trackingId));
}

// Returns an arbitrary multiplier for score based on
// size.

float sizeFactor(float size) {
  // Number of bytes transferred as part of a large request in in
  // the time of a round trip with no data transfer.
  constexpr float kBytesPerLatency = 10000;
  return kBytesPerLatency / (kBytesPerLatency + size);
}
} // namespace

void GroupTracker::addColumnScores(std::vector<SsdScore>& scores) const {
  if (!numOpens_) {
    return;
  }
  int32_t numFiles = numFiles_.count();
  auto stripesInFile = numOpenStripes_ / numOpens_;
  auto numStripes = numFiles * stripesInFile;
  for (auto& pair : columns_) {
    auto& data = pair.second;
    if (!data.numReads || !data.numReferences) {
      continue; // Unused.
    }
    float size = (data.referencedBytes / data.numReferences) * numStripes;
    float readSize = data.readBytes / data.numReads;
    float readFraction = readSize / size;
    float score = data.numReads * sizeFactor(size) * readFraction;
    scores.push_back(SsdScore{score, size, ssdFilterHash(name_.id(), pair.first)});
  }
}

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
  if (allFitOnSsd_) {
    return true;
  }
  uint64_t hash = ssdFilterHash(groupId, trackingId);
  return Bloom::test(saveToSsd_.data(), saveToSsd_.size(), hash);
}

void GroupStats::updateSsdFilter(uint64_t ssdSize) {
  std::vector<SsdScore> scores;
  for (auto& pair : groups_) {
    pair.second->addColumnScores(scores);
  }
  // Sort the scores, high score first.
  std::sort(
      scores.begin(),
      scores.end(),
      [](const SsdScore& left, const SsdScore& right) {
        return left.score > right.score;
      });
  float size = 0;

  int32_t i = 0;
  for (; i < scores.size(); ++i) {
    size += scores[i].size;
    if (size > ssdSize) {
      break;
    }
  }
  if (i == scores.size()) {
    allFitOnSsd_ = true;
  } else {
    allFitOnSsd_ = false;
    saveToSsd_.clear();
    saveToSsd_.resize(bits::nextPowerOfTwo(4 + (i / 8)));
    for (auto included = 0; included < i; ++included) {
      Bloom::set(saveToSsd_.data(), saveToSsd_.size(), scores[i].hash);
    }
  }
}

GroupStats& GroupStats::instance() {
  static GroupStats* stats = new GroupStats();
  return *stats;
}
} // namespace facebook::velox::cache
