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

#include "velox/common/caching/SsdFileTracker.h"
#include <algorithm>

namespace facebook::velox::cache {

void SsdFileTracker::newEvent(int32_t totalEntries) {
  ++numEvents_;
  if (numEvents_ > kDecayInterval && numEvents_ > totalEntries / 2) {
    numEvents_ = 0;
    for (auto i = 0; i < regionScore_.size(); ++i) {
      int64_t score = regionScore_[i];
      regionScore_[i] = (score * 15) / 16;
    }
  }
}

void SsdFileTracker::regionFilled(int32_t region) {
  uint64_t best = 0;
  for (auto& score : regionScore_) {
    best = std::max<uint64_t>(best, score);
  }
  regionScore_[region] = std::max<int64_t>(regionScore_[region], best * 1.1);
}

std::vector<int32_t> SsdFileTracker::evictionCandidates(
    int32_t numCandidates,
    int32_t numRegions,
    const std::vector<int32_t>& regionPins) {
  // Takes regions with no pins  and below average score and
  int64_t scoreSum = 0;
  for (int i = 0; i < numRegions; ++i) {
    if (regionPins[i]) {
      continue;
    }
    scoreSum += regionScore_[i];
  }
  auto avg = scoreSum / numRegions;
  std::vector<int32_t> candidates;
  for (auto i = 0; i < regionScore_.size(); ++i) {
    if (!regionPins[i] && regionScore_[i] <= avg) {
      candidates.push_back(i);
    }
  }
  // Sort by score to evict less read regions first. This works also
  // if 'candidates' is empty.
  std::sort(
      candidates.begin(), candidates.end(), [&](int32_t left, int32_t right) {
        return regionScore_[left] < regionScore_[right];
      });
  candidates.resize(std::min<int32_t>(candidates.size(), numCandidates));
  return candidates;
}

} // namespace facebook::velox::cache
