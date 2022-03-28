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

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SimdUtil.h"

#include <folly/Likely.h>

namespace facebook::velox {

// Abstract class defining the interface for a stream of values to be merged by
// TreeOfLosers or MergeArray.
class MergeStream {
  // True if this has a value. If this returns true, it is valid to
  // call < or next immediately after this. If false, no more
  // methods other than hasData() may be called.
  virtual bool hasData() const = 0;

  // Pops the first value off 'this'. One must call hasData() after next() to
  // determine if next or < may be called after this.
  virtual void next() = 0;

  // Returns true if the first element of 'this' is less than the first element
  // of 'other'. hasData() must be true of 'this' and 'other'.
  virtual bool operator<(const MergeStream& other) const = 0;
};

// Implements a tree of loserrs algorithm for merging ordered
// streams. The TreeOfLosers owns two or more instances of
// Source. At each call of next(), it returns the Source that has
// the lowest value as first value from the set of Sources. It
// returns nullptr when all Sources are at end. The order is
// determined by Source::operator<.
template <typename Source, typename TIndex = uint16_t>
class TreeOfLosers {
 public:
  TreeOfLosers(std::vector<std::unique_ptr<Source>> sources)
      : sources_(std::move(sources)) {
    VELOX_CHECK_LT(sources_.size(), std::numeric_limits<TIndex>::max());
    int32_t size = 0;
    int32_t levelSize = 1;
    int32_t numSources = sources_.size();
    while (numSources > levelSize) {
      size += levelSize;
      levelSize *= 2;
    }

    if (numSources == bits::nextPowerOfTwo(numSources)) {
      firstSource_ = size - levelSize;
    } else {
      // Some of the sources are on the last level and some on the level before.
      // The first source follows the last inner node in the node numbering.

      auto secondLastSize = levelSize / 2;
      auto overflow = numSources - secondLastSize;
      auto sourcesOnSecondLast = secondLastSize - overflow;
      // Suppose 12 sources. The last level has 16 places, the second
      // last 8. If we fill the second last level we have 8 sources
      // and 4 left over. These 4 need parents on the second last
      // level. So, we end up with 4 inner nodes on the second last
      // level and 8 nodes on the last level. The sources at the left
      // of the second last level become inner nodes and their sources
      // move to the level below.
      firstSource_ = (size - levelSize - secondLastSize) + overflow;
    }
    values_.resize(firstSource_, kEmpty);
  }

  Source* next() {
    if (lastIndex_ == kEmpty) {
      lastIndex_ = first(0);
    } else {
      sources_[lastIndex_]->next();
      lastIndex_ = propagate(
          parent(firstSource_ + lastIndex_),
          sources_[lastIndex_]->hasData() ? lastIndex_ : kEmpty);
    }
    return lastIndex_ == kEmpty ? nullptr : sources_[lastIndex_].get();
  }

 private:
  static constexpr TIndex kEmpty = std::numeric_limits<TIndex>::max();

  TIndex first(TIndex node) {
    if (node >= firstSource_) {
      return sources_[node - firstSource_]->hasData() ? node - firstSource_
	: kEmpty;
    }
    auto left = first(leftChild(node));
    auto right = first(rightChild(node));
    if (left == kEmpty) {
      return right;
    } else if (right == kEmpty) {
      return left;
    } else if (*sources_[left] < *sources_[right]) {
      values_[node] = right;
      return left;
    } else {
      values_[node] = left;
      return right;
    }
  }

  TIndex propagate(TIndex node, TIndex value) {
    while (UNLIKELY(values_[node] == kEmpty)) {
      if (UNLIKELY(node == 0)) {
        return value;
      }
      node = parent(node);
    }
    for (;;) {
      if (value == kEmpty) {
        value = values_[node];
        values_[node] = kEmpty;
      } else if (*sources_[values_[node]] < *sources_[value]) {
        // The node had the lower value, the value stays here and the previous
        // value goes up.
        std::swap(value, values_[node]);
      } else {
        // The value is less than the value in the node, No action, the value
        // goes up.
        ;
      }
      if (UNLIKELY(node == 0)) {
        return value;
      }
      node = parent(node);
    }
  }

  static TIndex parent(TIndex node) {
    return (node - 1) / 2;
  }

  static TIndex leftChild(TIndex node) {
    return node * 2 + 1;
  }

  static TIndex rightChild(TIndex node) {
    return node * 2 + 2;
  }
  std::vector<TIndex> values_;
  std::vector<std::unique_ptr<Source>> sources_;
  TIndex lastIndex_ = kEmpty;
  int32_t firstSource_;
};

template <typename Source>
class MergeArray {
 public:
  MergeArray(std::vector<std::unique_ptr<Source>> sources) {
    static_assert(std::is_base_of<MergeStream, Source>::value);
    for (auto& source : sources_) {
      if (source->hasData()) {
        sources_.push_back(std::move(source));
      }
    }
    std::sort(
        sources_.begin(),
        sources_.end(),
        [](const auto& left, const auto& right) { return *left < *right; });
  }
  Source* next() {
    if (!isFirst_) {
      isFirst_ = false;
      sources_[0]->next();
      if (!sources_[0]->hasData()) {
        sources_.erase(sources_.begin());
	return sources_.empty() ? nullptr : sources_[0].get();
      }
    }
    auto rawSources = reinterpret_cast<Source**>(sources_.data());
    auto first = rawSources[0];
    auto it = std::lower_bound(
            rawSources + 1,
            rawSources + sources_.size(),
            first,
            [](const Source* left, const Source* right) {
              return *left < *right;
            });
        auto offset = it - rawSources;
        if (offset > 1) {
          simd::memcpy(
              rawSources, rawSources + 1, (offset - 1) * sizeof(Source*));
          it[-1] = first;
        }
        return sources_[0].get();
  }

private:
  bool isFirst_{true};
  std::vector<std::unique_ptr<Source>> sources_;
};

} // namespace facebook::velox
