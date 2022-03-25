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

#include <memory>
#include <optional>
#include <vector>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

// Abstract class defining the interface for a stream of values to be merged by
// TreeOfLosers.
class TreeOfLosersSource {
  // True if this has a value. If this returns true, it is valid to
  // call < or next immediately after this. If false, no more
  // methods other than hasData() may be called.
  virtual bool hasData() const = 0;

  // Pops the first value off 'this'. One must call hasData() after next() to
  // determine if next or < may be called after this.
  virtual void next() = 0;

  // Returns true if the first element of 'this' is less than the first element
  // of 'other'. hasData() must be true of 'this' and 'other'.
  virtual bool operator<(const TreeOfLosersSource& other) const = 0;
};

// Implements a tree of loserrs algorithm for merging ordered
// streams. The TreeOfLosers owns two or more instances of
// Source. At each call of next(), it returns the Source that has
// the lowest value as first value from the set of Sources. It
// returns nullptr when all Sources are at end. The order is
// determined by Source::operator<.
template <typename Source>
class TreeOfLosers {
 public:
  // A leaf Node owns a Source. Inner Nodes own their children. A
  // node's 'state_' records whether the node is at end, which child's
  // value was last returned and whether one or both children are at
  // end.
  class Node {
   public:
    Node(std::unique_ptr<Source>&& source)
        : state_(State::kSource), source_(std::move(source)) {
      static_assert(std::is_base_of<TreeOfLosersSource, Source>::value);
    }

    Node(std::unique_ptr<Node> left, std::unique_ptr<Node>&& right)
        : state_(State::kInit),
          left_(std::move(left)),
          right_(std::move(right)) {}

    // Returns the Source with the lowest value of all the sources under 'this'.
    Source* FOLLY_NULLABLE next() {
      switch (state_) {
        case State::kAtEnd:
          return nullptr;
        case State::kSource:
          if (source_->hasData()) {
            return source_.get();
          }
          state_ = State::kAtEnd;
          return nullptr;
        case State::kInit:
          leftValue_ = left_->next();
          rightValue_ = right_->next();
          if (!leftValue_) {
            state_ = rightValue_ ? State::kRightOnly : State::kAtEnd;
            return rightValue_;
          } else if (!rightValue_) {
            state_ = State::kLeftOnly;
            return leftValue_;
          }
          return makeResult();
        case State::kLeftResult:
          leftValue_ = left_->next();
          if (!leftValue_) {
            state_ = State::kRightOnly;
            return rightValue_;
          }
          return makeResult();
        case State::kRightResult:
          rightValue_ = right_->next();
          if (!rightValue_) {
            state_ = State::kLeftOnly;
            return leftValue_;
          }
          return makeResult();
        case State::kLeftOnly:
          return left_->next();
        case State::kRightOnly:
          return right_->next();
      }
      VELOX_UNREACHABLE();
    }

   private:
    enum class State {
      kInit,
      kSource,
      kAtEnd,
      kLeftResult,
      kRightResult,
      kLeftOnly,
      kRightOnly
    };

    // Returns the lesser of the Sources from left and right children
    // and sets 'state_'.
    inline Source* makeResult() {
      if (*leftValue_ < *rightValue_) {
        state_ = State::kLeftResult;
        return leftValue_;
      }
      state_ = State::kRightResult;
      return rightValue_;
    }

    State state_;
    Source* leftValue_{nullptr};
    Source* rightValue_{nullptr};
    std::unique_ptr<Node> left_;
    std::unique_ptr<Node> right_;
    std::unique_ptr<Source> source_;
  };

  // Constructs a balanced binary tree where each leaf corresponds to an element
  // of 'sources'.
  TreeOfLosers(std::vector<std::unique_ptr<Source>>&& sources) {
    std::vector<std::unique_ptr<Node>> level(sources.size());
    for (auto i = 0; i < sources.size(); ++i) {
      level[i] = std::make_unique<Node>(std::move(sources[i]));
    }
    while (level.size() > 1) {
      std::vector<std::unique_ptr<Node>> nextLevel;
      for (auto i = 0; i < level.size(); i += 2) {
        if (i <= level.size() - 2) {
          nextLevel.push_back(std::make_unique<Node>(
              std::move(level[i]), std::move(level[i + 1])));
        } else {
          nextLevel.push_back(std::move(level[i]));
        }
      }
      level = std::move(nextLevel);
    }
    root_ = std::move(level[0]);
  }

  // Returns the Source with the lowest value in the first element and
  // nullptr if all sources are at end. On a non-first call, first
  // removes the first value off the source returned on the previous
  // call.
  Source* FOLLY_NULLABLE next() {
    if (lastValue_) {
      lastValue_->next();
    }
    lastValue_ = root_->next();
    return lastValue_;
  }

 private:
  Source* lastValue_{nullptr};
  std::unique_ptr<Node> root_;
};
} // namespace facebook::velox
