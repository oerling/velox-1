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

#include "velox/common/base/RawVector.h"

/// A utility for reusable scoped temporary scratch areas.
namespace facebook::velox {
/// A collection of temporary scratch vectors.
class Scratch {
 public:
  Scratch() = default;
  Scratch(const Scratch& other) = delete;
  void operator =(const Scratch& other) = delete;
  raw_vector<char> get() {
    if (scratch_.empty()) {
      return raw_vector<char>();
    }
    auto temp = std::move(scratch_.back());
    scratch_.pop_back();
    retainedSize_ -= temp.capacity();
    return temp;
  }

  void release(raw_vector<char>&& item) {
    scratch_.push_back(std::move(item));
    retainedSize_ += scratch_.back().capacity();
  }

  void trim() {
    scratch_.clear();
  }

  size_t retainedSize() {
    return retainedSize_;
  }

 private:
  std::vector<raw_vector<char>> scratch_;

  // The total size held. If too large from outlier use cases, 'this' should be
  // trimmed.
  int64_t retainedSize_{0};
};

/// A scoped lease for a scratch area of T.
template <typename T>
class ScratchPtr {
 public:
  ScratchPtr(Scratch& scratch) : scratch_(scratch) {}

  ~ScratchPtr() {
    if (ptr_) {
      scratch_.release(std::move(data_));
    }
  }

  T* get(int32_t size) {
    data_ = scratch_.get();
    data_.resize(size * sizeof(T));
    ptr_ = reinterpret_cast<T*>(data_.data());
  }

  bool hasData() const {
    return ptr_ != nullptr;
  }
  const raw_vector<char>& data() const {
    return data_;
  }
  
 private:
  Scratch& scratch_;
  raw_vector<char> data_;
  T* ptr_;
};

} // namespace facebook::velox
