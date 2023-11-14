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
  using Item = raw_vector<char>;

  Scratch() = default;
  Scratch(const Scratch& other) = delete;

  ~Scratch() {
    reserve(0);
    ::free(items_);
    items_ = nullptr;
    capacity_ = 0;
    fill_ = 0;
  }
  void operator=(const Scratch& other) = delete;

  /// Returns the next reusable scratch vector or makes a new one.
  Item get() {
    if (fill_ == 0) {
      return Item();
    }
    auto temp = std::move(items_[fill_ - 1]);
    --fill_;
    retainedSize_ -= temp.capacity();
    return temp;
  }

  void release(Item&& item) {
    retainedSize_ += item.capacity();
    if (fill_ == capacity_) {
      reserve(std::max(16, 2 * capacity_));
    }
    items_[fill_++] = std::move(item);
  }

  void trim() {
    reserve(0);
    retainedSize_ = 0;
  }

  size_t retainedSize() {
    return retainedSize_;
  }

 private:
  void reserve(int32_t newCapacity) {
    VELOX_CHECK_LE(fill_, capacity_);
    // Delete the items above the new capacity.
    for (auto i = newCapacity; i < fill_; ++i) {
      std::destroy_at(&items_[i]);
    }
    if (newCapacity > capacity_) {
      Item* newItems =
          reinterpret_cast<Item*>(::malloc(sizeof(Item) * newCapacity));
      memcpy(newItems, items_, fill_ * sizeof(Item));
      memset(newItems + fill_, 0, (newCapacity - fill_) * sizeof(Item));
      free(items_);
      items_ = newItems;
      capacity_ = newCapacity;
    }
    fill_ = std::min(fill_, newCapacity);
  }

  Item* items_{nullptr};
  int32_t fill_{0};
  int32_t capacity_{0};
  // The total size held. If too large from outlier use cases, 'this' should be
  // trimmed.
  int64_t retainedSize_{0};
};

/// A scoped lease for a scratch area of T.
template <typename T>
class ScratchPtr {
 public:
  ScratchPtr(Scratch& scratch) : scratch_(&scratch) {}

  ScratchPtr(ScratchPtr&& other) {
    *this = std::move(other);
  }

  ~ScratchPtr() {
    if (ptr_) {
      scratch_->release(std::move(data_));
    }
  }

  ScratchPtr(const ScratchPtr& other) = delete;

  void operator=(ScratchPtr&& other) {
    scratch_ = other.scratch_;
    data_ = std::move(other.data_);
    other.ptr_ = nullptr;
  }

  void operator=(const ScratchPtr& other) = delete;

  T* get(int32_t size) {
    VELOX_CHECK(data_.empty());
    data_ = std::move(scratch_->get());
    data_.resize(size * sizeof(T));

    ptr_ = reinterpret_cast<T*>(data_.data());
    return ptr_;
  }

  T* get() const {
    VELOX_DCHECK_NOT_NULL(ptr_);
    return ptr_;
  }

  bool hasData() const {
    return ptr_ != nullptr;
  }

  const raw_vector<char>& data() const {
    return data_;
  }

 private:
  Scratch* scratch_{nullptr};
  raw_vector<char> data_;
  T* ptr_{nullptr};
};

} // namespace facebook::velox
