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

#include <atomic>
#include <semaphore>

namespace facebook::velox::cuda {

template <typename T>

using vx_atomic<T> = std::atomic<T>;

template <typename T>
struct Span {
  T* data;
  int32_t size;
};

// Allocates small objects in unified memory. Used for plan objects. STL
// compatible.

class UnifiedAllocator {
  void* allocate(int64_t size) {
    return malloc(size);
  }

  void free(void* data) {
    ::free(data);
  }
}

// STL Allocator
template <class T>
struct StlAllocator {
  using value_type = T;

  explicit StlAllocator(UnifiedAllocator* FOLLY_NONNULL allocator)
      : allocator_{allocator} {
    VELOX_CHECK(allocator);
  }

  template <class U>
  explicit StlAllocator(const StlAllocator<U>& allocator)
      : allocator_{allocator.allocator()} {
    VELOX_CHECK(allocator_);
  }

  T* FOLLY_NONNULL allocate(std::size_t n) {
    return reinterpret_cast<T*>(
        allocator_->allocate(checkedMultiply(n, sizeof(T)))->begin());
  }

  void deallocate(T* FOLLY_NONNULL p, std::size_t /*n*/) noexcept {
    allocator_->free(p);
  }

  UnifiedAllocator* FOLLY_NONNULL allocator() const {
    return allocator_;
  }

  friend bool operator==(const StlAllocator& lhs, const StlAllocator& rhs) {
    return lhs.allocator_ == rhs.allocator_;
  }

  friend bool operator!=(const StlAllocator& lhs, const StlAllocator& rhs) {
    return !(lhs == rhs);
  }

 private:
  UnifiedAllocator* FOLLY_NONNULL allocator_;
};

} // namespace facebook::velox::cuda
