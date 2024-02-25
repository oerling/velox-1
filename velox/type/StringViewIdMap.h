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
#include "velox/type/StringView.h"

namespace facebook::velox {

using int32x4 = int32_t[4];

class StringViewIdMap {
 public:
  static constexpr int64_t kNotFound = ~0L;

  StringViewIdMap(int32_t size);
  

  /// Returns a unique id for 'view'. If 'view' was added and did not
  /// fit inline, '*copyPtr' is set to point to an address that the
  /// caller will fill with the pointer of an externally managed copy
  /// of the string in 'view'. this copy is expected to have the
  /// lifetime of 'this'. If the string was found or fit inline,
  /// '*copyPtr' is set to nullptr.
  int32_t makeId(const StringView& view, char** copyPtr) {
    uint64_t sizeAndPrefix = *reinterpret_cast<const uint64_t*>(&view);
    uint32_t size = static_cast<uint32_t>(sizeAndPrefix);

    if (UNLIKELY(size == 0)) {
      if (emptyId_ != kNoEmpty) {
        return emptyId_;
      }
      emptyId_ = numEntries_++;
      return emptyId_;
    }
    auto hash = hash1(view) & sizeMask_;
    return findEntry<true>(hash, view, sizeAndPrefix, copyPtr);
  }

  int32_t findId(const StringView& view) {
    uint64_t sizeAndPrefix = *reinterpret_cast<const uint64_t*>(&view);
    uint32_t size = static_cast<uint32_t>(sizeAndPrefix);

    if (UNLIKELY(size == 0)) {
      if (emptyId_ != kNoEmpty) {
        return emptyId_;
      }
      return kNotFound;
    }
    auto hash = hash1(view) & sizeMask_;
    return findEntry<false>(hash, view, sizeAndPrefix, nullptr);
  }

  xsimd::batch<int64_t>
  makeIds(const StringView* views, int32x4 indices, char*** copyPtr) {
    int64_t result[4];
    result[0] = makeId(views[indices[0]], copyPtr[0]);
    result[1] = makeId(views[indices[1]], copyPtr[1]);
    result[2] = makeId(views[indices[2]], copyPtr[2]);
    result[3] = makeId(views[indices[3]], copyPtr[3]);
    return xsimd::load_unaligned(&result[0]);
  }

  xsimd::batch<int64_t> findIds(const StringView* views, int32x4 indices) {
    int64_t result[4];
    result[0] = findId(views[indices[0]]);
    result[1] = findId(views[indices[1]]);
    result[2] = findId(views[indices[2]]);
    result[3] = findId(views[indices[3]]);
    return xsimd::load_unaligned(&result[0]);
  }

  template <bool makeNew>
  int32_t findEntry(
      int32_t index,
      const StringView& view,
      uint64_t sizeAndPrefix,
      char** copyPtr) {
    auto size = static_cast<uint32_t>(sizeAndPrefix);
    for (;;) {
      auto entry = table_.data() + index * 3;
      uint64_t word = *entry;
      if (word == sizeAndPrefix) {
        if (testRestOfHit(entry, view, size)) {
          return entry[2];
        }
      } else if (word == 0) {
        if (!makeNew) {
          return kNotFound;
        }
        if (UNLIKELY(numEntries_ >= maxEntries_)) {
          resize(capacity_ * 2);
          return makeId(view, copyPtr);
        }
        auto id = numEntries_++;
        copyPtr = size > StringView::kInlineSize
            ? reinterpret_cast<char**>(entry + 1)
            : nullptr;
        addEntry(entry, view, id);
        return id;
      }
      ++collisions_;
      index = (index + 1) & sizeMask_;
    }
  }
  void clear();
  
 private:
  static constexpr int32_t kNoEmpty = ~0;
  uint64_t hash1(const StringView& view) {
    auto h =  bits::hashBytes(1, view.data(), view.size());
    return h ^ (h >> 32);
    uint64_t sizeAndPrefix = view.sizeAndPrefixAsInt64();
    uint32_t hash = static_cast<uint32_t>(sizeAndPrefix);
    uint32_t size = hash;
    hash *= (sizeAndPrefix >> 32);
    hash = hash ^ (hash  >> 16);
    uint64_t tail = 1;
    if (size > StringView::kInlineSize) {
      tail = *reinterpret_cast<const uint64_t*>(view.value_.data + size - 8);
    } else if (size > 4) {
      tail = *reinterpret_cast<const uint64_t*>(
          reinterpret_cast<const char*>(&view) + size - 4);
    }
    return hash ^ (tail + (tail >> 32));
  }

  // Tests if entry matches view after the length and prefix have compared
  // equal.
  bool
  testRestOfHit(const uint64_t* entry, const StringView& view, int32_t size) {
    if (size <= StringView::kInlineSize) {
      if (size <= 4) {
        return true;
      }
      return entry[1] == reinterpret_cast<const uint64_t*>(&view)[1];
    }
    bool flag = memcmp(
               reinterpret_cast<const char*>(entry[1]) + 4,
               view.value_.data + 4,
               size - 4) == 0;
    ++collisions2_;
    return flag;
  }

  void addEntry(uint64_t* entry, const StringView& view, int32_t id) {
    memcpy(entry, &view, sizeof(view));
    entry[2] = id;
  }

  void resize(int32_t newSize);

  int32_t emptyId_{kNoEmpty};
  // Count of 3 word entries in 'table_'.
  uint64_t capacity_{0};

  // Mask corresponding to the table_.size().
  uint64_t sizeMask_{0};
  // Table with 3 words per entry: 2 for StringView, 1 for id.
  raw_vector<uint64_t> table_;

  // Count of non-empty entries in 'table_'.
  int32_t numEntries_{0};

  // Count of entries after which a resize() should be done.
  int32_t maxEntries_;
  int32_t collisions_{0};
  int32_t collisions2_{0};
};

} // namespace facebook::velox
