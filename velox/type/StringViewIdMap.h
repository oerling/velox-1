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
    auto hash = (hash1(view) & sizeMask_) * kEntrySize;
    return findEntry<true, true>(hash, view, sizeAndPrefix, copyPtr);
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
    auto hash =( hash1(view) & sizeMask_) * kEntrySize;
    return findEntry<false, false>(hash, view, sizeAndPrefix, nullptr);
  }



  void findIds8(const StringView* views, const int32_t* indices, bool makeIds, int32_t* ids, char** tails);

  void findIds8Scalar(const StringView* views, const int32_t* indices, bool makeIds, int32_t* ids, char** tails) {
    if (makeIds) {
      ids[0] = makeId(views[indices[0]], &tails[0]);
    ids[1] = makeId(views[indices[1]], &tails[1]);
    ids[2] = makeId(views[indices[2]], &tails[2]);
    ids[3] = makeId(views[indices[3]], &tails[3]);
      ids[4] = makeId(views[indices[4]], &tails[4]);
    ids[5] = makeId(views[indices[5]], &tails[5]);
    ids[6] = makeId(views[indices[6]], &tails[6]);
    ids[7] = makeId(views[indices[7]], &tails[7]);
    } else {
      ids[0] = findId(views[indices[0]]);
      ids[1] = findId(views[indices[1]]);
      ids[2] = findId(views[indices[2]]);
      ids[3] = findId(views[indices[3]]);
      ids[4] = findId(views[indices[4]]);
      ids[5] = findId(views[indices[5]]);
      ids[6] = findId(views[indices[6]]);
      ids[7] = findId(views[indices[7]]);
    }
  }

  template <bool makeNew, bool mayRehash>
  int32_t findEntry(
      int32_t offset,
      const StringView& view,
      uint64_t sizeAndPrefix,
      char** copyPtr) {
    auto size = static_cast<uint32_t>(sizeAndPrefix);
    for (;;) {
      auto entry = reinterpret_cast<int64_t*>(table_.data() + offset);
      uint64_t word = *entry;
      if (word == sizeAndPrefix) {
        if (testRestOfHit(entry, view, size)) {
          return reinterpret_cast<int32_t*>(entry)[4];
        }
      } else if (word == kEmpty) {
        if (!makeNew) {
          return kNotFound;
        }
        if (UNLIKELY(mayRehash && numEntries_ >= maxEntries_)) {
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
      offset = nextOffset(offset);
    }
  }

  void clear();

 private:
  static constexpr int32_t kNoEmpty = ~0;
  static constexpr int64_t kEmpty = 0;
  static constexpr int32_t kEntrySize = 20;

  uint64_t hash1(const StringView& view) {
    //auto h = bits::hashBytes(1, view.data(), view.size());
    //return h; // ^ (h >> 32);
    uint64_t sizeAndPrefix = view.sizeAndPrefixAsInt64();
    int32_t size = static_cast<int32_t>(sizeAndPrefix);
    if (size > StringView::kInlineSize) {
      auto tail = *reinterpret_cast<const uint64_t*>(view.value_.data + size - 8);
      if (LIKELY(tail)) {
	return simd::crc32U64(sizeAndPrefix >> 32, tail);
      }
      } else if (size > 4) {
      auto tail = *reinterpret_cast<const uint64_t*>(
          reinterpret_cast<const char*>(&view) + size - 8);
      if (LIKELY(tail)) {
	return simd::crc32U64(sizeAndPrefix >> 32, tail);
      }
      }
    return simd::crc32U64(1, sizeAndPrefix);
  }


  // Tests if entry matches view after the length and prefix have compared
  // equal.
  bool
  testRestOfHit(const int64_t* entry, const StringView& view, int32_t size) {
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

  void addEntry(int64_t* entry, const StringView& view, int32_t id) {
    memcpy(entry, &view, sizeof(view));
    reinterpret_cast<int32_t*>(entry)[4] = id;
  }

  void resize(int32_t newSize);
  int32_t nextOffset(int32_t offset) {
    return offset == lastEntryOffset_ ? 0 : offset + kEntrySize;
  }
  
  int32_t emptyId_{kNoEmpty};
  // Count of 20 byte entries in 'table_'.
  uint64_t capacity_{0};

  // Mask corresponding to the table_.size().
  uint64_t sizeMask_{0};

  // Offset of last entry in 'table_'
  int32_t lastEntryOffset_{0};

  // Table with 20 bytes per entry, 16 for StringView and 4 for its id.
  raw_vector<uint8_t> table_;

  // Count of non-empty entries in 'table_'.
  int32_t numEntries_{0};

  // Count of entries after which a resize() should be done.
  int32_t maxEntries_;
  int32_t collisions_{0};
  int32_t collisions2_{0};
};

#if 0
  int32_t check1(StringView* view, int32_t& id, int32_t& offset) {
    auto head = view->sizeAndPrefixAsInt64();
    auto size  static_cast<uint32_t>(head);
    if (size <= 4) {
      h = offset(1, head);
      word = reinterpret_cast<int64_t*>(table + offset);
      if (word == head) {
	id = reinterpret_cast<int32_t*>(table + offset + sizeof(StringView));
	done |= nth;
      } else if (word == kEmpty) {
	simd::memcpy(table + offset, view, sizeof(StringView));
	table + offset + sizeof(StringView) = numEntries_++;
	done |= nth;
      } else {
	offset = nextOffset(offset);
      }
    } else if (size <= 12) {
      offset = offset(head >> 32, reinterpret_cast<int64_t*>(view)[1]);
      word = ref64(table, offset);
      if (head == word && reinterpret_cast<int64_t*>(table + offset)[1] == ref64(view, 8))
	id = ref32(table + offset + 16);
	done = nth;
	} else if (word == kEmpty) 
      } else {
    offset = nextOffset(offset);
  } else {
  tail = ref64(view, 8);
  last = reinterpret_cast<int64_t*>(tail + size - 8);
  offset = offset(head >> 32, last);
  word = ref64(table, offset);
  if (head == word)
    auto tail = ref64(table + offset + 8);
  if (memcmp(ref64(view, 8), last, size)) {
    id = ref32(table. sizeof(StringView));
    done |= nth;
  }
 }

  
  #endif

  
} // namespace facebook::velox
