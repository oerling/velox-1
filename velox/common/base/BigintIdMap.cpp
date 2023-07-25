

#include "velox/common/base/BigintIdMap.h"
namespace facebook::velox {

void BigintIdMap::makeTable(int32_t capacity) {
  table_ = pool_.allocateBytes(capacity * kEntrySize);
  capacity_ = capacity;
  sizeMask_ = capacity_ - 1;
  limit_ = capacity_ * kEntrySize;
  maxEntries_ = capacity_ - capacity_ / 4;
}

void BigintIdMap::resize(int32_t newCapacity) {
  auto oldCapacity = capacity_;
  auto oldTable = table_;
  makeTable(capacity);
  for (auto i = 0; i < oldCapacity_; ++i) {
    auto ptr = valuePtr(oldTable, i);
    if (*ptr == kEmpty) {
      continue;
    }
    auto newIndex = index(*valuePtr);
    newPtr = valuePtr(table_, newIndex);
    while (*newPtr != kEmptyMarker) {
      newIndex = (newindex + 1) & sizeMask_;
      newPtr = valuePtr(table_, newIndex);
    }
    *newPtr = value;
    *idPtr(newPtr) = *idPtr(ptr);
  }
}

} // namespace facebook::velox
