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

#include "velox/exec/Strings.h"

namespace facebook::velox::aggregate::prestosql {

StringView Strings::append(StringView value, HashStringAllocator& allocator) {
  // A string in Strings needs 8 bytes below itself for the last 8
  // bytes of the previous string, which copied to the next string
  // when setting a continued pointer. It also needs 8 bytes at its
  // tail for the same continue pointer to the next.
  constexpr int32_t kOverhead =
      HashStringAllocator::Header::kContinuedPtrSize + 8;
  VELOX_DCHECK(!value.isInline());

  maxStringSize = std::max<int32_t>(maxStringSize, value.size());
  ++numStrings;

  // Request sufficient amount of memory to store the whole string
  // (value.size()) and allow some memory left for bookkeeping (header + link
  // to next block).
  const int32_t requiredBytes = value.size() + kOverhead;
  // The first 2 are allocated at size. Subsequent ones are allocated
  // at 4 * the maximum or requested size. Using maximum size if this
  // is below 200. Else using requested size times 4 if the size is
  // under 200, else allocating strings one by one. Allocating singly
  // for over 200 size is fine, the overhead is under 10%.
  const int32_t roundedUpBytes = numStrings <= 2 ? requiredBytes
    : maxStringSize < 200                    ? maxStringSize * 4 + kOverhead
    : value.size() * (value.size() < 200 ? 4 : 1) + kOverhead;
  ByteStream stream(&allocator);
  if (firstBlock == nullptr) {
    // Allocate first block.
    currentBlock = allocator.newWrite(stream, requiredBytes);
    firstBlock = currentBlock.header;
  } else {
    allocator.extendWrite(currentBlock, stream);
  }

  // Check if there is enough space left.
  if (stream.ranges().back().size <
      value.size() + HashStringAllocator::Header::kContinuedPtrSize) {
    // Not enough space. Allocate new block.
    ByteRange newRange;
    allocator.newContiguousRange(roundedUpBytes, &newRange);

    stream.setRange(newRange);
  }

  VELOX_DCHECK_LE(
      value.size() + HashStringAllocator::Header::kContinuedPtrSize,
      stream.ranges().back().size);

  // Copy the string and return a StringView over the copy.
  char* start = stream.writePosition();
  stream.appendStringPiece(folly::StringPiece(value.data(), value.size()));
  // There will always be at least enough space for the continue
  // pointer so the tail of the string does not get overwritten.
  currentBlock = allocator
                     .finishWrite(
                         stream,
                         HashStringAllocator::Header::kContinuedPtrSize +
                             roundedUpBytes - requiredBytes)
                     .second;
  return StringView(start, value.size());
}

void Strings::free(HashStringAllocator& allocator) {
  if (firstBlock != nullptr) {
    allocator.free(firstBlock);
    firstBlock = nullptr;
    currentBlock = {nullptr, nullptr};
  }
}
} // namespace facebook::velox::aggregate::prestosql
