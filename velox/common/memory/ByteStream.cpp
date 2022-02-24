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

#include "velox/common/memory/ByteStream.h"

namespace facebook::velox {
void ByteStream::flush(OutputStream* out) {
  for (int32_t i = 0; i < ranges_.size(); ++i) {
    int32_t count = ranges_[i].numValues();
    int32_t bytes = isBits_ ? bits::nbytes(count) : count;
    if (isBits_ && isReverseBitOrder_ && !isReversed_) {
      bits::reverseBits(ranges_[i].buffer, bytes);
    }
    out->write(reinterpret_cast<char*>(ranges_[i].buffer), bytes);
  }
  if (isBits_ && isReverseBitOrder_) {
    isReversed_ = true;
  }
}

void ByteStream::extend(int32_t bytes) {
  if (current_) {
    current_->fill = current_->position;
  }
  if (current_ && current_ != &ranges_.back()) {
    ++current_;
    current_->fill = current_->position;
    if (current_->fill < current_->position) {
    }
    current_->position = 0;
    return;
  }
  ranges_.emplace_back();
  current_ = &ranges_.back();
  arena_->newRange(bytes, current_);
}

namespace {
void freeFunc(void* data, void* userData) {
  auto ptr = reinterpret_cast<std::shared_ptr<StreamArena>*>(userData);
  delete ptr;
}
} // namespace

std::unique_ptr<folly::IOBuf> IOBufOutputStream::getIOBuf() {
  // Make an IOBuf for each range. Transfer ownership of arena_ to
  // the IOBuf chain so that wen the last IOBuf of the chain is
  // destructed the arena is freed.
  std::unique_ptr<folly::IOBuf> iobuf;
  for (auto& range : out_->ranges()) {
    auto userData = new std::shared_ptr<StreamArena>(arena_);
    std::unique_ptr<folly::IOBuf> iobuf;
    auto newBuf = folly::IOBuf::takeOwnership(
        reinterpret_cast<char*>(range.buffer),
        range.position,
        freeFunc,
        userData);
    if (iobuf) {
      iobuf->prev()->appendChain(std::move(newBuf));
    } else {
      iobuf = std::move(newBuf);
    }
  }
}

std::streampos IOBufOutputStream::tellp() const {
  auto pos = out_->tellp();
  std::streampos offset = std::get<1>(pos);
  auto ranges = out_->ranges();
  for (auto& range : ranges) {
    if (&range == std::get<0>(pos)) {
      break;
    }
    offset += range.position;
  }
  return offset;
}

void IOBufOutputStream::seekp(std::streampos pos) {
  auto& ranges = out_->ranges();
  int64_t start = 0;
}

} // namespace facebook::velox
