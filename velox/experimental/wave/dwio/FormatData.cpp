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

#include "velox/experimental/wave/dwio/FormatData.h"

namespace facebook::velox::wave {

BufferId SplitStageing::add(Staging& staging) {
  staging_.push_back(staging);
  offsets_.push_back(fill_);
  fill_ += bits::roundUp(staging.size, 8);
  return offsets_.size() - 1;
}

void SplitStaging::registerPointer(BufferId id, void** ptr) {
  patch_.push_back(std::make_pair(id, ptr));
}

// Starts the transfers registered with add(). 'stream' is set to a stream
// where operations depending on the transfer may be queued.
void SplitStaging::transfer(WaveStream& stream, Stream*& stream) {
  deviceBuffer_ = waveStream.arena()->allocate(fill);
  auto universal = deviceBuffer_->as<char>();
  for (auto i = 0; i < offsets_.size(); ++i) {
    memcpy(universal + offsets_[i], staging_[i].hostData, staging_[i].size);
  }
  stream->PREFETCH(getDevice(), deviceBuffer_->as<char>, deviceBuffer_->size());
  for (auto& pair : patch_) {
    *pair.second += reinterpret_cast<int64_t>(universal) + offsets_[pair.first];
  }
}

BufferId ResultStageing::reserve(int32_t bytes) {
  offsets_.push_back(fill_);
  fill_ += bits::roundUp(bytes, 8);
  return offsets_.size() - 1;
}

void ResultStaging::registerPointer(BufferId id, void** pointer) {
  patch_.push_back(std::make_pair(id, pointer));
}

void ResultStaging::setReturnBuffer(GpuArena& arena, DecodePrograms& programs) {
  programs.result = arena.allocate(fill_);
  auto address = reinterpret_cast<int64_t>(programs.result->as<char>());
  // Patch all the registered pointers to point to buffer at offset offset_[id]
  // + the offset already in the registered pointer.
  for (auto& pair : patch_) {
    int64_t* pointer = reinterpret_cast<int64_t*>(pair.second);
    *pointer += address + offsets_[pair.first];
  }
}

} // namespace facebook::velox::wave
