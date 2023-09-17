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

#include "velox/dwio/common/ScanReusableData.h"
#include "velox/dwio/common/SelectiveColumnReader.h"

namespace facebook::velox::dwio::common {

std::unique_ptr<SelectiveColumnReader> ScanReusableData::getColumnReader(
    TypeKind kind) {
  int32_t index = static_cast<int32_t>(kind);
  std::lock_guard<std::mutex> l(mutex_);
  if (readers_.size() <= index) {
    return nullptr;
  }
  if (readers_[index].readers.empty()) {
    return nullptr;
  }
  auto result = std::move(readers_[index].readers.back());
  readers_[index].readers.pop_back();
  return result;
}

void ScanReusableData::releaseColumnReader(
    std::unique_ptr<SelectiveColumnReader> reader) {
  auto kind = reader->requestedType()->kind();
  auto index = static_cast<int32_t>(kind);
  switch (kind) {
    case TypeKind::REAL: {
      std::lock_guard<std::mutex> l(mutex_);
      if (readers_.size() <= index) {
        readers_.resize(index + 1);
      }
      readers_[index].readers.push_back(std::move(reader));
      return;
    }
    default: {
      auto children = reader->releaseChildren();
      for (auto& child : children) {
        releaseColumnReader(std::move(child));
      }
    }
  }
}

} // namespace facebook::velox::dwio::common
