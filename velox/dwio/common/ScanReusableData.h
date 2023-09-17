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

#include  <vector>
#include <memory>
#include <string>
#include <string_view>
#include "velox/common/memory/MemoryPool.h"
#include "velox/type/Type.h"




namespace facebook::velox::dwio::common {

class SelectiveColumnReader;

struct ReaderSet {
  std::vector<std::unique_ptr<SelectiveColumnReader>> readers;
};

/// An abstract class for reusable pieces of table scan for a particular query
/// and result memory pool. Useful if reading thousands of columns with frequent
/// construction of new reader trees with near identical column readers, streams
/// and decoders. The precise contents depend on the file format.
class ScanReusableData {
 public:
  ScanReusableData(
      const std::string& id,
      memory::MemoryPool* pool,
      std::function<void(ScanReusableData*)> freeFunc)
      : scanId_(id), pool_(pool), freeFunc_(freeFunc) {}

  virtual ~ScanReusableData() {
    freeFunc_(this);
  }

  std::pair<std::string_view, memory::MemoryPool*> key() {
    auto* temp = pool_;
    return std::make_pair<std::string_view, memory::MemoryPool*>(
        scanId_, std::move(temp));
  }

  std::unique_ptr<SelectiveColumnReader> getColumnReader(TypeKind kind);

    void releaseColumnReader(std::unique_ptr<SelectiveColumnReader> reader);

 protected:
  // Serializes any get/release.
  std::mutex mutex_;

  const std::string scanId_;
  memory::MemoryPool* pool_;
  std::function<void(ScanReusableData*)> freeFunc_;
  // Reusable readers, indexed on TypeKind.
  std::vector<ReaderSet> readers_;
};


}


