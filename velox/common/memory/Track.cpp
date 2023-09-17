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

#include "velox/common/memory/Track.h"
#include <folly/container/F14Map.h>
#include <gflags/gflags.h>
#include <iostream>

DEFINE_bool(trace_malloc, false, "Count selected allocations");

namespace facebook::velox {

struct MKey {
  const char* file;
  int32_t line;
};

struct MKeyHasher {
  size_t operator()(const MKey& key) const {
    return key.line;
  }
};

struct MKeyComparer {
  bool operator()(const MKey& left, const MKey& right) const {
    return left.line == right.line;
  }
};

struct MCounts {
  int64_t bytes{0};
  int64_t count{0};
};

std::mutex trackMutex;
folly::F14FastMap<MKey, MCounts, MKeyHasher, MKeyComparer> tracked;

void mtrack(const char* file, int32_t line, int64_t bytes) {
  if (!FLAGS_trace_malloc) {
    return;
  }
  MKey key;
  key.file = file;
  key.line = line;

  std::lock_guard<std::mutex> l(trackMutex);
  auto it = tracked.find(key);
  if (it == tracked.end()) {
    MCounts counts = MCounts{1, bytes};
    tracked[key] = counts;
  } else {
    ++it->second.count;
    it->second.bytes += bytes;
  }
}
struct Item {
  MKey key;
  MCounts counts;
};
void mreport() {
  std::vector<Item> data;
  data.reserve(tracked.size());
  for (auto& pair : tracked) {
    data.emplace_back();
    data.back().key = pair.first;
    data.back().counts = pair.second;
  }
  std::sort(data.begin(), data.end(), [](const Item& left, const Item& right) {
    return left.counts.count > right.counts.count;
  });
  for (auto& item : data) {
    std::cout << item.counts.count << " " << item.counts.bytes << " at "
              << item.key.file << ":" << item.key.line << std::endl;
  }
}

} // namespace facebook::velox
