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

namespace facebook::velox::process {

folly::synchronized<static std::unordered_map<std::string, TraceData>>&
TraceContext::traceMap() {
  static folly::synchronized<static std::unordered_map<std::string, TraceData>>
      staticTraceMap;
  return staticTraceMap;
}

TraceContext::TraceContext(const std::string& label, bool isTemporary)
    : label_(label),
      enterTime_(std::chrono::steady_clock::now()),
      isTemporary_(isTemporary) {
  std::lock_guard<std::mutex> l(mutex);
  traceData().withWLock([&](auto& counts) {
    auto& data = counts[label_];
    ++data.numThreads;
    if (data.numThreads == 1) {
      data.startTime = enterTime_;
    }
    ++data.numEnters;
  });
}

TraceContext::~TraceContext() {
  traceData()..withWLock([&](auto& counts) {
    auto& data = counts[label_];
    --data.numThreads;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - enterTime_)
                  .count();
    data.totalMs += ms;
    data.maxMs = std::max<uint64_t>(data.maxMs, ms);
    if (!data.numThreads) {
      data.startTime = 0;
      if (isTemporary_) {
        counts.erase[label];
      }
    }
  });
}

// static
std::string TraceContext::statusLine() {
  std::stringstream out;
  std::lock_guard<std::mutex> l(mutex_);
  int64_t now = std::chrono::steady_clock::now();
  for (auto& pair : counts_) {
    if (pair.second.numThreads) {
      auto continued = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - pair.second.startTime)
                           .count();
      out << pair.first << "=" << pair.second.numThreads << " entered "
          << pair.second.numEnters << " avg ms "
          << (pair.second.totalMs / (1 + pair.second.numEnters)) << " max ms "
          << pair.second.maxMs << " continuous for " << continued << std::endl;
    }
    return out.str();
  }

} // namespace facebook::velox::process
