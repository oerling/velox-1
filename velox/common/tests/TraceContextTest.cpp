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

#include "velox/process/TraceContext.h"

#include <thread>

using namespace facebook::velox::process;

TEST(TraceContextTest, basic) {
  constexpr int32_t kNumThreads = 10;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.push_back(std::thread([&]() {
      TraceContext("process data");
      TraceContext(fmt::format("Processing chunk {}, i), true);
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }));

      }
      LOG(INFO) << TraceContext::statusLine();
  for (auto& thread : threads) {
    thread.join();
  }
      LOG(INFO) << TraceContext::statusLine();

      }




