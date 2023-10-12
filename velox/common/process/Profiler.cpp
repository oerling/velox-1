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

#include "velox/common/process/Profiler.h"
#include "velox/common/file/File.h"

#include <memory>
#include <mutex>
#include <fstream>
#include <iostream>
#include <thread>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <sys/types.h>
#include <unistd.h>

DEFINE_string(profile_path, "/tmp", "Path prefix for profiles");

namespace facebook::velox::process {

bool Profiler::profileStarted_;
std::thread Profiler::profileThread_;
std::mutex Profiler::profileMutex_;
std::shared_ptr<velox::filesystems::FileSystem> Profiler::fileSystem_;

void Profiler::copyToResult(int32_t counter, const std::string& task) {
  std::ifstream in(fmt::format("/tmp/perf{}", counter));
  in.seekg(std::ios_base::end);
  size_t size = in.tellg();
  in.seekg(std::ios_base::beg);
  auto bufferSize = std::min<size_t>(400000, size);
  char* buffer = reinterpret_cast<char*>(malloc(bufferSize));
  in.read(buffer, size);
  auto path = fmt::format("{}/{}/prof-{}", FLAGS_profile_path, task, counter);
  try {
    auto out = fileSystem_->openFileForWrite(path);
    out->append(std::string_view(buffer, bufferSize));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error opening/writing " << path << ":" << e.what();
  }
  ::free(buffer);
}

void Profiler::makeProfileDir(std::string task) {
  auto path = fmt::format("{}/{}", FLAGS_profile_path, task);
  try {
    fileSystem_->mkdir(path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to create directory " << path << ":" << e.what();
  }
}

void Profiler::threadFunction(std::string task) {
  int32_t counter = 0;
  const int32_t pid = getpid();
  makeProfileDir(task);
  for (;;) {
    std::thread systemThread([&]() {
      system(fmt::format(
          "cd /tmp; perf record --pid {};"
          "perf report --sort symbol > /tmp/perf{};"
          "sed --in-place 's/      / /' /tmp/perf{}; sed --in-place 's/      / /' /tmp/perf{}; ",
          pid,
          counter,
          counter,
          counter).c_str());
      copyToResult(counter, task);
    });
    std::this_thread::sleep_for(std::chrono::seconds(counter < 2 ? 60 : 300));
    system("killall -2 perf");
    systemThread.join();
  }
}

void Profiler::start(const std::string& task) {
  {
    std::lock_guard<std::mutex> l(profileMutex_);
    if (profileStarted_) {
      return;
    }
    profileStarted_ = true;
  }
  fileSystem_ = velox::filesystems::getFileSystem(FLAGS_profile_path, nullptr);
  if (!fileSystem_) {
    LOG(ERROR) << "Failed to find file system for " << FLAGS_profile_path << ". Profiler not started.";
    return;
  }
  makeProfileDir(task);
  profileThread_ = std::thread([&]() {
    threadFunction(task); });
}

} // namespace facebook::velox::process
