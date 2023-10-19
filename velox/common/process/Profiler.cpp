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
#include <iostream>
#include "velox/common/file/File.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace facebook::velox::process {

int32_t
startCmd(std::string command, std::vector<std::string> args, int32_t& fd) {
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(command.c_str()));
  for (auto& a : args) {
    argv.push_back(const_cast<char*>(a.c_str()));
  }
  argv.push_back(nullptr);

  int fds[2];

  if (pipe(fds) == -1) {
    LOG(ERROR) << "Failed to make a pipe";
    return 0;
  }
  pid_t pid = fork();
  if (pid < 0) {
    LOG(ERROR) << "Fork failed";
    return 0;
  } else if (pid == 0) {
    // This code runs in the child process.
    close(fds[0]);
    int outputFd = fds[1];
    // Replace the child's stdout and stderr handles with the log file handle:
    if (dup2(outputFd, STDOUT_FILENO) < 0) {
      LOG(FATAL) << "Failed dup2";
    }
    if (dup2(outputFd, STDERR_FILENO) < 0) {
      LOG(FATAL) << "Failed dup";
      std::exit(1);
    }
    if (execvp(argv[0], argv.data()) < 0) {
      // These messages will actually go to parent.
      std::cerr << "Failed to exec program " << command << ":" << std::endl;
      std::perror("execl");
      std::exit(1);
    }
    std::exit(0);
  }
  close(fds[1]);
  fd = fds[0];
  return pid;
}

void waitCmd(int32_t pid, int32_t fd, std::string* result = nullptr) {
  char buffer[10000];
  while (auto bytes = read(fd, buffer, sizeof(buffer))) {
    if (bytes < 0) {
      LOG(INFO) << "PROFILE: Error reading child";
      break;
    }
    if (result) {
      *result += std::string(buffer, bytes);
    } else {
      LOG(INFO) << "PROFILE: " << std::string(buffer, bytes);
    }
  }
}

void execCmd(const std::string& cmd, std::vector<std::string> args) {
  int32_t fd;
  auto pid = startCmd(cmd, args, fd);
  if (!pid) {
    LOG(ERROR) << "PROFILE: Error in exec of " << cmd;
    return;
  }
  waitCmd(pid, fd);
}

bool Profiler::profileStarted_;
std::thread Profiler::profileThread_;
std::mutex Profiler::profileMutex_;
std::shared_ptr<velox::filesystems::FileSystem> Profiler::fileSystem_;
bool Profiler::isSleeping_;
bool Profiler::shouldStop_;
folly::Promise<bool> Profiler::sleepPromise_;

  void Profiler::copyToResult(int32_t counter, const std::string& path, std::string* data = nullptr) {
    char* buffer;
    int32_t resultSize;
    std::string temp;
    if (result) {
      buffer = result->data();
      resultSize = std::min<int32_t>(result->size(), 400000);
    } else {
      int32_t fd = open("/tmp/perf", O_RDONLY);
      if (fd < 0) {
	return;
      }
      auto bufferSize = 400000;
      temp.resize(400000);
      buffer = temp.data();
      resultSize = ::read(fd, buffer, bufferSize);
      close(fd);
      auto target = fmt::format("{}/prof-{}", path, counter);
    }
  }
  try {
    try {
      fileSystem_->remove(target);
    } catch (const std::exception& e) {
      // ignore
    }
    auto out = fileSystem_->openFileForWrite(target);
    out->append(std::string_view(buffer, readSize));
    out->flush();
    LOG(INFO) << "PROFILE: Produced result " << target << " " << resultSize
              << " bytes";
  } catch (const std::exception& e) {
    LOG(ERROR) << "PROFILE: Error opening/writing " << target << ":"
               << e.what();
  }
}

void Profiler::makeProfileDir(std::string path) {
  try {
    fileSystem_->mkdir(path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "PROFILE: Failed to create directory " << path << ":"
               << e.what();
  }
}

void Profiler::threadFunction(std::string path) {
  const int32_t pid = getpid();
  makeProfileDir(path);
  for (int32_t counter = 0;; ++counter) {
    std::thread systemThread([&]() {
#if 0
      system(
          fmt::format(
              "(cd /tmp; (/usr/bin/perf record --pid {} >>/tmp/perfstart.out);"
              "perf report --sort symbol > /tmp/perf;"
              "sed --in-place 's/      / /' /tmp/perf;"
	      "sed --in-place 's/      / /' /tmp/perf; date) "
	      ">> /tmp/perftrace 2>>/tmp/perftrace2",
              pid)
              .c_str());
#else
      int32_t fd;
      perfPid =
          startCmd("perf", {"record", "--pid", fmt::format("{}", pid)}, fd);
      waitCmd(perfPid, fd);
      int32_t reportFd;
      auto reportPid =
          startCmd("perf", {"report", "--sort", "symbol"}, reportFd);
      std::string report;
      waitCmd(reportPid, reportFd, report);

#endif
      copyToResult(counter, path);
    });
    folly::SemiFuture<bool> sleepFuture(false);
    {
      std::lock_guard<std::mutex> l(profileMutex_);
      isSleeping_ = true;
      sleepPromise_ = folly::Promise<bool>();
      sleepFuture = sleepPromise_.getSemiFuture();
    }
    if (!shouldStop_) {
      try {
        auto& executor = folly::QueuedImmediateExecutor::instance();
        std::move(sleepFuture)
            .via(&executor)
            .wait((std::chrono::seconds(counter < 2 ? 60 : 300)));
      } catch (std::exception& e) {
      }
    }
    {
      std::lock_guard<std::mutex> l(profileMutex_);
      isSleeping_ = false;
    }
    system("killall -2 perf");
    systemThread.join();
    if (shouldStop_) {
      return;
    }
  }
}

bool Profiler::isRunning() {
  std::lock_guard<std::mutex> l(profileMutex_);
  return profileStarted_;
}

void Profiler::start(const std::string& path) {
  {
#if !defined(linux)
    VELOX_FAIL("Profiler is only available for Linux");
#endif
    std::lock_guard<std::mutex> l(profileMutex_);
    if (profileStarted_) {
      return;
    }
    profileStarted_ = true;
  }
  fileSystem_ = velox::filesystems::getFileSystem(path, nullptr);
  if (!fileSystem_) {
    LOG(ERROR) << "PROFILE: Failed to find file system for " << path
               << ". Profiler not started.";
    return;
  }
  makeProfileDir(path);
  atexit(Profiler::stop);
  LOG(INFO) << "PROFILE: Starting profiling to " << path;
  profileThread_ = std::thread([path]() { threadFunction(path); });
}

void Profiler::stop() {
  {
    std::lock_guard<std::mutex> l(profileMutex_);
    shouldStop_ = true;
    if (isSleeping_) {
      sleepPromise_.setValue(true);
    }
  }
  profileThread_.join();
  {
    std::lock_guard<std::mutex> l(profileMutex_);
    profileStarted_ = false;
  }
  LOG(INFO) << "Stopped profiling";
}

} // namespace facebook::velox::process
