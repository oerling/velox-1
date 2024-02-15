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
#include <fstream>
#include <iostream>
#include "velox/common/file/File.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

DEFINE_string(profile_tmp_dir, "/tmp", "Writable temp for perf.data");

namespace facebook::velox::process {

constexpr int32_t kErrorToStdout = -2;
constexpr int32_t kMakeErrorPipe = -3;

int32_t startCmd(
    std::string command,
    std::vector<std::string> args,
    int32_t inFd,
    int32_t& outFd,
    int32_t& errorFdInOut,
    const char* dir = nullptr) {
  if (dir) {
    if (::chdir(dir) < 0) {
      LOG(ERROR) << "Failed to cd to " << dir;
    }
  }

  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(command.c_str()));
  for (auto& a : args) {
    argv.push_back(const_cast<char*>(a.c_str()));
  }
  argv.push_back(nullptr);

  int fds[2];
  int errorFds[2];

  if (errorFdInOut == kMakeErrorPipe) {
    if (pipe(errorFds) < 0) {
      LOG(FATAL) << "Failed to make error pipe";
    }
  }
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
    if (inFd >= 0) {
      if (dup2(inFd, STDIN_FILENO) < 0) {
        LOG(FATAL) << "Failed dup2 for input";
      }
    } else {
      close(fds[0]);
    }
    if (errorFdInOut == kMakeErrorPipe) {
      close(errorFds[0]);
    }
    int outputFd = fds[1];
    if (dup2(outputFd, STDOUT_FILENO) < 0) {
      LOG(FATAL) << "Failed dup2";
    }
    int32_t errorFd = errorFdInOut == kMakeErrorPipe ? errorFds[1]
        : errorFdInOut == kErrorToStdout             ? fds[1]
                                                     : errorFdInOut;
    if (dup2(errorFd, STDERR_FILENO) < 0) {
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
  if (errorFdInOut == kMakeErrorPipe) {
    close(errorFds[1]);
    errorFdInOut = errorFds[0];
  }
  outFd = fds[0];
  return pid;
}

void waitCmd(
    int32_t pid,
    int32_t fd,
    int32_t errorFd,
    std::string* result = nullptr,
    std::string* error = nullptr) {
  char buffer[10000];
  for (;;) {
    if (fd >= 0) {
      int32_t bytes = read(fd, buffer, sizeof(buffer));
      if (bytes <= 0) {
        if (bytes < 0) {
          LOG(INFO) << "PROFILE: Error reading child";
        }
        close(fd);
        fd = -1;
      } else {
        if (result) {
          *result += std::string(buffer, bytes);
        } else {
          LOG(INFO) << "PROFILE: " << std::string(buffer, bytes);
        }
      }
    }
    if (errorFd >= 0) {
      int32_t bytes = read(errorFd, buffer, sizeof(buffer));
      if (bytes <= 0) {
        if (bytes < 0) {
          LOG(INFO) << "PROFILE: Error reading child";
        }
        close(errorFd);
        errorFd = -1;
      } else {
        if (error) {
          *error += std::string(buffer, bytes);
        } else {
          LOG(INFO) << "PROFILE: stderr:" << std::string(buffer, bytes);
        }
      }
    }
    if (fd < 0 && errorFd < 0) {
      break;
    }
  }
}

void execCmd(
    const std::string& cmd,
    std::vector<std::string> args,
    const char* dir = nullptr) {
  int32_t fd;
  int32_t errorFd = kErrorToStdout;
  auto pid = startCmd(cmd, args, -1, fd, errorFd, dir);
  if (!pid) {
    LOG(ERROR) << "PROFILE: Error in exec of " << cmd;
    return;
  }
  waitCmd(pid, fd, -1);
}

bool Profiler::profileStarted_;
std::thread Profiler::profileThread_;
std::mutex Profiler::profileMutex_;
std::shared_ptr<velox::filesystems::FileSystem> Profiler::fileSystem_;
bool Profiler::isSleeping_;
bool Profiler::shouldStop_;
folly::Promise<bool> Profiler::sleepPromise_;

void testWritable(const std::string& dir) {
  auto testPath = fmt::format("{}/test", dir);
  int32_t fd = open(testPath.c_str(), O_RDWR | O_CREAT);
  if (fd < 0) {
    LOG(ERROR) << "Can't open " << testPath << " for write";
    return;
  }
  if (4 != write(fd, "test", 4)) {
    LOG(ERROR) << "Can't write to " << testPath;
  }
  close(fd);
}

void Profiler::copyToResult(
    int32_t counter,
    const std::string& path,
    const std::string* data) {
  char* buffer;
  int32_t resultSize;
  std::string temp;
  if (data) {
    buffer = const_cast<char*>(data->data());
    resultSize = std::min<int32_t>(data->size(), 400000);
  } else {
    testWritable(FLAGS_profile_tmp_dir);
    auto reportFile = fmt::format("{}/perf", FLAGS_profile_tmp_dir);
    int32_t fd = open(reportFile.c_str(), O_RDONLY);
    if (fd < 0) {
      LOG(ERROR) << "PROFILE: << Could not open report file at " << reportFile;
      return;
    }
    auto bufferSize = 400000;
    temp.resize(400000);
    buffer = temp.data();
    resultSize = ::read(fd, buffer, bufferSize);
    close(fd);
  }
  auto target = fmt::format("{}/prof-{}", path, counter);
  try {
    try {
      fileSystem_->remove(target);
    } catch (const std::exception& e) {
      // ignore
    }
    auto out = fileSystem_->openFileForWrite(target);
    out->append(std::string_view(buffer, resultSize));
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
    int32_t perfPid = 0;
    std::thread systemThread([&]() {
#if !defined(WITH_PIPE)
      system(fmt::format(
                 "(cd {}; /usr/bin/perf record --pid {};"
                 "perf report --sort symbol > perf ;"
                 "sed --in-place 's/      / /' perf;"
                 "sed --in-place 's/      / /' perf; date) "
                 ">> {}/perftrace 2>>{}/perftrace2",
                 FLAGS_profile_tmp_dir,
                 pid,
                 FLAGS_profile_tmp_dir,
                 FLAGS_profile_tmp_dir)
                 .c_str());
      copyToResult(counter, path);

#else
      int32_t fd;
      int32_t errorFd = kMakeErrorPipe;
      auto workingDir = FLAGS_profile_tmp_dir;
      perfPid = startCmd(
          "perf",
          {"record", "--pid", fmt::format("{}", pid) /*, "-m", "100" */},
          -1,
          fd,
          errorFd,
          workingDir.c_str());
      int32_t reportFd;
      auto reportPid = startCmd(
          "perf",
          {"report", "--sort", "symbol"},
          fd,
          reportFd,
          errorFd,
          workingDir.c_str());
      std::string report;
      std::string error;
      waitCmd(reportPid, reportFd, errorFd, &report, &error);
      wait(reportPid);
      wait(perfPid);
      LOG(INFO) << "PROFILE: stderr:" << error;
      copyToResult(counter, path, &report);

#endif
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
    LOG(INFO) << "PROFILE: Signalling perf at " << perfPid;
#if WITH_PIPE
    system(fmt::format("kill -2 {}", perfPid).c_str());
#else
    system("killall -2 perf");
#endif
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
