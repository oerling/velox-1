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

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/portability/SysUio.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>


#include "velox/common/time/Timer.h"

#include <gflags/gflags.h>


DEFINE_string(path, "ssdmeter.tmp", "Path of test file");
DEFINE_int32(size_gb, 2, "Test file size in GB");
DEFINE_int32(num_threads, 16, "Test paralelism");
DEFINE_int(mode, 2, "0 = use pred/pwrite 1 = use pred for consecutive, 2 = use preadv for consecutive");
DEFINE_bool(init_file, false, "Write initial contents to file");

enum class Mode { Single = 0, SingleParallel = 1, Multiple = 2};

class BM {
public:
  BM() {
    executor = std::make_unique<folly::IOThreadPoolExecutor(4 * FLAGS_num_threads);
    fd_ = open(FLAGS_path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR);
    if (fd_ < 0) {
      LOG(ERROR) << "Could not open " << FLAGS_path;
      exit(1);
    }
    fileSize_ = FLAGS_file_gb << 30;
    numRegions_ = fileSize_ / kRegionSize;
    auto rc = ftruncate(fd_, fileSize_);
    if (rc < 0) {
      LOG(ERROR) << "Could not resize file";
      exit(1);
    }
    pins_.resize(numRegions_);
    if (FLAGS_init_file) {
      initFile();
    }
  }

  void initFile() {
    for (int i = 0; i < numRegions; ++i) {
      executor_->add([&]() { initRegion(i);});
    }
    executor->join();
  }

  void initRegion(int region) {
    
  }

  void run() {
    
  }
  
private:
  static constexpr int64_t kRegionSize = 64 << 20; // 64MB
  static constexpr int32_t kWrite = -10000;
  // 0 means no op, kWrite means being written, other numbers are reader counts.
  std::vector<int32_t> pins_;
  int32_t numRegions_;
  int32_t fd_;
  std::unique_ptr<folly::IoThreadPoolExecutor> executor_;
  
};











int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  run();
}
