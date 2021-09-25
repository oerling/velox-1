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
#include <folly/Random.h>

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

  // Writes 'region' full of words that are the offset from the beginning of the file.
  void initRegion(int region) {
    uint64_t offset = region * kRegionSize;
    writeBatch_.resize(writeBatchSize_);
    
    for (auto i = 0; i < kRegionSize / batchSize; ++i) {

      for (auto i = 0; i < writeBatch_.size(); i += sizeof(uint64_t)) {
	*reinterpret_cast<uint64_t*>(writeBatch_.data() + i) = offset + i;
      }
      std::vector<struct iovec> iovecs;
      fillIovecs(writeBatch>.data(), wirteBatch_.size(), iovecs);
	   auto rc = folly::pwritev(&iovecs[0], iovecs.size(), offset);
      DCHECK_EQ(rc == writeBatchSize_);
      offset += writeBatchSize_;
    }
  }

  void fillIovecs(char* data, int32_t bytes, std::vector<struct iovec>& iovecs) {
    int unit = 100;
    int32_t position = 0;
    while (position < size) {
      iovecs.push_back({data + position, std::min (size - position, unit)});
      position += unit;
      unit *= 2;
    }
  }


  
private:
  static constexpr int64_t kRegionSize = 64 << 20; // 64MB
  static constexpr int32_t kWrite = -10000;
  // 0 means no op, kWrite means being written, other numbers are reader counts.
  std::vector<int32_t> pins_;
  int32_t numRegions_;
  int32_t writeBatchSize_{4 << 20};
  std::string writeBatch_;
  int32_t fd_;
  std::unique_ptr<folly::IoThreadPoolExecutor> executor_;
   folly::Random::DefaultGenerator rng_;
 
};






int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  Bm bm;
  bm.run();
}
