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

#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/futures/Future.h>

#include <folly/Random.h>
#include <folly/Synchronized.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <folly/portability/SysUio.h>
#include "velox/common/file/File.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "velox/common/time/Timer.h"

#include <gflags/gflags.h>

#include <iostream>

DEFINE_string(path, "ssdmeter.tmp", "Path of test file");
DEFINE_int64(size_gb, 2, "Test file size in GB");
DEFINE_int32(num_threads, 16, "Test paralelism");
DEFINE_bool(init_file, false, "Write initial contents to file");
DEFINE_int32(seed, 0, "Random seed, 0 means no seed");
DEFINE_bool(odirect, true, "Use O_DIRECT");

enum class Mode { Pread = 0, Preadv = 1, Multiple = 2 };

using namespace facebook::velox;

struct Scratch {
  std::string buffer;
  std::string bufferCopy;
};

class BM {
 public:
  BM() {
    executor_ =
        std::make_unique<folly::IOThreadPoolExecutor>(FLAGS_num_threads);
    fd_ = open(
        FLAGS_path.c_str(),
        O_CREAT | O_RDWR | (FLAGS_odirect ? O_DIRECT : 0),
        S_IRUSR | S_IWUSR);
    if (fd_ < 0) {
      LOG(ERROR) << "Could not open " << FLAGS_path;
      exit(1);
    }
    fileSize_ = FLAGS_size_gb << 30;
    numRegions_ = fileSize_ / kRegionSize;
    if (FLAGS_init_file) {
      auto rc = ftruncate(fd_, fileSize_);
      if (rc < 0) {
        LOG(ERROR) << "Could not resize file";
        exit(1);
      }
    }
    auto actualSize = lseek(fd_, 0, SEEK_END);
    fileSize_ = std::min(fileSize_, actualSize);
    readFile_ = std::make_unique<LocalReadFile>(fd_);
    if (FLAGS_seed) {
      rng_.seed(FLAGS_seed);
    }
    pins_.resize(numRegions_);
    if (FLAGS_init_file) {
      initFile();
    }
  }

  void initFile() {
#if 0
    for (int i = 0; i < numRegions_; ++i) {
      executor_->add([&]() { initRegion(i); });
    }
    auto& exec = folly::QueuedImmediateExecutor::instance();
    for (int32_t i = futures.size() - 1; i >= 0; --i) {
      std::move(futures[i]).via(&exec).wait();
    }
#endif
  }

  // Writes 'region' full of words that are the offset from the beginning of the
  // file.
  void initRegion(int region) {
    uint64_t offset = region * kRegionSize;
    writeBatch_.resize(writeBatchSize_);

    for (auto i = 0; i < kRegionSize / writeBatchSize_; ++i) {
      for (auto i = 0; i < writeBatch_.size(); i += sizeof(uint64_t)) {
        *reinterpret_cast<uint64_t*>(writeBatch_.data() + i) = offset + i;
      }
      std::vector<struct iovec> iovecs;
      fillIovecs(writeBatch_.data(), writeBatch_.size(), iovecs);
      auto rc = folly::pwritev(fd_, &iovecs[0], iovecs.size(), offset);
      DCHECK_EQ(rc, writeBatchSize_);
      offset += writeBatchSize_;
    }
  }

  void
  fillIovecs(char* data, int32_t bytes, std::vector<struct iovec>& iovecs) {
    LOG(FATAL) << "NYI";
#if 0
    int unit = 100;
    int32_t position = 0;
    while (position < size) {
      iovecs.push_back({data + position, std::min(size - position, unit)});
      position += unit;
      unit *= 2;
    }
#endif
  }

  void clearCache() {
    system("echo 3 >/proc/sys/vm/drop_caches");
  }

  Scratch& getScratch(int32_t size) {
    auto scratch = scratch_.withWLock([&](auto& table) {
      auto& ptr = table[std::this_thread::get_id()];
      if (!ptr) {
        ptr = std::make_unique<Scratch>();
      }
      ptr->buffer.resize(size);
      ptr->bufferCopy.resize(size);
      return ptr.get();
    });
    return *scratch;
  }

  void randomReads(
      int32_t size,
      int32_t gap,
      int32_t count,
      int32_t repeats,
      Mode mode,
      bool parallel) {
    clearCache();
    std::vector<folly::Promise<bool>> promises;
    std::vector<folly::SemiFuture<bool>> futures;
    uint64_t usec = 0;
    std::string label;
    {
      MicrosecondTimer timer(&usec);
      int32_t rangeSize = size * count + gap * (count - 1);
      auto& scratch = getScratch(rangeSize);
      scratch.buffer.resize(rangeSize);
      scratch.bufferCopy.resize(rangeSize);
      for (auto i = 0; i < repeats; ++i) {
        std::unique_ptr<folly::Promise<bool>> promise;
        if (parallel) {
          auto [tempPromise, future] = folly::makePromiseContract<bool>();
          promise = std::make_unique<folly::Promise<bool>>();
          *promise = std::move(tempPromise);
          futures.push_back(std::move(future));
        }
        int64_t offset = folly::Random::rand64(rng_) % (fileSize_ - rangeSize);
        switch (mode) {
          case Mode::Pread:
            label = "1 pread";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              size,
                              count,
                              rangeSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeSize);
                readFile_->pread(offset, rangeSize, scratch.buffer.data());
                for (auto i = 0; i < count; ++i) {
                  memcpy(
                      scratch.bufferCopy.data() + i * size,
                      scratch.buffer.data() + i * (size + gap),
                      size);
                }
                capturedPromise->setValue(true);
              }

              );
            } else {
              readFile_->pread(offset, rangeSize, scratch.buffer.data());
              for (auto i = 0; i < count; ++i) {
                memcpy(
                    scratch.bufferCopy.data() + i * size,
                    scratch.buffer.data() + i * (size + gap),
                    size);
              }
            }
            break;
          case Mode::Preadv: {
            label = "1 preadv";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              size,
                              count,
                              rangeSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeSize);
                std::vector<folly::Range<char*>> ranges;
                for (auto start = 0; start < rangeSize; start += size + gap) {
                  ranges.push_back(
                      folly::Range<char*>(scratch.buffer.data() + start, size));
                  if (gap && start + gap < rangeSize) {
                    ranges.push_back(folly::Range<char*>(nullptr, gap));
                  }
                }
                readFile_->preadv(offset, ranges);
                capturedPromise->setValue(true);
              });
            } else {
              std::vector<folly::Range<char*>> ranges;
              for (auto start = 0; start < rangeSize; start += size + gap) {
                ranges.push_back(
                    folly::Range<char*>(scratch.buffer.data() + start, size));
                if (gap && start + gap < rangeSize) {
                  ranges.push_back(folly::Range<char*>(nullptr, gap));
                }
              }
              readFile_->preadv(offset, ranges);
            }

            break;
          }
          case Mode::Multiple: {
            label = "multiple pread";
            if (parallel) {
              executor_->add([offset,
                              gap,
                              size,
                              count,
                              rangeSize,
                              this,
                              capturedPromise = std::move(promise)]() {
                auto& scratch = getScratch(rangeSize);
                for (auto counter = 0; counter < count; ++counter) {
                  readFile_->pread(
                      offset + counter * (size + gap),
                      size,
                      scratch.buffer.data() + counter * size);
                }
                capturedPromise->setValue(true);
              });
            } else {
              for (auto counter = 0; counter < count; ++counter) {
                readFile_->pread(
                    offset + counter * (size + gap),
                    size,
                    scratch.buffer.data() + counter * size);
              }
            }
            break;
          }
        }
      }
      if (parallel) {
        auto& exec = folly::QueuedImmediateExecutor::instance();
        for (int32_t i = futures.size() - 1; i >= 0; --i) {
          std::move(futures[i]).via(&exec).wait();
        }
      }
    }
    std::cout << fmt::format(
                     "{} MB/s {} {}",
                     (count * size * repeats) / usec,
                     label,
                     parallel ? "mt" : "")
              << std::endl;
  }

  void modes(int32_t size, int32_t gap, int32_t count) {
    int repeats = std::max<int32_t>(3, (100 << 20) / (size * count));
    std::cout << fmt::format(
                     "Run: {} Gap: {} Count: {} Repeats: {}",
                     size,
                     gap,
                     count,
                     repeats)
              << std::endl;
    randomReads(size, gap, count, repeats, Mode::Pread, false);
    randomReads(size, gap, count, repeats, Mode::Preadv, false);
    randomReads(size, gap, count, repeats, Mode::Multiple, false);
    randomReads(size, gap, count, repeats, Mode::Pread, true);
    randomReads(size, gap, count, repeats, Mode::Preadv, true);
    randomReads(size, gap, count, repeats, Mode::Multiple, true);
  }

  void run() {
    modes(1100, 0, 10);
    modes(1100, 1200, 10);
    modes(16 * 1024, 0, 10);
    modes(16 * 1024, 10000, 10);
    modes(1000000, 0, 8);
    modes(1000000, 100000, 8);
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
  std::unique_ptr<folly::IOThreadPoolExecutor> executor_;
  std::unique_ptr<ReadFile> readFile_;
  folly::Random::DefaultGenerator rng_;
  int64_t fileSize_;

  folly::Synchronized<
      std::unordered_map<std::thread::id, std::unique_ptr<Scratch>>>
      scratch_;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  BM bm;
  bm.run();
}
