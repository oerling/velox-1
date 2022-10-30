
#include <stdio.h>
#include <semaphore>
#include <thread>
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/SymdUtil.h"
#include "velox/common/time/Timer.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
using facebook::velox;

int64_t sum(int64_t* data, int32_t size) {
  int64_t sum = 0;
  for (auto i = 0; i < size; ++i) {
    sum += data[i];
  }
  return sum;
}

struct CopyCallable {
  void* source;
  void* destination;
  int64_t bytes;
  std::counting_semaphore* sem;

  void operator()() {
    simd::memcpy(destination, source, bytes);
  }
};

// Arguments: repeats kb threads
// Copies 'kb' KB parallelized over 'threads'. Repeats the copy 'repeats' times.
int main(int argc, char** argv) {
  int32_t numRepeats = atoi(argv[1]);
  int32_t kb = atoi(argv[2]);
  int32_t numThreads = atoi(argv[3]);
  int64_t bytes = kb * 1024;
  auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(numThreads);
  void* other = malloc(bytes);
  void* source = malloc(bytes);
  void* destination = malloc(bytes);
  // Write both outside of timed section to make them resident.
  memset(other, 1, bytes);
  memset(source, 1, bytes);
  memset(destination, 1, bytes);

  std::vector<std::shared_ptr<AsyncSource<int32_t>>> sources;
  std::counting_semaphore sem;
  std::vector<CopyCallable> ops;
  ops.resize(numThreads);
  for (auto repeat = 0; repeat < numRepeats; ++repeat) {
    sum += sum(other, bytes / sizeof(int64_t));
    auto offset = 0;
    uint64_t usec = 0;
    {
      MicrosecondTimer timer(&usec);
      for (auto i = 0; i < numThreads; ++i) {
        int64_t offset1 = chunk * i / 64 * 64;
        int64_t offset2 = chunk * (i + 1) / 64 * 64 ops[i].source =
                              reinterpret_cast<char*>(source) + offset1;
        ops[i].destination = reinterpret_cast<char*>(destination) + offset1;
        ops[i].size = offset2 - offset1;
        sem.release();
        executor.add(ops[i]);
      }
      for (auto i = 0; i < numThreads; ++i) {
        sem.acquire();
      }
    }
    totalUsec += usec;
  }
  std::cout << fmt::format(
                   "{} repeats {} bytes {} threads: {} usec {} GB/s",
                   numRepeats,
                   bytes,
                   numThreads,
                   totalUsec,
                   bytes * repeats / static_cast<float>(1 << 30) /
                       (static_cast<float>(totalUsec + 1) / 1000000.0))
            << std::endl;
  return 0;
}
