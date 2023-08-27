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

#include <iostream>

#include <folly/init/Init.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Semaphore.h"
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/time/Timer.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/common/tests/CudaTest.h"
#include "velox/experimental/wave/common/tests/BlockTest.h"

DEFINE_int32(num_streams, 0, "Number of paralll streams");
DEFINE_int32(op_size, 0, "Size of invoke kernel (ints read and written)");
DEFINE_int32(
    num_ops,
    0,
    "Number of consecutive kernel executions on each stream");
DEFINE_bool(
    use_callbacks,
    false,
    "Queue a host callback after each kernel execution");
DEFINE_bool(
    sync_streams,
    false,
    "Use events to synchronize all parallel streams before calling the next kernel on each stream.");
DEFINE_bool(
    prefetch,
    true,
    "Use prefetch to move unified memory to device at start and to host at end");
DEFINE_int32(num_threads, 1, "Threads in reduce test");
DEFINE_int64(working_size, 100000000, "Bytes in flight per thread");
DEFINE_int32(num_columns, 10, "Columns in reduce test");
DEFINE_int32(num_rows, 100000, "Batch size in reduce test");
DEFINE_int32(num_batches, 100, "Batches in reduce test");


using namespace facebook::velox;
using namespace facebook::velox::wave;

// Dataset for data transfer test.
struct DataBatch {
  std::vector<BufferPtr> columns;
  // Sum of the int64s in buffers. Numbers below the size() of each are added up.
  int64_t sum{0};
  int64_t byteSize;
  int64_t dataSize;
};

/// Base class modeling processing a batch of data. Inits, continues and tests for ready.
struct ProcessBatchBase {
  // Starts processing 'batch'. Use isReady() to check for result.
  virtual void init(DataBatch* data, GpuArena* arena, folly::CPUThreadPoolExecutor* executor) {
    data_ = data;
    arena_ = arena;
    executor_ = executor;
    sums_.resize(data->columns.size());
    numRows_ = data_->columns[0]->size() / sizeof(int64_t);
    numBlocks_ = bits::roundUp(numRows_, 256) / 256;
    result_ = arena->allocate<int64_t>(data_->columns.size() * numBlocks_);

  }

  DataBatch* batch() {
    return data_;
  }
  // Returns true if ready and sets 'result'. Returns false if pending. If 'wait' is true, blocks until ready.
  virtual bool isReady(int64_t& result, bool wait) = 0;


protected:
  Device* device_{getDevice()};
  DataBatch* data_{nullptr};
  int32_t numBlocks_;
  int32_t numRows_;

  GpuArena* arena_{nullptr};
  folly::CPUThreadPoolExecutor* executor_{nullptr};
  std::vector<WaveBufferPtr> deviceBuffers_;
  std::vector<int64_t*> deviceArrays_;
  std::vector<int64_t> sums_;
  WaveBufferPtr result_;
  int64_t sum_{0};
  std::vector<std::unique_ptr<BlockTestStream>> streams_;
  std::vector<std::unique_ptr<Event>> events_;
  Semaphore sem_{0};
  int32_t toAcquire_{0};
};

class ProcessUnifiedN : public ProcessBatchBase {
public:
  void init(DataBatch* data, GpuArena* arena, folly::CPUThreadPoolExecutor* executor) override {
    ProcessBatchBase::init(data, arena, executor);
    
    deviceBuffers_.resize(data->columns.size());
    streams_.resize(deviceBuffers_.size());
    events_.resize(deviceBuffers_.size());
    toAcquire_ = data->columns.size();
    for (auto i = 0; i < data_->columns.size(); ++i) {
      deviceBuffers_[i] = arena->allocate<char>(data_->columns[i]->size());
      executor_->add([i, this]() {
	setDevice(device_);
	simd::memcpy(deviceBuffers_[i]->as<char>(), data_->columns[i]->as<char>(), data_->columns[i]->size());
	streams_[i] = std::make_unique<BlockTestStream>();
	streams_[i]->prefetch(device_, deviceBuffers_[i]->as<char>(), data_->columns[i]->size());
	auto resultIndex = i * numBlocks_;
	streams_[i]->testSum64(numBlocks_, deviceBuffers_[i]->as<int64_t>(), result_->as<int64_t>() + resultIndex);
	events_[i] = std::make_unique<Event>();
	events_[i]->record(*streams_[i]);
	sem_.release();
      });
    }
  }

  bool isReady(int64_t& result, bool wait) override {
    if (toAcquire_) {
      if (wait) {
	while (toAcquire_) {
	  sem_.acquire();
	  --toAcquire_;
	}
      } else {
	while (toAcquire_) {
	  if (sem_.count() == 0) {
	    return false;
	  }
	  sem_.acquire();
	  --toAcquire_;
	  
	}
      }
      
    }
    for (auto i = 0; i < events_.size(); ++i) {
      if (wait) {
	events_[i]->wait();
      } else {
	if (!events_[i]->query()) {
	  return false;
	}
      }
    }
    int64_t sum = 0;
    for (auto i = 0; i < data_->columns.size() * numBlocks_; ++i) {
      sum += result_->as<int64_t>()[i];
    }
    result = sum;
    return true;
  }


};


class CudaTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
  }

  void setupMemory(
		   int64_t capacity = 16UL << 30) {
    memory::MemoryManagerOptions options;
    options.capacity = capacity;
    memory::MmapAllocator::Options opts{(uint64_t)options.capacity};
    mmapAllocator_ = std::make_shared<memory::MmapAllocator>(opts);
    memory::MemoryAllocator::setDefaultInstance(mmapAllocator_.get());

    options.allocator = mmapAllocator_.get();
    manager_ = std::make_shared<memory::MemoryManager>(options);
  }

  

  
  void streamTest(
      int32_t numStreams,
      int32_t numOps,
      int32_t opSize,
      bool prefetch,
      bool useCallbacks,
      bool syncStreams) {
    int32_t firstNotify = useCallbacks ? 1 : numOps - 1;
    constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
    std::vector<std::unique_ptr<TestStream>> streams;
    std::vector<std::unique_ptr<Event>> events;
    std::vector<int32_t*> ints;
    std::mutex mutex;
    int32_t initValues[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto initVector = xsimd::load_unaligned(&initValues[0]);
    auto increment = xsimd::broadcast<int32_t>(1);
    std::vector<int64_t> delay;
    delay.reserve(numStreams * (numOps + 2));

    auto start = getCurrentTimeMicro();
    for (auto i = 0; i < numStreams; ++i) {
      streams.push_back(std::make_unique<TestStream>());
      ints.push_back(reinterpret_cast<int32_t*>(
          allocator_->allocate(opSize * sizeof(int32_t))));
      auto last = ints.back();
      auto data = initVector;
      for (auto i = 0; i < opSize; i += kBatch) {
        data.store_unaligned(last + i);
        data += increment;
      }
    }
    for (auto i = 0; i < numStreams; ++i) {
      streams[i]->addCallback([&]() {
        auto d = getCurrentTimeMicro() - start;
        {
          std::lock_guard<std::mutex> l(mutex);
          delay.push_back(d);
        }
      });
      if (prefetch) {
        streams[i]->prefetch(device_, ints[i], opSize * sizeof(int32_t));
      }
    }

    Semaphore sem(0);
    for (auto counter = 0; counter < numOps; ++counter) {
      if (counter > 0 && syncStreams) {
        waitEach(streams, events);
      }
      for (auto i = 0; i < numStreams; ++i) {
        streams[i]->addOne(ints[i], opSize);
        if (counter == 0 || counter >= firstNotify) {
          streams[i]->addCallback([&]() {
            auto d = getCurrentTimeMicro() - start;
            {
              std::lock_guard<std::mutex> l(mutex);
              delay.push_back(d);
            }
            sem.release();
          });
        }
        if (counter == numOps - 1) {
          if (prefetch) {
            streams[i]->prefetch(nullptr, ints[i], opSize * sizeof(int32_t));
          }
        }
      }
      if (syncStreams && counter < numOps - 1) {
        recordEach(streams, events);
      }
    }
    // Destroy the streams while items pending. Items should finish.
    streams.clear();
    for (auto i = 0; i < numStreams * (numOps + 1 - firstNotify); ++i) {
      sem.acquire();
    }
    for (auto i = 0; i < numStreams; ++i) {
      auto* array = ints[i];
      auto data = initVector + numOps;
      xsimd::batch_bool<int32_t> error;
      error = error ^ error;
      for (auto j = 0; j < opSize; j += kBatch) {
        error = error | (data != xsimd::load_unaligned(array + j));
        data += increment;
      }
      ASSERT_EQ(0, simd::toBitMask(error));
      delay.push_back(getCurrentTimeMicro() - start);
    }
    for (auto i = 0; i < numStreams; ++i) {
      allocator_->free(ints[i], sizeof(int32_t) * opSize);
    }
    std::cout << "Delays: ";
    int32_t counter = 0;
    for (auto d : delay) {
      std::cout << d << " ";
      if (++counter % numStreams == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
    float toDeviceMicros = delay[(2 * numStreams) - 1] - delay[0];
    float inDeviceMicros =
        delay[delay.size() - numStreams - 1] - delay[numStreams * 2 - 1];
    float toHostMicros = delay.back() - delay[delay.size() - numStreams];
    float gbSize =
        (sizeof(int32_t) * numStreams * static_cast<float>(opSize)) / (1 << 30);
    std::cout << "to device= " << toDeviceMicros << "us ("
              << gbSize / (toDeviceMicros / 1000000) << " GB/s)" << std::endl;
    std::cout << "In device (ex. first pass): " << inDeviceMicros << "us ("
              << gbSize * (numOps - 1) / (inDeviceMicros / 1000000) << " GB/s)"
              << std::endl;
    std::cout << "to host= " << toHostMicros << "us ("
              << gbSize / (toHostMicros / 1000000) << " GB/s)" << std::endl;
  }

  void recordEach(
      std::vector<std::unique_ptr<TestStream>>& streams,
      std::vector<std::unique_ptr<Event>>& events) {
    for (auto& stream : streams) {
      events.push_back(std::make_unique<Event>());
      events.back()->record(*stream);
    }
  }

  // Every stream waits for every event recorded on each stream in the previous
  // call to recordEach.
  void waitEach(
      std::vector<std::unique_ptr<TestStream>>& streams,
      std::vector<std::unique_ptr<Event>>& events) {
    auto firstEvent = events.size() - streams.size();
    for (auto& stream : streams) {
      for (auto eventIndex = firstEvent; eventIndex < events.size();
           ++eventIndex) {
        events[eventIndex]->wait(*stream);
      }
    }
  }

  void createData(int32_t numBatches, int32_t numColumns, int32_t numRows) {
    batchPool_ = memory::addDefaultLeafMemoryPool();
  int32_t sequence = 1;
  for (auto i = 0; i < numBatches; ++i) {
      auto batch = std::make_unique<DataBatch>();
      for (auto j = 0; j < numColumns; ++j) {
	auto buffer = AlignedBuffer::allocate<int64_t>(numRows, batchPool_.get(), sequence);
	batch->byteSize += buffer->capacity();
	batch->dataSize += buffer->size();
	batch->columns.push_back(buffer);
	batch->sum += numRows * sequence;
	++sequence;
      }
    }
  }

  DataBatch* getBatch() {
    auto number = ++batchIndex_;
    if (number > batches_.size()) {
      return nullptr;
    }
    return batches_[number - 1].get();
  }
  
  // 
  void processBatches(int64_t workingSize, GpuArena* arena, std::function<std::unique_ptr<ProcessBatchBase>()> factory) {
    int64_t pendingSize = 0;
    std::deque<std::unique_ptr<ProcessBatchBase>> work;
    for (;;) {
      int64_t result;
      auto * batch = getBatch();
      if (!batch) {
	for (auto& item : work) {
	  item->isReady(result, true);
	  EXPECT_EQ(item->batch()->sum, result);
	  processedBytes_ += item->batch()->dataSize;
	  pendingSize -= item->batch()->byteSize;
	  item.reset();
	}
	return;
      }
      if (pendingSize > workingSize) {
	work.front()->isReady(result, true);
	pendingSize -= work.front()->batch()->byteSize;
	processedBytes_+= work.front()->batch()->dataSize;
	EXPECT_EQ(result, work.front()->batch()->sum);
	work.pop_front();
      }
      auto item = factory();
      item->init(batch, arena, executor_.get());
      pendingSize += batch->byteSize;
      work.push_back(std::move(item));
      if (work.front()->isReady(result, false)) {
	EXPECT_EQ(result, work.front()->batch()->sum);
	pendingSize -= work.front()->batch()->byteSize;
	processedBytes_+= work.front()->batch()->dataSize;
	work.pop_front();
      }
    }
  }

  static std::unique_ptr<ProcessBatchBase> makeWork(){
    std::unique_ptr<ProcessBatchBase> ptr;
    ptr.reset(new ProcessUnifiedN());
    return ptr;
  }

  
  float reduceTest(int32_t numThreads, int64_t workingSize){
      std::vector<std::thread> threads;
  threads.reserve(numThreads);
  auto start = getCurrentTimeMicro();
  processedBytes_ = 0;
  batchIndex_ = 0;
  auto factory = makeWork;  for (int32_t i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread([&]() {
      auto arena = std::make_unique<GpuArena>(100 << 20, allocator_);
      processBatches(workingSize, arena.get(), factory);
    }));
  }
    for (auto& thread : threads) {
    thread.join();
  }

    auto time = getCurrentTimeMicro()  - start;
    float gbs = (processedBytes_ / 1024.0) / time;
    std::cout << time << "us " << gbs << " GB/s" << std::endl;
    return gbs;
  }

  std::shared_ptr<memory::MemoryPool> batchPool_;
std::vector<std::unique_ptr<DataBatch>> batches_;
  std::atomic<int32_t> batchIndex_{0};
  std::atomic<int64_t> processedBytes_{0};
  Device* device_;
  GpuAllocator* allocator_;
  std::shared_ptr<memory::MemoryManager> manager_;
  std::shared_ptr<memory::MmapAllocator> mmapAllocator_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
};

TEST_F(CudaTest, stream) {
  constexpr int32_t opSize = 1000000;
  TestStream stream;
  auto ints = reinterpret_cast<int32_t*>(
      allocator_->allocate(opSize * sizeof(int32_t)));
  for (auto i = 0; i < opSize; ++i) {
    ints[i] = i;
  }
  stream.prefetch(device_, ints, opSize * sizeof(int32_t));
  stream.addOne(ints, opSize);
  stream.prefetch(nullptr, ints, opSize * sizeof(int32_t));
  stream.wait();
  for (auto i = 0; i < opSize; ++i) {
    ASSERT_EQ(ints[i], i + 1);
  }
  allocator_->free(ints, sizeof(int32_t) * opSize);
  
  
}

TEST_F(CudaTest, callback) {
  streamTest(10, 10, 1024 * 1024, true, false, false);
}

TEST_F(CudaTest, custom) {
  if (FLAGS_num_streams == 0) {
    return;
  }
  streamTest(
      FLAGS_num_streams,
      FLAGS_num_ops,
      FLAGS_op_size,
      FLAGS_prefetch,
      FLAGS_use_callbacks,
      FLAGS_sync_streams);
}

TEST_F(CudaTest, copyReduce) {
  setupMemory();
  if (FLAGS_num_streams == 0) {
    return;
  }
  executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(64);
  createData(	     FLAGS_num_batches,
		     FLAGS_num_columns, FLAGS_num_rows);
  reduceTest(
	     FLAGS_num_threads,
	     FLAGS_working_size);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv);
  return RUN_ALL_TESTS();
}
