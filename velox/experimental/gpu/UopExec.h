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


#pragma once

#include <atomic>
#include <semaphore>

namespace facebook::velox::cuda {

  template <typename T>
  using vx_atomic<T> = std::atomic<T>
  
template <typename T>
struct Span {
  T* data;
  int32_t size;
};


struct Uop {
  // Marks that this is in the process reading its input and writing its output. Neither may be touched by other Uops.
  static constexpr int32_t kRunning = -1;

  // Added to 'state' for each input that does not have a value.
  static constexpr int32_t kInputNeeded = 1;

  // Added to 'state' to mark that an output has not been consumed. 
  static constexpr int32_t kOutputConsumed = 1000;
  
  vx_atomic<int32_t> state;

  
  
  // The  uOPS that must have a result for 'this' to be runnable.
  UopSpan inputs;

  // The Uops that need a result from 'this' to be runnable.
  Uopspan dependents;

  // The Uops whose result is needed only if 'this' becomes runnable
  UopSpan conditionalInputs;

  bool noMoreInput{false};
  bool finished{false};
};

  using UopSpan = Span<Uop* FOLLY_NONNULL>;
  
  struct Decompress : public Uop {
    // Double buffer for compressed input.
    Span<char> compressedData[2];
    Span<char> decompressedData;
    // Bit 0 decides between first and second compressedData.
    uint8_t counter{0};
  };
  
  
  // Commands from GPU to CPU
  enum class HostOp {
    kNeedData,
    kResultReady
  }

  // Commands from host to GPU
    enum GpuOp {
    kDataReady,
    kResultConsumed
  };

  struct HostCommand {
    HostOp op;
    int64_t taskId;
    union {
      struct {
	uint64_t fileId;
	uint64_t offset;
	int32_t size;
      } input;
      struct {
	// 0 when no more will be produced.
	int32_t numRows;
      } ready;
    } _;
  };

  struct GpuCommand {
    static constexpr kUopBatch = 16;
    GpuOp op;
    Task* task;
    union {
      struct {
	int32_t numUops;
	Uop* uops[kUopBatch];
	uint64_t uopData[kUopBatch];
      } input;
    } _;
  };
  
  
    
  template <  typename T>
  class RingBuffer {
    RingBuffer(Span<T> data)
      : data_(data),
	readSem_(0),
	writeSem_(data.size()),
	sizeMask_(data.size() - 1) {
      assert(data.size() == bits::nextPowerOfTwo(data.size());
    }

	void dequeue(T& data) {
	readSem_.acquire();
	auto index = readIndex_++ & sizeMask_;
	result = data_[index];
	writeSem_.release();
      }

      void enqueue() {
	writeSem_.acquire();
	int32_t index = writeIndex_++ & sizeMask_;
	data_[index] = data;
	readSem_.release();
      }
      
    private:
    Span<T> data;
    vx_atomic<int32_t> writeIndex_;
    vx_atomic<int32_t> readIndex_;
    Semaphore readSem;
    Semaphore writeSem;
    const int32_t sizeMask_;
    };
  
// Represents the state of a Task kernel
struct Task {
  // Runs with up to 'numParallel' concurrent  Uops. Returns When all Uops have a no more input state and have completed running.
  void run(int32_t numParallel);
  
  // Runs runnable 'Uops' in turn, returns when all Uops have a no more input state and none is runnable.
  void runThread();
  
  // Unique id
  int64_t id;
  vx_atomic<int32_t> numRunnable;
  RingBuffer<Uop* FOLLY_NONNULL> runnable_;
  Span<Uop*> uops;
  
};

  // Represents the state of interaction with one GPU
  struct GpuState {
    GpuState() {}

    void startTask(uint64_t id, UopSpan uops); 
    
      RingBuffer<HostCommand> hostCommands;
  RingBuffer<GpuCommand> gpuCommands;

  }
 

}

