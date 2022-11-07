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

#include "velox/experimental/gpu/Util.h"

namespace facebook::velox::cuda {

  class Task;
class Uop;

using UopSpan = Span<Uop * FOLLY_NONNULL>;

  // Device interpretable buffer. Interpretation depends on the receiving Uop.
  struct InputBuffer {
    char* data;
    int64_t size;
  };

  
  /// A collection of Uops that proceed in sync. 
  class Operator {

    vx_atomic<int32_t> numUopsStarted_;
    vx_atomic<int32_t> numUopsConsumed_;
    
    Task* task_;

    // Constituent uops. The running ones will be producing data for 'batch_'.
    UopSpan uops_;

    // Number of the batch being produced by the uops. 
    int64_t batch_;

    bool inputProcessed_;
  };


  enum class UopKind uint8_t {kDecompress, kColumnScan, kFilter, kProject, kAggregate};

  // Function pointer types for 'virtual' functions. A real virtual function does not work because instances are made on host and not device.
 
  using UopRunFunc = void(*)(Uop* FOLLY_NONNULL uop);

  /// Arrays of function pointers for virtual functions. Subscript is UopKind. 
  Span(UopRunFunc) inputReadyFuncs;
  Span(UopRunFunc) OutputConsumedFuncs;

  // Initializes function pointer arrays and other global state.
  void globalInit();
  
  


class Uop {
 public:
  Uop(UopKind kind, Operator& operator, UopSpan inputs, UopSpan conditionalInputs, UopSpan consumers)
    : kind_(kind),
      operator_(operator),
      state_(inputs.empty() ? kInputNeeded : inputs.size() * kInputNeeded),
        inputs(inputs_),
        conditionalInputs_(conditionalInputs),
        consumers_(consumers)

        // Marks that this is in the process reading its input and writing its
        // output. Neither may be touched by other Uops.
        static constexpr int32_t kRunning = -1;

  // Added to 'state' for each input that does not have a value.
  static constexpr int32_t kInputNeeded = 1;

  // Added to 'state' to mark that an output has not been consumed.
  static constexpr int32_t kOutputConsumed = 1000;

  void setInput(InputBuffer input) {
    input_ = input;
  }

  // Called by producer to signal its input is ready.
  void inputReady();

  // Called by this to signal to consumers that their input is ready.
  void outputReady();

  // Called by 'this' after noMoreInput() has been called on 'this' and all
  // output generated.
  void noMoreOutput();

  // Called by 'this' to signal that the producers are free to rewrite the input
  // 'this' consumed.
  void inputConsumed();

  // Called by producers to signal that they will not produce any more.
  void noMoreInput();


  // Points to valid output when calling outputReady().
  Span<Vector> output_;
  
  Task* task_;

  vx_atomic<int32_t> state_;

  // The  uOPS that must have a result for 'this' to be runnable.
  UopSpan inputs_;

  // The Uops that need a result from 'this' to be runnable.
  Uopspan consumers_;

  // The Uops whose result is needed only if 'this' becomes runnable
  UopSpan conditionalInputs_;


  const UopKind kind_;
  bool noMoreInput_{false};
  bool finished_{false};
};

class Decompress : public Uop {
  // Double buffer for compressed input.
  Span<char> compressedData_[2];
  Span<char> decompressedData_;
  // Bit 0 decides between first and second compressedData.
  uint8_t counter{0};
};

  class ColumnScan : public Uop {
    // The row where the read starts when this is next runnable.
    int64_t* startRow_;

    // The first unprocessed row 
    int64_t currentRow_;

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

template <typename T>
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
  void inputReady(UopSpan leaves, Span<InputBuffer> input);

  // Runs with up to 'numParallel' concurrent  Uops. Returns When all Uops have
  // a no more input state and have completed running.
  void run(int32_t numParallel);

  // Runs runnable 'Uops' in turn, returns when all Uops have a no more input
  // state and none is runnable.
  void runThread();

  // Unique id
  int64_t id;
  vx_atomic<int32_t> numRunnable;
  RingBuffer<Uop * FOLLY_NONNULL> runnable_;
  Span<Uop*> uops;
};

// Represents the state of interaction with one GPU
struct GpuState {
  GpuState() {}

  void startTask(uint64_t id, UopSpan uops);

  RingBuffer<HostCommand> hostCommands;
  RingBuffer<GpuCommand> gpuCommands;
}

} // namespace facebook::velox::cuda
