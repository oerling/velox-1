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

#include "velox/experimental/wave/exec/Wave.h"

namespace facebook::velox::wave {

WaveStream::~WaveStream() {
  // TODO: wait for device side work to finish before freeing associated memory
  // owned by exes and buffers in 'this'.
  for (auto& stream : streams_) {
    releaseStream(std::move(stream));
  }
  for (auto& event : allEvents_) {
    std::unique_ptr<Event> temp(event);
    releaseEvent(std::move(temp));
  }
}

std::mutex WaveStream::reserveMutex_;
std::vector<std::unique_ptr<Stream>> WaveStream::streamsForReuse_;
std::vector<std::unique_ptr<Event>> WaveStream::eventsForReuse_;
bool WaveStream::exitInited_{false};

Stream* WaveStream::newStream() {
  auto stream = streamFromReserve();
  auto id = streams_.size();
  stream->userData() = reinterpret_cast<void*>(id);
  auto result = stream.get();
  streams_.push_back(std::move(stream));
  lastEvent_.push_back(nullptr);
  return result;
}

// static
void WaveStream::clearReusable() {
  streamsForReuse_.clear();
  eventsForReuse_.clear();
}

// static
std::unique_ptr<Stream> WaveStream::streamFromReserve() {
  std::lock_guard<std::mutex> l(reserveMutex_);
  if (streamsForReuse_.empty()) {
    auto result = std::make_unique<Stream>();
    if (!exitInited_) {
      // Register handler for clearing resources after first call of API.
      exitInited_ = true;
      atexit(WaveStream::clearReusable);
    }

    return result;
  }
  auto item = std::move(streamsForReuse_.back());
  streamsForReuse_.pop_back();
  return item;
}

//  static
void WaveStream::releaseStream(std::unique_ptr<Stream>&& stream) {
  std::lock_guard<std::mutex> l(reserveMutex_);
  streamsForReuse_.push_back(std::move(stream));
}
Event* WaveStream::newEvent() {
  auto event = eventFromReserve();
  auto result = event.release();
  allEvents_.insert(result);
  return result;
}

// static
std::unique_ptr<Event> WaveStream::eventFromReserve() {
  std::lock_guard<std::mutex> l(reserveMutex_);
  if (eventsForReuse_.empty()) {
    return std::make_unique<Event>();
  }
  auto item = std::move(eventsForReuse_.back());
  eventsForReuse_.pop_back();
  return item;
}

//  static
void WaveStream::releaseEvent(std::unique_ptr<Event>&& event) {
  std::lock_guard<std::mutex> l(reserveMutex_);
  eventsForReuse_.push_back(std::move(event));
}

namespace {
// Copies from pageable host to unified address. Multithreaded memcpy is
// probably best.
void copyData(std::vector<Transfer>& transfers) {
  // TODO: Put memcpys or ppieces of them on AsyncSource if large enough.
  for (auto& transfer : transfers) {
    ::memcpy(transfer.to, transfer.from, transfer.size);
  }
}
} // namespace

void Executable::startTransfer(
    OperandSet outputOperands,
    WaveBufferPtr&& operands,
    std::vector<WaveVectorPtr>&& outputVectors,
    std::vector<Transfer>&& transfers,
    WaveStream& waveStream) {
  auto exe = std::make_unique<Executable>();
  exe->outputOperands = outputOperands;
  exe->output = std::move(outputVectors);
  exe->transfers = std::move(transfers);
  exe->deviceData = operands;
  exe->operands = operands->as<Operand>();
  exe->outputOperands = outputOperands;
  copyData(exe->transfers);
  auto* device = waveStream.device();
  waveStream.installExecutables(
      folly::Range(&exe, 1),
      [&](Stream* stream, folly::Range<Executable**> executables) {
        for (auto& transfer : executables[0]->transfers) {
          stream->prefetch(device, transfer.to, transfer.size);
        }
        waveStream.markLaunch(*stream, *executables[0]);
      });
}

void WaveStream::installExecutables(
    folly::Range<std::unique_ptr<Executable>*> executables,
    std::function<void(Stream*, folly::Range<Executable**>)> launch) {
  folly::F14FastMap<
      OperandSet,
      std::vector<Executable*>,
      OperandSetHasher,
      OperandSetComparer>
      dependences;
  for (auto& exeUnique : executables) {
    executables_.push_back(std::move(exeUnique));
    auto exe = executables_.back().get();
    VELOX_CHECK(exe->stream == nullptr);
    OperandSet streamSet;
    exe->inputOperands.forEach([&](int32_t id) {
      auto* source = operandToExecutable_[id];
      VELOX_CHECK(source != nullptr);
      auto stream = source->stream;
      if (stream) {
        // Compute pending, mark depenedency.
        auto sid = reinterpret_cast<uintptr_t>(stream->userData());
        streamSet.add(sid);
      }
    });
    dependences[streamSet].push_back(exe);
    exe->outputOperands.forEach([&](int32_t id) {
      VELOX_CHECK_EQ(0, operandToExecutable_.count(id));
      operandToExecutable_[id] = exe;
    });
  }

  // exes with no dependences go on a new stream. Streams with dependent compute
  // get an event. The dependent computes ggo on new streams that first wait for
  // the events.
  folly::F14FastMap<int32_t, Event*> streamEvents;
  for (auto& pair : dependences) {
    std::vector<Stream*> required;
    pair.first.forEach(
        [&](int32_t id) { required.push_back(streams_[id].get()); });
    Executable** start = pair.second.data();
    int32_t count = pair.second.size();
    auto exes = folly::range(start, start + count);
    if (required.empty()) {
      auto stream = newStream();
      launch(stream, exes);
    } else {
      for (auto* req : required) {
        auto id = reinterpret_cast<uintptr_t>(req->userData());
        if (streamEvents.count(id) == 0) {
          auto event = newEvent();
          lastEvent_[id] = event;
          event->record(*req);
          streamEvents[id] = event;
        }
      }
      auto launchStream = newStream();
      pair.first.forEach(
          [&](int32_t id) { streamEvents[id]->wait(*launchStream); });
      launch(launchStream, exes);
    }
  }
}

bool WaveStream::isArrived(
    const OperandSet& ids,
    int32_t sleepMicro,
    int32_t timeoutMicro) {
  OperandSet waitSet;
  ids.forEach([&](int32_t id) {
    auto exe = operandToExecutable_[id];
    VELOX_CHECK_NOT_NULL(exe);
    if (!exe->stream) {
      return;
    }
    auto streamId = reinterpret_cast<uintptr_t>(exe->stream->userData());
    if (!lastEvent_[streamId]) {
      lastEvent_[streamId] = newEvent();
      lastEvent_[streamId]->record(*exe->stream);
    }
    if (lastEvent_[streamId]->query()) {
      return;
    }
    waitSet.add(streamId);
  });
  if (waitSet.empty()) {
    return true;
  }
  if (sleepMicro == -1) {
    return false;
  }
  auto start = getCurrentTimeMicro();
  int64_t elapsed = 0;
  while (timeoutMicro == 0 || elapsed < timeoutMicro) {
    bool ready = true;
    waitSet.forEach([&](int32_t id) {
      if (!lastEvent_[id]->query()) {
        ready = false;
      }
    });
    if (ready) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(sleepMicro));
    elapsed = getCurrentTimeMicro() - start;
  }
  return false;
}

LaunchControl* WaveStream::prepareProgramLaunch(
    int32_t key,
    folly::Range<Executable**> exes,
    int32_t blocksPerExe,
    Stream* stream) {
  //  First calculate total size.
  // 3 int arrays: blockBase, programIdx, numParams.
  int32_t size = exes.size() * 3 * sizeof(int32_t) * blocksPerExe;
  // Arrays of pointers. program, actualParams,  paramCopy.
  size += exes.size() * sizeof(void*) * blocksPerExe * 3;
  // Exe dependent sizes for parameters.
  int32_t numTotalOps = 0;
  for (auto& exe : exes) {
    int numOps = exe->inputOperands.size() + exe->intermediates.size() +
        exe->outputOperands.size();
    numTotalOps += numOps * blocksPerExe;
    size += numOps * (sizeof(Operand) + sizeof(void*));
  }
  buffer = extraData_[key];
  if (!buffer || buffer->capacity() < size) {
    buffer = arena_.allocate(size);
    extraData_[key] = buffer;
  }
  LaunchControl control;
  // Now we fill in the various arrays and put their start addresses in
  // 'control'.
  auto start = buffer->as<int32_t>();
  int32_t numBlocks = exes.size() * blocksPerExe;
  control.blockBase = start;
  control.programIdx = start + numBlocks;
  control.numParams = start + numBlocks * 2;
  int32_t offset = bits::roundUp(numBlocks * 12, 8);
  void** start8 = buffer->as<void*>() + (start8 / 8) control.program =
                      reinterpret_cast<ThreadBlockProgram**>(start8);
  control.actualOperands = reinterpret_cast<Operand***>(start8 + numBlocks);
  control.operandCopies =
      reinterpret_cast<Operand**>(start8 + numBlocks + numAllOperands);
  int32_t fill = 0;
  for (auto exeIdx = 0; exeIdx < exes.size(); ++exeIdx) {
    int32_t numParams = exe->inputOperands.size() + exe->intermediates.size() +
        exe->outputOperands.size();
    // We get the actual input operands for the exe from the exes this depends
    // on and repeat them for each TB.
    paramTemp_.resize(numParams);
    int32_t nth = 0;
    exe->inputOperands.forEach([&](int32_t id) {
      inputExe = operandToExecutable_[id];
      paramTemp[nth++] = inputExe->outputParams[id];
    });
    for (auto tbIdx = 0; tbIdx < blocksPerExe; ++tbIdx) {
      control.blockBase[fill] = exeIdx * blocksPerExe;
      control.programIdx[fill] = exeIdx;
      control.numParams = numParams;
      control.program[fill] = exe->program;
      control.actualParams[fill] = paramFill;
      control.paramSpace[fill] = numParams * sizeof(Operand);
      memcpy(
          control.actualParams + actualFill,
          paramTemp_.data(),
          numParams * sizeof(void*));
      operandCopies += numParams * sizeof(Operand);
      paramAddress += numParams * sizeof(Operand);
      ++fill;
    }
  }
}

void Program::prepareForDevice(GpuArena& arena) {}

} // namespace facebook::velox::wave
