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
#include "velox/experimental/wave/exec/Vectors.h"

namespace facebook::velox::wave {

const SubfieldMap*& threadSubfieldMap() {
  thread_local const SubfieldMap* subfields;
  return subfields;
}

std::string definesToString(const DefinesMap* map) {
  std::stringstream out;
  for (const auto& [value, id] : *map) {
    out
        << (value.subfield ? value.subfield->toString()
                           : value.expr->toString(1));
    out << " = " << id->id << " (" << id->type->toString() << ")" << std::endl;
  }
  return out.str();
}

AbstractOperand* pathToOperand(
    const DefinesMap& map,
    std::vector<std::unique_ptr<common::Subfield::PathElement>>& path) {
  if (path.empty()) {
    return kNoOperand;
  }
  common::Subfield field(std::move(path));
  const auto subfieldMap = threadSubfieldMap();
  auto it = threadSubfieldMap()->find(field.toString());
  if (it == subfieldMap->end()) {
    return kNoOperand;
  }
  Value value(it->second.get());
  auto valueIt = map.find(value);
  path = std::move(field.path());
  if (valueIt == map.end()) {
    return nullptr;
  }
  return valueIt->second;
}

WaveVector* Executable::operandVector(OperandId id) {
  WaveVectorPtr* ptr = nullptr;
  if (outputOperands.contains(id)) {
    auto ordinal = outputOperands.ordinal(id);
    ptr = &output[ordinal];
  }
  if (localOperands.contains(id)) {
    auto ordinal = localOperands.ordinal(id);
    ptr = &intermediates[ordinal];
  }
  if (*ptr) {
    return ptr->get();
  }
  return nullptr;
}

WaveVector* Executable::operandVector(OperandId id, const TypePtr& type) {
  WaveVectorPtr* ptr = nullptr;
  if (outputOperands.contains(id)) {
    auto ordinal = outputOperands.ordinal(id);
    ptr = &output[ordinal];
  } else if (localOperands.contains(id)) {
    auto ordinal = localOperands.ordinal(id);
    ptr = &intermediates[ordinal];
  } else {
    VELOX_FAIL("No local/output operand found");
  }
  if (*ptr) {
    return ptr->get();
  }
  *ptr = WaveVector::create(type, waveStream->arena());
  return ptr->get();
}

WaveStream::~WaveStream() {
  // TODO: wait for device side work to finish before freeing associated memory
  // owned by exes and buffers in 'this'.
  for (auto& exe : executables_) {
    if (exe->releaser) {
      exe->releaser(exe);
    }
  }
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
  exe->deviceData.push_back(operands);
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
  for (auto& [ids, exeVector] : dependences) {
    folly::Range<Executable**> exes(exeVector.data(), exeVector.size());
    std::vector<Stream*> required;
    ids.forEach([&](int32_t id) { required.push_back(streams_[id].get()); });
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
      ids.forEach([&](int32_t id) { streamEvents[id]->wait(*launchStream); });
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

void WaveStream::exeLaunchInfo(
    Executable& exe,
    int32_t numBlocks,
    ExeLaunchInfo info) {
  // The exe has an Operand* for each input/local/output/literal
  // op. It has an Operator for each local/output/literal op. It has
  // an array of numBlock int32_t*'s for every distinct wrapAt in
  // its local/output operands where the wrapAt does not occur in
  // any of the input Operands.
  info.numBlocks = numBlocks;
  exe.input.forEach([&](auto id) {
    auto op = operandAt(id);
    if (op->wrapAt != AbsttractOperand::kNoWrap) {
      inputExe = operandExecutable(op->id);
      indices = inputExe->wraps[op->wrapAt];
      VELOX_CHECK_NOT_NULL(indices);
      info.inputWrap[op->wrapAt] = indices;
    }
  });

  exe.local.forEach([&](auto id) {
    auto op = operandAt(id);
    if (op->wrapAt != AbstractOperand::kNoWrap) {
      if (inputWrap.find(id) == inputWrap.end()) {
        if (info.localWrap.find(op->wrapAt) != info.localWrap.end()) {
          localWrap[op->wrapAt] = localWrap.size() * numBlocks * sizeof(void*);
        }
      }
    }
  });
  exe.output.forEach([&](auto id) {
    auto op = operandAt(id);
    if (op->wrapAt != AbstractOperand::kNoWrap) {
      if (inputWrap.find(id) == inputWrap.end()) {
        if (info.localWrap.find(op->wrapAt) != info.localWrap.end()) {
          localWrap[op->wrapAt] = localWrap.size() * numBlocks * sizeof(void*);
        }
      }
    }
  });
  auto numLiteral = exe->literalOperands ? exe->literalOperands->size() : 0;
  auto numLocalOps = exe.local.size() + exe.output.size() + numLiteral;
  info.totalBytes =
      // Pointer to Operand for input and local Operands.
      sizeof(void*) * (numLocalOps + exe.input.size()) +
      // Flat array of Operand for all but input.
      sizeof(Operand) * numLocalOps +
      // Space for the 'indices' for each distinct wrapAt.
      (localWrap.size() * numBlocks * sizeof(void*));
}

Operand**
WaveStream::fillOperands(Executable& exe, char* start, ExeLaunchInfo& info) {
  OperandPtr*** operandPtrBegin = addBytes<OperandPtr***>(start, 0);
  exe->inputOperands.forEach([&](int32_t id) {
    auto* inputExe = operandToExecutable_[id];
    int32_t ordinal = inputExe->outputOperands.ordinal(id);
    *operandPtrBegin = &inputExe->operands[ordinal];
    ++operandPtrBegin;
  });
  Operand* operandBegin = addBytes<Operand*>(
      start, (info.numInput + info.numLocalOps) * sizeof(void*));
  int32_t* indicesBegin =
      addBytes<int32_t*>(operandBegin, info.numLocalOps * sizeof(Operand));
  for (auto& [id, ptr] : info.localWrap) {
    info.localWrap[id] =
        addBytes<int32_t*>(indicesBegin, reinterpret_cast<int64_t>(ptr));
  }
  exe.wrap = std::move(info.localWrap);
  for (auto& [id, ptr] : info.inputWrap) {
    exe.wrap[id] = ptr;
  }
  exe.intermediate.forEach([&](auto id) {
    auto op = operandAt(id);
    auto vec = getVector(op);
    vec->toOperand(operandBegin);
    if (op->wrapAt != AbstractOperand::kNoWrap) {
      operandBegin->indices = exe.wrap[op->wrapAt];
      VELOX_CHECK_NOT_NULL(operandBegin->indices);
    }
    *operandPtrBegin = operandBegin;
    ++operandPtrBegin;
    ++operandBegin;
  });
  exe.outputOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    auto vec = getVector(op);
    vec->toOperand(operandBegin);
    if (op->wrapAt != AbstractOperand::kNoWrap) {
      operandBegin->indices = exe.wrap[op->wrapAt];
      VELOX_CHECK_NOT_NULL(operandBegin->indices);
    }
    *operandPtrBegin = operandBegin;
    ++operandPtrBegin;
    ++operandBegin;
  });

  auto numConstants = exe.literals ? exe.literals->size() : 0;
  if (numConstants) {
    memcpy(operandBegin, exe->literals->data(), numConstants * sizeof(Operand));
    for (auto i = 0; i < numConstants; ++i) {
      *operandPtrBegin = operandBegin;
      ++operandPtrBegin;
      ++operandBegin;
    }
  }

  return addBytes<OperandPtr**>(start, 0);
}

LaunchControl* WaveStream::prepareProgramLaunch(
    int32_t key,
    int32_t inputRows,
    folly::Range<Executable**> exes,
    int32_t blocksPerExe,
    bool initStatus,
    Stream* stream) {
  static_assert(Operand::kPointersInOperand * sizeof(void*) == sizeof(Operand));
  int32_t shared = 0;

  //  First calculate total size.
  // 2 int arrays: blockBase, programIdx.
  int32_t numBlocks = std::min<int32_t>(1, exes.size()) * blocksPerExe;
  int32_t size = 2 * numBlocks * sizeof(int32_t);
  std::vector<ExeLaunchInfo> info(exes.size());
  auto exeOffset = size;
  // 2 pointers per exe: TB program and start of its param array.
  size += exes.size() * sizeof(void*) * 2;
  auto operandOffset = size;
  // Exe dependent sizes for parameters.
  for (auto i = 0; i < exes.size(); ++i) {
    launchInfo(exes[i], numBlocks, info[i]);
    paramBytes += info[i].totalBytes;
    markLaunch(*stream, *exe);
    shared = std::max(shared, exe->programShared->sharedMemorySize());
  }
  size += paramBytes;
  int32_t statusOffset = 0;
  if (initStatus) {
    statusOffset = size;
    //  Pointer to return block for each tB.
    size += blocksPerExe * sizeof(BlockStatus);
  }
  auto buffer = arena_.allocate<char>(size);

  auto controlUnique = std::make_unique<LaunchControl>(key, inputRows);
  auto& control = *controlUnique;

  control.sharedMemorySize = shared;
  // Now we fill in the various arrays and put their start addresses in
  // 'control'.
  auto start = buffer->as<int32_t>();
  control.blockBase = start;
  control.programIdx = start + numBlocks;
  control.programs = addBytes<ThreadBlockProgram**>(
      control.programIdx, numBlocks * sizeof(int32_t));
  control.operands =
      addBytes<Operand***>(control.programs, exes.size() * sizeof(void*));
  if (initStatus) {
    // If the launch produces new statuses (as opposed to updating status of a
    // previous launch), there is an array with a status for each TB. If there
    // are multiple exes, they all share the same error codes. A launch can have
    // a single cardinality change, which will update the row counts in each TB.
    // Writing errors is not serialized but each lane with at least one error
    // will show one error.
    control.status = addBytes<BlockStatus*>(start, statusOffset);
    memset(control.status, 0, blocksPerExe * sizeof(BlockStatus));
    for (auto i = 0; i < blocksPerExe; ++i) {
      auto status = &control.status[i];
      status->numRows =
          i == blocksPerExe - 1 ? inputRows % kBlockSize : kBlockSize;
    }
  } else {
    control.status = nullptr;
  }

  for (auto i = 0; i < exes.size(); ++i) {
    control.programs[i] = exes[i]->program;
    control.operands[i] = fillOperands(exes[i], paramAreaStart, info[i]);
    paramAreaStart += info[i].totalBytes;
  }
  if (numConstants) {
    memcpy(
        operandArrayBegin,
        exe->literals->data(),
        numConstants * sizeof(Operand));
    for (auto i = 0; i < numConstants; ++i) {
      *operandPtrBegin = operandArrayBegin;
      ++operandPtrBegin;
      ++operandArrayBegin;
    }
  }
    for (auto tbIdx = 0; tbIdx < blocksPerExe; ++tbIdx) {
      control.blockBase[fill] = exeIdx * blocksPerExe;
      control.programIdx[fill] = exeIdx;
    }
  }
  control.deviceData = std::move(buffer);
  launchControl_[key].push_back(std::move(controlUnique));
  return &control;
}

void WaveStream::getOutput(
    folly::Range<const OperandId*> operands,
    WaveVectorPtr* waveVectors) {
  for (auto i = 0; i < operands.size(); ++i) {
    auto id = operands[i];
    auto exe = operandExecutable(id);
    VELOX_CHECK_NOT_NULL(exe);
    auto ordinal = exe->outputOperands.ordinal(id);
    waveVectors[i] = std::move(exe->output[ordinal]);
    if (waveVectors[i] == nullptr) {
      exe->ensureLazyArrived(operands);
      waveVectors[i] = std::move(exe->output[ordinal]);
      VELOX_CHECK_NOT_NULL(waveVectors[i]);
    }
  }
}

WaveTypeKind typeKindCode(TypeKind kind) {
  return static_cast<WaveTypeKind>(kind);
}

#define IN_HEAD(abstract, physical, _op)             \
  auto* abstractInst = &instruction->as<abstract>(); \
  space->opCode = _op;                               \
  auto physicalInst = new (&space->_) physical();

#define IN_OPERAND(member) \
  physicalInst->member = operandIndex(abstractInst->member)

void Program::prepareForDevice(GpuArena& arena, OperandSet& wrappedOperands) {
  int32_t codeSize = 0;
  int32_t sharedMemorySize = 0;
  for (auto& instruction : instructions_)
    switch (instruction->opCode) {
      case OpCode::kFilter: {
        auto& filter = instruction->as<AbstractFilter>();
        markInput(filter.flags);
        markResult(filter.indices);
        break;
      }
      case OpCode::kWrap: {
        auto& wrap = instruction->as<AbstractWrap>();
        markInput(wrap.indices);
        std::vector<OperandIndex> indices(wrap.targets.size());
        wrap.literalOffset = addLiteral(indices.data(), indices.size());
        for (auto i = 0; i < wrap.target.size(); ++i) {
          auto target = wrap.target[i];
          markInput(wrap->source[i]);
          if (wrappedOperands.contains(target->id)) {
            continue;
          }
          if (target != wrap->source[i]) {
            markResult(target);
          }
          wrappedOperands_.add(target->id);
          operandsWithIndices_.add(target->id);
        }
        break;
      }
      case OpCode::kPlus: {
        auto& bin = instruction->as<AbstractBinary>();
        markInput(bin.left);
        markInput(bin.right);
        markResult(bin.result);
        markInput(bin.predicate);
        codeSize += sizeof(Instruction);
        break;
      }
      case OpCode::kNegate: {
        auto& un = instruction->as<AbstractUnary>();
        markInput(un.input);
        markResult(un.result);
        markInput(un.predicate);
        codeSize += sizeof(Instruction);
        break;
      }

      default:
        VELOX_UNSUPPORTED(
            "OpCode {}", static_cast<int32_t>(instruction->opCode));
    }
  sortSlots();
  arena_ = &arena;
  deviceData_ = arena.allocate<char>(
      codeSize + instructions_.size() * sizeof(void*) + literalArea_.size() +
      sizeof(ThreadBlockProgram));
  program_ = deviceData_->as<ThreadBlockProgram>();
  auto instructionArray = addBytes<Instruction**>(program_, sizeof(*program_));
  program_->sharedMemorySize = sharedMemorySize;
  program_->numInstructions = instructions_.size();
  program_->instructions = instructionArray;
  Instruction* space = addBytes<Instruction*>(
      instructionArray, instructions_.size() * sizeof(void*));
  for (auto& instruction : instructions_) {
    *instructionArray = space;
    ++instructionArray;
    switch (instruction->opCode) {
      case OpCode::kPlus: {
        IN_HEAD(
            AbstractBinary,
            binary,
            OP_MIX(
                instruction->opCode,
                instruction->as<AbstractBinary>().left->type()->kind()));

        IN_OPERAND(left);
        IN_OPERAND(right);
        IN_OPERAND(result);
        IN_OPERAND(predicate);
        break;
      }
      case OpCode::kFilter: {
        IN_HEAD(AbstractFilter, IFilter, OpCode::kFilter);
        IN_OPERAND(flags);
        IN_OPERAND(indices);
        ++space;
        break;
      }
      default:
        VELOX_UNSUPPORTED("Bad OpCode");
    }
  }
  char* literalArea = reinterpret_cast<char*> space;
  memcpy(literalArea, literalArea_.data(), constantArea_.size());
  literalOperands_.resize(constants_.size());
  int32_t counter = 0;
  for (auto& [op, id] : constants_) {
    literalToOperand(op, literalOperands[counter]);
  }
}
void Program::literalToOperand(AbstractOperand* abstractOp, Operand& op) {
  op.indexMask = 0;
  op.indices = nullptr;
  if (abstractOp->constantNull) {
    op.nulls = deviceConstants + op->constantOffset;
  } else {
    op.base = deviceConstants_ + abstractOp->constantOffset;
  }
}

void Program::sortSlots() {
  // Assigns offsets to input and local/output slots so that all
  // input is first and output next and within input and output, the
  // slots are ordered with lower operand id first. So, if inputs
  // are slots 88 and 22 and outputs are 77 and 33, then the
  // complete order is 22, 88, 33, 77. Constants are sorted after everything
  // else.
  std::vector<AbstractOperand*> ids;
  for (auto& pair : input_) {
    ids.push_back(pair.first);
  }
  std::sort(
      ids.begin(),
      ids.end(),
      [](AbstractOperand*& left, AbstractOperand*& right) {
        return left->id < right->id;
      });
  for (auto i = 0; i < ids.size(); ++i) {
    input_[ids[i]] = i;
  }
  ids.clear();
  for (auto& pair : local_) {
    ids.push_back(pair.first);
  }
  std::sort(
      ids.begin(),
      ids.end(),
      [](AbstractOperand*& left, AbstractOperand*& right) {
        return left->id < right->id;
      });
  for (auto i = 0; i < ids.size(); ++i) {
    local_[ids[i]] = i + input_.size();
  }
  for (auto& [op, id] : literals_) {
    literals_[op] = input_.size() + local_.size() + id;
  }
}

OperandIndex Program::operandIndex(AbstractOperand* op) const {
  auto it = input_.find(op);
  if (it != input_.end()) {
    return it->second;
  }
  it = local_.find(op);
  if (it != local_.end()) {
    return it->second;
  }
  it = literals_.find(op);
  if (it != literals_.end()) {
    return it->second;
  }
  VELOX_FAIL("Operand not found");
}

template <typename T>
int32_t Program::addLiteral(T* value, int32_t count) {
  nextConstant_ = bits::roundUp(nextConstant_, sizeof(T));
  auto start = nextConstant_;
  nextLiteral_ += sizeof(T) * count;
  literalArea_.resize(nextLiteral_);
  memcpy(literalArea_.data(), value, sizeof(T) * count);
  return start;
}

template <TypeKind kind>
int32_t Program::addLiteralTyped(AbstractOperand* op) {
  if (op->literalOffset != AbstractOperand::kNoConstant) {
    return op->constantOffset;
  }
  using T = typename TypeTraits<kind>::NativeType;
  if (op->constant->isNullAt(0)) {
    op->constantNull = true;
    char zero = 0;
    op->constantOffset = addConstant<char>(&zero, 1);
    return;
  }
  T value = op->constant->as<SimpleVector<T>>->valueAt(0);
  if (constexpr(std::is_same_v<T, StringView>)) {
    int64_t inline = 0;
    StringView* value = reinterpret_cast<StringView*>(&value);
    if (stringView->size() <= 6) {
      int64_t inline = static_cast<int64_t>(stringView->size()) << 48;
      memcpy(
          reinterpret_cast<char*>(&inline) + 2,
          stringView->data(),
          stringView->size());
      op->constantOffset = addConstant(&inline, 1);
    } else {
      int64_t zero = 0;
      op->constantOffset = addConstant(&zero);
      addConstant(stringView->data(), stringView->size());
    }
  } else {
    op->constantOffset = addConstant(&T);
  }
}

void Program::markInput(AbstractOperand* op) {
  if (!op) {
    return;
  }
  if (op->constant) {
    VELOX_SCALAR_TYPE_DISPATCH(
        addLiteralTyped, op->constant->type()->kind(), op);
    constant_[op] = constant_.size();
    return;
  }
  if (!local_.count(op)) {
    input_[op] = input_.size();
  }
}

void Program::markResult(AbstractOperand* op) {
  if (!local_.count(op)) {
    local_[op] = local_.size();
  }
}

std::unique_ptr<Executable> Program::getExecutable(
    int32_t maxRows,
    const std::vector<std::unique_ptr<AbstractOperand>>& operands) {
  std::unique_ptr<Executable> exe;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!prepared_.empty()) {
      exe = std::move(prepared_.back());
      prepared_.pop_back();
    }
  }
  if (!exe) {
    exe = std::make_unique<Executable>();
    exe->programShared = shared_from_this();
    exe->program = program_;
    for (auto& pair : input_) {
      exe->inputOperands.add(pair.first->id);
    }
    for (auto& pair : local_) {
      exe->outputOperands.add(pair.first->id);
    }
    exe->output.resize(local_.size());
    exe->literals = &literalOperands_;
    exe->operandsWithIndices = &operandsWithIndices_;
    exe->releaser = [](std::unique_ptr<Executable>& ptr) {
      auto program = ptr->programShared.get();
      ptr->reuse();
      program->releaseExe(std::move(ptr));
    };

  } // We have an exe, whether new or reused. Check the vectors.
  int32_t nth = 0;
  exe->outputOperands.forEach([&](int32_t id) {
    ensureWaveVector(
        exe->output[nth], operands[id]->type, maxRows, true, *arena_);
    ++nth;
  });
  return exe;
}
} // namespace facebook::velox::wav
