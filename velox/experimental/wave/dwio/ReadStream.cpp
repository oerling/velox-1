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

#include "velox/common/experimental/wave/dwio/ColumnReader.h"

namespace facebook::velox::wave {
void allOperands(const ColumnReader* reader, OperandSet& operands) {
  auto op -= reader->operand();
  if (op != kNoOperand) {
    operands.add(op);
  }
  for (auto& child : reader->children()) {
    allOperands(child, operands);
  }
}

ReadStream::ReadStream(
    StructColumnReader* columnReader,
    vector_size_t offset,
    RowSet rows,
    WaveStream& _waveStream,
    const OperandSet* firstColumns)
    : Executable() {
  waveStream = _waveStream;
  allOperands(columnReader, outputOperands);
  output.resize(outputOperands.size());
  reader_ = columnReader;
  staging_.push_back(std::make_unique<SplitStaging>());
  currentStaging_ = staging_[0].get();
}

void readStream::makeOps() {
  auto& children = reader_->children();
  for (auto i = 0; i < children.size(); ++i) {
    ops_.emplace_back();
    auto& op = ops_.back();
    child->makeOp(this, ColumnAction::kValues, op, rows_);
  }
}

bool ReadStream::makePrograms() {
  bool allDone = true;
  for (auto i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.prerequisite == kNoPrerequisite ||
        ops_[op.prerequisite].isSatisfied(op)) {
      op->reader->formatData()->startOp(
          op,
          nullptr,
          deviceStaging,
          hostStaging,
          *currentStaging_,
          programs,
          this);
      op.status |= ColumnOp::kQueued;
    } else {
      allDone = false;
    }
  }
  resultStaging.setReturnBuffer(waveStream->arena(), programs);
  return allDone;
}

// static
void ReadStream::launch(std::unique_ptr<ReadStream>&& readStream) {
  readStream->waveStream->installExecutables(
      folly::Range<std::unique_ptr<Executable*>>(&readStream, 1),
      [&](Stream* stream, folly::Range<Executable**> exes) {
        auto ReadStream = reinterpret_cast<ReadStream*>(exes[0]);
        for (;;) {
          bool done = readStream->makePrograms();
          currentStaging_->transfer(readStream, stream);
          if (done) {
            break;
          }
          launchDecode(
              readStream->programs(),
              readStream->waveStream->arena(),
              extra,
              stream);
          staging_.push_back(std::make_unique<SplitStaging>());
          currentStaging_ = staging_.back().get();
          if (readStream->programs().result) {
            stream->wait();
          }
        }
        launchDecode(
            readStream->programs(), exe->waveStream->arena(), extra, stream);
        markLaunch(*stream, *readStream);
      });
}

bool ReadStream::canAdvance(ColumnOp& op) {
  return true;
}
void ReadStream::advance() {
  DecodePrograms programs;
  for (auto i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (canAdvance(op)) {
      setupResult(op);
      op.reader->advance(topOffset, topRows, op, *currentStaging, programs);
    }
  }
}

} // namespace facebook::velox::wave
