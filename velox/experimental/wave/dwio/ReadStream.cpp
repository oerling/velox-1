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

#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/dwio/StructColumnReader.h"

DEFINE_int32(
    wave_reader_rows_per_tb,
    1024,
    "Number of items per thread block in Wave reader");

namespace facebook::velox::wave {

void allOperands(
    const ColumnReader* reader,
    OperandSet& operands,
    std::vector<AbstractOperand*>* abstractOperands) {
  auto op = reader->operand();
  if (op != nullptr) {
    operands.add(op->id);
    if (abstractOperands) {
      abstractOperands->push_back(op);
    }
  }

  for (auto& child : reader->children()) {
    allOperands(child, operands, abstractOperands);
  }
}

ReadStream::ReadStream(
    StructColumnReader* columnReader,
    vector_size_t offset,
    RowSet rows,
    WaveStream& _waveStream,
    const OperandSet* firstColumns)
    : Executable(), offset_(offset), rows_(rows) {
  waveStream = &_waveStream;
  allOperands(columnReader, outputOperands, &abstractOperands_);
  output.resize(outputOperands.size());
  reader_ = columnReader;
  staging_.push_back(std::make_unique<SplitStaging>());
  currentStaging_ = staging_[0].get();
}

  void ReadStream::makeGrid(Stream* stream) {
  auto total = reader_->formatData()->totalRows();
  auto blockSize = FLAGS_wave_reader_rows_per_tb;
  if (total < blockSize) {
    return;
  }
  auto numBlocks = bits::roundUp(total, blockSize) / blockSize;
  auto& children = reader_->children();
  for (auto i = 0; i < children.size(); ++i) {
    auto* child = reader_->children()[i];
    // TODO:  Must  propagate the incoming nulls from outer to inner structs.
    // griddize must decode nulls if present.
    child->formatData()->griddize(
				  blockSize,
        numBlocks,
        deviceStaging_,
        resultStaging_,
				  *currentStaging_,
        programs_,
				  *this);
  }
  if (!programs_.programs.empty()) {
    WaveStats& stats = waveStream->stats();
    stats.bytesToDevice += currentStaging_->bytesToDevice();
          ++stats.numKernels;
          stats.numPrograms += programs_.programs.size();
          stats.numThreads += programs_.programs.size() *
              std::min<int32_t>(rows_.size(), kBlockSize);

    currentStaging_->transfer(*waveStream, *stream);
    WaveBufferPtr extra;
    launchDecode(programs_, &waveStream->arena(), extra, stream);
          staging_.push_back(std::make_unique<SplitStaging>());
          currentStaging_ = staging_.back().get();
  }
}

void ReadStream::makeOps() {
  auto& children = reader_->children();
  for (auto i = 0; i < children.size(); ++i) {
    auto* child = reader_->children()[i];
    if (child->scanSpec().filter()) {
      hasFilters_ = true;
      filters_.emplace_back();
      bool filterOnly = !child->scanSpec().keepValues();
      child->makeOp(
          this,
          filterOnly ? ColumnAction::kFilter : ColumnAction::kValues,
          offset_,
          rows_,
          ops_.back());
    }
  }
  for (auto i = 0; i < children.size(); ++i) {
    auto* child = reader_->children()[i];
    if (child->scanSpec().filter()) {
      continue;
    }
    ops_.emplace_back();
    auto& op = ops_.back();
    child->makeOp(this, ColumnAction::kValues, offset_, rows_, op);
  }
}

  bool ReadStream::decodenonFiltersInFiltersKernel() {
    return ops_.size() == 1;
  }

  
bool ReadStream::makePrograms(bool& needSync) {
  bool allDone = true;
  needSync = false;
  programs_.clear();
  ColumnOp* previousFilter = nullptr;
  if (!filtersDone_ && !filters_.empty()) {
    // Filters are done consecutively, each TB does all the filters for its
    // range.
    for (auto& filter : filters_) {
      filter.reader->formatData()->startOp(
          filter,
          previousFilter,
          deviceStaging_,
          resultStaging_,
          *currentStaging_,
          programs_,
          *this);
      previousFilter = &filter;
    }
    filtersDone_ = true;
    if (!decodenonFiltersInFiltersKernel()) {
      return false;
    }
  }
  for (auto i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.isFinal) {
      continue;
    }
    if (op.prerequisite == ColumnOp::kNoPrerequisite ||
        ops_[op.prerequisite].isFinal) {
      op.reader->formatData()->startOp(
          op,
          previousFilter,
          deviceStaging_,
          resultStaging_,
          *currentStaging_,
          programs_,
          *this);
      if (!op.isFinal) {
        allDone = false;
      }
      if (op.needsResult) {
        needSync = true;
      }
    } else {
      allDone = false;
    }
  }
  if (filters_.empty() && allDone) {
    auto setCount = std::make_unique<GpuDecode>();
    setCount->step = DecodeStep::kRowCountNoFilter;
    setCount->data.rowCountNoFilter.numRows = rows_.size();
    setCount->data.rowCountNoFilter.status =
        control_->deviceData->as<BlockStatus>();
    programs_.programs.emplace_back();
    programs_.programs.back().push_back(std::move(setCount));
  }
  ++nthWave_;
  resultStaging_.setReturnBuffer(waveStream->arena(), programs_);
  return allDone;
}

// static
void ReadStream::launch(std::unique_ptr<ReadStream>&& readStream) {
  using UniqueExe = std::unique_ptr<Executable>;
  // The function of control here is to have a status and row count for each
  // kBlockSize top level rows of output and to have Operand structs for the
  // produced column.
  readStream->makeControl();
  auto numRows = readStream->rows_.size();
  auto waveStream = readStream->waveStream;
  WaveStats& stats = waveStream->stats();
  waveStream->installExecutables(
      folly::Range<UniqueExe*>(reinterpret_cast<UniqueExe*>(&readStream), 1),
      [&](Stream* stream, folly::Range<Executable**> exes) {
        auto* readStream = reinterpret_cast<ReadStream*>(exes[0]);
        bool needSync = false;
	readStream->makeGrid(stream);
	readStream->makeOps();

        for (;;) {
          bool done = readStream->makePrograms(needSync);
          stats.bytesToDevice += readStream->currentStaging_->bytesToDevice();
          ++stats.numKernels;
          stats.numPrograms += readStream->programs_.programs.size();
          stats.numThreads += readStream->programs_.programs.size() *
              std::min<int32_t>(readStream->rows_.size(), kBlockSize);
          readStream->currentStaging_->transfer(*waveStream, *stream);
          if (done) {
            break;
          }
          WaveBufferPtr extra;
          launchDecode(
              readStream->programs(), &waveStream->arena(), extra, stream);
          readStream->staging_.push_back(std::make_unique<SplitStaging>());
          readStream->currentStaging_ = readStream->staging_.back().get();
          if (needSync) {
            waveStream->setState(WaveStream::State::kWait);
            stream->wait();
            readStream->waveStream->setState(WaveStream::State::kHost);
          } else {
            readStream->waveStream->setState(WaveStream::State::kParallel);
          }
        }

        WaveBufferPtr extra;
        launchDecode(
            readStream->programs(),
            &readStream->waveStream->arena(),
            extra,
            stream);
        readStream->waveStream->setState(WaveStream::State::kParallel);
        readStream->waveStream->markLaunch(*stream, *readStream);
      });
}

void ReadStream::makeControl() {
  auto numRows = rows_.size();
  numBlocks_ = bits::roundUp(numRows, kBlockSize) / kBlockSize;
  waveStream->setNumRows(numRows);
  WaveStream::ExeLaunchInfo info;
  waveStream->exeLaunchInfo(*this, numBlocks_, info);
  auto statusBytes = sizeof(BlockStatus) * numBlocks_;
  auto deviceBytes = statusBytes + info.totalBytes;
  auto control = std::make_unique<LaunchControl>(0, numRows);
  control->deviceData = waveStream->arena().allocate<char>(deviceBytes);
  control->status = control->deviceData->as<BlockStatus>();

  operands = waveStream->fillOperands(
      *this, control->deviceData->as<char>() + statusBytes, info)[0];
  control_ = control.get();
  waveStream->addLaunchControl(0, std::move(control));
}

} // namespace facebook::velox::wave
