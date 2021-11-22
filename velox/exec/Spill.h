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

#include "velox/common/file/File.h"
#include "velox/exec/TreeOfLosers.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

// A lightweight, non-owning reference to a row in a RowVector read from
// spill.
struct VectorRow {
  std::vector<std::unique_ptr<DecodedVector>>* decoded;
  RowVector* rowVector;
  vector_size_t index;
};

// Input stream backed by spill file.
class SpillInput : public ByteStream {
 public:
  // Reads from 'input' using 'buffer' for buffering reads.
  SpillInput(std::unique_ptr<ReadFile>&& input, BufferPtr buffer)
      : input_(std::move(input)), buffer_(std::move(buffer)) {
    size_ = input_->size();
    next(true);
  }

  void next(bool throwIfPastEnd) override;

  // True if of the file has been read into vectors.
  bool atEnd() const {
    return offset_ == size_ && ranges()[0].position >= ranges()[0].size;
  }

  // Returns the offset of the first byte of input that is not yet consumed. May
  // be used with seekp to rewind a read.
  Position tellp() const override {
    return std::make_tuple(
        nullptr, offset_ - current_->size + current_->position);
  }

  // Positions the file so tht the byte at 'position' is the first to be read.
  void seekp(Position position) override;

 private:
  std::unique_ptr<ReadFile> input_;
  uint64_t size_;
  BufferPtr buffer_;
  uint64_t offset_ = 0;
};

// A source of VectorRows coming either from a file or memory.
class SpillStream {
 public:
  SpillStream(TypePtr type, memory::MemoryPool& pool)
      : type_(std::dynamic_pointer_cast<const RowType>(type)), pool_(pool) {
    ordinal_ = ++ordinalCounter_;
  }

  virtual ~SpillStream() = default;

  virtual bool atEnd() const = 0;

  virtual std::string label() const {
    return fmt::format("{}", ordinal_);
  }

  VectorRow next() {
    if (index_ == numRows_) {
      nextBatch();
      index_ = 0;
      numRows_ = rowVector_->size();
      SelectivityVector allRows(numRows_);
      auto width = rowVector_->childrenSize();
      if (decoded_.size() < width) {
        decoded_.resize(width);
      }
      for (auto i = 0; i < width; ++i) {
        if (!decoded_[i]) {
          decoded_[i] = std::make_unique<DecodedVector>();
        }
        decoded_[i]->decode(*rowVector_->childAt(i), allRows);
      }
    }
    return {&decoded_, rowVector_.get(), index_++};
  }

  virtual uint64_t size() const {
    return 0;
  }

 protected:
  virtual void nextBatch() = 0;

  std::shared_ptr<const RowType> type_;
  memory::MemoryPool& pool_;
  RowVectorPtr rowVector_;
  vector_size_t index_ = 0;
  vector_size_t numRows_ = 0;
  std::vector<std::unique_ptr<DecodedVector>> decoded_;
  int32_t ordinal_;
  static std::atomic<int32_t> ordinalCounter_;
};

// Represents a spill file that is first in write mode and then
// turns into an source of VectorRow. Owns a file system file that
// contains the spilled data and is live for the duration of 'this'.
class SpillFile : public SpillStream {
 public:
  SpillFile(TypePtr type, const std::string& path, memory::MemoryPool& pool)
      : SpillStream(type, pool), path_(path) {}

  ~SpillFile();

  // Returns a file for writing spilled data.
  WriteFile& output();

  // Finishes writing and flushes any unwritten data.
  void finishWrite() {
    VELOX_CHECK(output_);
    size_ = output_->size();
    output_ = nullptr;
  }

  // Prepares 'this' for reading. Positions the read at the first row of
  // content.
  void startRead();

  // Returns the file size in bytes.
  uint64_t size() const override {
    if (output_) {
      return output_->size();
    }
    return size_;
  }

  // True if no more VectorRows can be read from 'this'.
  bool atEnd() const override {
    return index_ >= numRows_ && stream_->atEnd();
  }

  // Sets 'result' to refer to the next row of content of 'this'.
  void read(RowVector& result);

 protected:
  void nextBatch() override {
    VectorStreamGroup::read(stream_.get(), &pool_, type_, &rowVector_);
    index_ = 0;
    numRows_ = rowVector_->size();
  }

 private:
  std::string path_;
  uint64_t size_ = 0;
  std::unique_ptr<WriteFile> output_;
  std::unique_ptr<SpillInput> stream_;
};

// Describes a bit field inside a 64 bit hash number for use in
// partitioning data over multiple sets of spill files.
struct HashBitField {
  // Low bit number of hash number bit field
  uint8_t begin;
  // Bit number of first bit above the hash number bit field.
  uint8_t end;
};

// Sequence of files for one partition of the spilled data. If data is
// sorted, each file is sorted. The globally sorted order is produced
// by merging the constituent files.
class FileList {
 public:
  // Constructs a set of spill files. 'type' a RowType describing the content.
  // 'path' is a file path prefix.  FileList( 'targetBatchSize is the target
  // size of a single RowVector in rows. 'targetFileSize' is the target byte
  // size of a single file in the file set. 'pool' and 'mappedMemory' are used
  // for buffering and constructing the result data read from 'this'.
  FileList(
      TypePtr type,
      const std::string path,
      uint64_t targetBatchSize,
      uint64_t targetFileSize,
      memory::MemoryPool& pool,
      memory::MappedMemory* mappedMemory)
      : type_(type),
        path_(path),
        targetBatchSize_(targetBatchSize),
        targetFileSize_(targetFileSize),
        pool_(pool),
        mappedMemory_(mappedMemory) {}

  // Adds 'rows' for the positions in 'indices' into 'this'. The indices
  // must produce a view where the rows are sorted if sorting is desired.
  // Consecutive calls must have sorted data so that the first row of the
  // next call is not less than the last row of the previous call.
  void write(RowVectorPtr rows, const folly::Range<IndexRange*> indices);
  std::vector<std::unique_ptr<SpillFile>> files() {
    return std::move(files_);
  }

  // Closes one file of output. Subsequent calls to 'write' start a
  // different sorted run and must be ordered between themselves but not
  // with respect to calls before flush(). If 'close' is true, write must
  // not be called and the file set can only be read.
  void flush(bool close);

 private:
  const TypePtr type_;
  const std::string path_;
  const uint64_t targetBatchSize_;
  const uint64_t targetFileSize_;
  memory::MemoryPool& pool_;
  memory::MappedMemory* const mappedMemory_;
  std::unique_ptr<VectorStreamGroup> batch_;
  bool isOpen_ = false;
  std::vector<std::unique_ptr<SpillFile>> files_;
};

// Represents all spilled data of an operator, e.g. order by or group
// by. This has one FileList per partition of spill data.
class SpillState {
 public:
  // Constructs a SpillState. 'type' is the content RowType. 'path' is the file
  // system path prefix. 'bits' is the hash bit field for partitioning data
  // between files. This also gives the maximum number of partitions.
  // 'targetFileSize' is the target size of a single file. 'targetBatchSize is
  // the target number of rows in a single RowVector written to a spill file.
  // 'pool' and 'mappedMemory' own the memory for state and results.
  SpillState(
      TypePtr type,
      const std::string& path,
      HashBitField bits,
      uint64_t targetFileSize,
      uint64_t targetBatchSize,
      memory::MemoryPool& pool,
      memory::MappedMemory* mappedMemory)
      : type_(type),
        path_(path),
        bits_(bits),
        fieldMask_(((1UL << (bits_.end - bits_.begin))) - 1),
        targetFileSize_(targetFileSize),
        targetBatchSize_(targetBatchSize),
        pool_(pool),
        mappedMemory_(mappedMemory) {}

  // Returns which spill partition 'hash' falls. Returns -1 if the
  // partition of 'hash' is has not been started.
  int32_t way(uint64_t hash) {
    auto field = (hash >> bits_.begin) & fieldMask_;
    return field < numWays_ ? field : -1;
  }

  int32_t numWays() const {
    return numWays_;
  }

  // Sets how many of the spill partitions are in use.
  void setNumWays(int32_t numWays);

  // Returns how many ways spilled data can be partitioned.
  uint16_t maxWays() {
    return 1 << (bits_.end - bits_.begin);
  }

  uint64_t targetFileSize() const {
    return targetFileSize_;
  }

  uint64_t targetBatchSize() const {
    return targetBatchSize_;
  }

  memory::MemoryPool& pool() {
    return pool_;
  }

  // Appends data to 'way'. The rows  given by 'indices'  must be sorted and
  // must hash to 'way'.
  void write(
      uint16_t way,
      RowVectorPtr rows,
      const folly::Range<IndexRange*> indices);

  // Finishes a sorted run for 'way'. If write is called for 'way'
  // again, the data does not have to be sorted relative to the data
  // written so far.
  void finishWrite(uint16_t way) {
    files_[way]->flush(true);
  }

  // Starts reading values for 'way'. If 'extra' is non-null, it can be
  // a stream of rows from a RowContainer so as to merge unspilled
  // data with spilled data.
  std::unique_ptr<TreeOfLosers<VectorRow, SpillStream>> startMerge(
      uint16_t way,
      std::unique_ptr<SpillStream>&& extra);

  // Helper function for comparing current elements of streams to merge.
  static int32_t compareSpilled(
      const VectorRow& left,
      const VectorRow& right,
      int32_t numKeys);

 private:
  const TypePtr type_;
  const std::string path_;
  const HashBitField bits_;
  const uint64_t fieldMask_;
  // Number of spilled ranges. All hashes where the 'bits_' <
  // numSpilledRanges spill to the file list in the corresponding
  // place in 'files_'.
  int32_t numWays_ = 0;
  const uint64_t targetFileSize_;
  const uint64_t targetBatchSize_;
  // A file list for each spilled range.
  std::vector<std::unique_ptr<FileList>> files_;
  memory::MemoryPool& pool_;
  memory::MappedMemory* const mappedMemory_;
  static uint64_t sequence_;
  static std::mutex mutex_;
};

} // namespace facebook::velox::exec
