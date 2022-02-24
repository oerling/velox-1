/*
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

enum class SpillKind { kGroupBy, kHashBuild, kHashProbe };

class SpillInput : public ByteStream {
 public:
  SpillInput(std::unique_ptr<ReadFile>&& input, BufferPtr buffer)
      : input_(std::move(input)), buffer_(std::move(buffer)) {
    size_ = input_->size();
    next(true);
  }

  void next(bool throwIfPastEnd) override {
    int32_t readBytes = std::min(input_->size() - offset_, buffer_->capacity());
    setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
    input_->pread(offset_, readBytes, buffer_->asMutable<char>());
    offset_ += readBytes;
  }

  bool atEnd() const {
    return offset_ == size_ && ranges()[0].position >= ranges()[0].size;
  }

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
  int ordinal_;
  static int ordinalCounter_;
};

class SpillFile : public SpillStream {
 public:
  SpillFile(TypePtr type, const std::string& path, memory::MemoryPool& pool)
      : SpillStream(type, pool), path_(path) {}

  ~SpillFile();

  WriteFile* output() {
    if (!output_) {
      auto fs = getFileSystem(path_, Config());
      output_ = fs->openFileForWrite(path_);
    }
    return output_.get();
  }

  void finishWrite() {
    VELOX_CHECK(output_);
    size_ = output_->size();
    output_ = nullptr;
  }

  void startRead() {
    constexpr uint64_t kMaxReadBufferSize = 1 << 20; // 1MB
    VELOX_CHECK(!output_);
    auto file = generateReadFile(path_);
    auto buffer = AlignedBuffer::allocate<char>(
        std::min<uint64_t>(size_, kMaxReadBufferSize), &pool_);
    stream_ = std::make_unique<SpillInput>(std::move(file), std::move(buffer));
    next();
    index_ = 0;
  }

  uint64_t size() const override {
    if (output_) {
      return output_->size();
    }
    return size_;
  }

  bool atEnd() const override {
    return index_ >= numRows_ && stream_->atEnd();
  }

  void read(RowVector& result);

 protected:
  void nextBatch() override {
    VectorStreamGroup::read(stream_.get(), &pool_, type_, &rowVector_);
    if (rowVector_->size() &&
        rowVector_->childAt(0)->as<FlatVector<int32_t>>()->valueAt(0) ==
            199419) {
      LOG(INFO) << "Read 199419 as first";
    }
    index_ = 0;
    numRows_ = rowVector_->size();
  }

 private:
  std::string path_;
  uint64_t size_ = 0;
  std::unique_ptr<WriteFile> output_;
  std::unique_ptr<SpillInput> stream_;
};

struct HashBitField {
  // Low bit number of hash number bit field
  uint8_t begin;
  // Bit number of first bit above the hash number bit field.
  uint8_t end;
};

// Sequence of files for one range of hash numbers
class FileList {
 public:
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
  void write(RowVectorPtr rows, const folly::Range<IndexRange*> indices);
  std::vector<std::unique_ptr<SpillFile>> files() {
    return std::move(files_);
  }

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

// Set of files keyed by hash number range.
class SpillState {
 public:
  SpillState(
      TypePtr type,
      const std::string& path,
      HashBitField bits,
      uint64_t targetSize,
      uint64_t targetBatchSize,
      memory::MemoryPool& pool,
      memory::MappedMemory* mappedMemory)
      : type_(type),
        path_(path),
        bits_(bits),
        fieldMask_(((1UL << bits_.end - bits_.begin)) - 1),
        targetSize_(targetSize),
        targetBatchSize_(targetBatchSize),
        pool_(pool),
        mappedMemory_(mappedMemory) {}

  int32_t way(uint64_t hash) {
    auto field = (hash >> bits_.begin) & fieldMask_;
    return field < numWays_ ? field : -1;
  }

  int32_t numWays() const {
    return numWays_;
  }

  void setNumWays(int32_t numWays);

  uint16_t maxWays() {
    return 1 << bits_.end - bits_.begin;
  }

  uint64_t targetSize() const {
    return targetSize_;
  }

  uint64_t targetBatchSize() const {
    return targetBatchSize_;
  }

  memory::MemoryPool& pool() {
    return pool_;
  }

  void write(
      uint16_t way,
      RowVectorPtr rows,
      const folly::Range<IndexRange*> indices);

  void finishWrite(uint16_t way) {
    files_[way]->flush(true);
  }

  void finishWrite() {
    for (auto& file : files_) {
      file->flush(true);
    }
  }

  void merge(std::vector<std::unique_ptr<SpillState>>&& others);

  static uint64_t spillFileId() {
    std::lock_guard<std::mutex> l(mutex_);
    return ++sequence_;
  }

  std::unique_ptr<TreeOfLosers<VectorRow, SpillStream>> startMerge(
      uint16_t way,
      std::unique_ptr<SpillStream>&& extra);

 private:
  const TypePtr type_;
  const std::string path_;
  const HashBitField bits_;
  const uint64_t fieldMask_;
  // Number of spilled ranges. All hashes where the 'bits_' <
  // numSpilledRanges spill to the file list in the corresponding
  // place in 'files_'.
  int32_t numWays_ = 0;
  const uint64_t targetSize_;
  const uint64_t targetBatchSize_;
  // A file list for each spilled range.
  std::vector<std::unique_ptr<FileList>> files_;
  memory::MemoryPool& pool_;
  memory::MappedMemory* const mappedMemory_;
  static uint64_t sequence_;
  static std::mutex mutex_;
};

} // namespace facebook::velox::exec
