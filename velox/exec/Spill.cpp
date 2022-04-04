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

#include "velox/exec/Spill.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::exec {

std::atomic<int32_t> SpillStream::ordinalCounter_;

  void SpillInput::next(bool /*throwIfPastEnd*/) {
  int32_t readBytes = std::min(input_->size() - offset_, buffer_->capacity());
  setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
  input_->pread(offset_, readBytes, buffer_->asMutable<char>());
  offset_ += readBytes;
}

void SpillInput::seekp(std::streampos position) {
  auto bufferOffset = offset_ - current_->size;
  if (bufferOffset <= position && bufferOffset + current_->size < position) {
    current_->position = std::streamoff(position) - bufferOffset;
  } else {
    // The seek target is not in the buffer.
    offset_ = position;
    current_->position = 0;
    current_->size = 0;
    next(true);
  }
}

void SpillStream::pop() {
  if (++index_ >= size_) {
    nextBatch();
  }
}

SpillFile::~SpillFile() {
  // TBD: FileSystem must have a file deletion method.
  VELOX_CHECK_EQ(
      path_[0],
      '/',
      "Spill only supports absolute paths to local files, not {}.",
      path_);
  if (unlink(path_.c_str()) != 0) {
    LOG(ERROR) << "Error deleting spill file " << path_ << " errno: " << errno;
  }
}

WriteFile& SpillFile::output() {
  if (!output_) {
    auto fs = filesystems::getFileSystem(path_, nullptr);
    output_ = fs->openFileForWrite(path_);
  }
  return *output_;
}

void SpillFile::startRead() {
  constexpr uint64_t kMaxReadBufferSize = 1 << 20; // 1MB
  VELOX_CHECK(!output_);
  VELOX_CHECK(!input_);
  auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(path_);
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(fileSize_, kMaxReadBufferSize), &pool_);
  input_ = std::make_unique<SpillInput>(std::move(file), std::move(buffer));
  nextBatch();
}

void SpillFile::nextBatch() {
  index_ = 0;
  if (input_->atEnd()) {
    size_ = 0;
    return;
  }
  VectorStreamGroup::read(input_.get(), &pool_, type_, &rowVector_);
  size_ = rowVector_->size();
}

WriteFile& SpillFileList::currentOutput() {
  if (files_.empty() || !files_.back()->isWritable() ||
      files_.back()->size() > targetFileSize_ * 1.5) {
    if (!files_.empty() && files_.back()->isWritable()) {
      files_.back()->finishWrite();
    }
    files_.push_back(std::make_unique<SpillFile>(
        type_,
        numSortingKeys_,
        fmt::format("{}-{}", path_, files_.size()),
        pool_));
  }
  return files_.back()->output();
}

void SpillFileList::flush() {
  if (batch_) {
    std::stringstream stringStream;
    OStreamOutputStream out(&stringStream);
    batch_->flush(&out);
    batch_.reset();
    std::string str = stringStream.str();
    if (!str.empty()) {
      currentOutput().append(str);
    }
  }
}

void SpillFileList::write(
    const RowVectorPtr& rows,
    const folly::Range<IndexRange*>& indices) {
  if (!batch_) {
    batch_ = std::make_unique<VectorStreamGroup>(&mappedMemory_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rows->type()), 1000);
  }
  batch_->append(rows, indices);

  flush();
}

void SpillFileList::finishFile() {
  flush();
  if (files_.empty()) {
    return;
  }
  if (files_.back()->isWritable()) {
    files_.back()->finishWrite();
  }
}

void SpillState::setNumPartitions(int32_t numPartitions) {
  VELOX_CHECK_LE(numPartitions, maxPartitions());
  VELOX_CHECK_GT(numPartitions, numPartitions_, "May only add partitions");
  numPartitions_ = numPartitions;
}

void SpillState::appendToPartition(
    int32_t partition,
    const RowVectorPtr& rows) {
  // Ensure that all partitions exist before writing.
  for (auto newPartition = files_.size(); newPartition < numPartitions_;
       ++newPartition) {
    files_.push_back(std::make_unique<SpillFileList>(
        std::static_pointer_cast<const RowType>(rows->type()),
        numSortingKeys_,
        fmt::format("{}-{}", path_, newPartition),
        1 << 20,
        targetFileSize_,
        pool_,
        mappedMemory_));
  }

  IndexRange range{0, rows->size()};
  files_[partition]->write(rows, folly::Range<IndexRange*>(&range, 1));
}

std::unique_ptr<TreeOfLosers<SpillStream>> SpillState::startMerge(
    int32_t partition,
    std::unique_ptr<SpillStream>&& extra) {
  VELOX_CHECK_LT(partition, files_.size());
  auto list = std::move(files_[partition]);
  auto files = list->files();
  std::vector<std::unique_ptr<SpillStream>> result;
  for (auto& file : files) {
    file->startRead();
    result.push_back(std::move(file));
  }
  if (extra) {
    result.push_back(std::move(extra));
  }
  return std::make_unique<TreeOfLosers<SpillStream>>(std::move(result));
}

} // namespace facebook::velox::exec
