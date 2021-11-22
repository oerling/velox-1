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

#include "velox/exec/Spill.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::exec {

int SpillStream::ordinalCounter_;
std::mutex SpillState::mutex_;
uint64_t SpillState::sequence_ = 0;

void SpillInput::next(bool throwIfPastEnd) {
    int32_t readBytes = std::min(input_->size() - offset_, buffer_->capacity());
    setRange({buffer_->asMutable<uint8_t>(), readBytes, 0});
    input_->pread(offset_, readBytes, buffer_->asMutable<char>());
    offset_ += readBytes;
  }


void SpillInput::seekp(Position position) {
  auto target = std::get<1>(position);
  auto bufferOffset = offset_ - current_->size;
  if (bufferOffset <= target && bufferOffset + current_->size < target) {
    current_->position = target - bufferOffset;
  } else {
    // The seek target is not in the buffer.
    offset_ = target;
    current_->position = 0;
    current_->size = 0;
    next(true);
  }
}
  

SpillFile::~SpillFile() {
  if (path_[0] == '/') {
    if (unlink(path_.c_str()) != 0) {
      LOG(ERROR) << "Error deleting spill file " << path_
                 << " errno: " << errno;
    }
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
  auto fs = filesystems::getFileSystem(path_, nullptr);
  auto file = fs->openFileForRead(path_);
  auto buffer = AlignedBuffer::allocate<char>(
      std::min<uint64_t>(size_, kMaxReadBufferSize), &pool_);
  stream_ = std::make_unique<SpillInput>(std::move(file), std::move(buffer));
  next();
  index_ = 0;
}

void FileList::flush(bool close) {
  std::string str;
  {
    if (batch_) {
      std::stringstream out;
      batch_->flush(&out);
      batch_.reset();
      str = out.str();
      if (!isOpen_ || (!close && files_.back()->size() > targetFileSize_)) {
        if (isOpen_) {
          files_.back()->finishWrite();
        }
        files_.push_back(std::make_unique<SpillFile>(
            type_, fmt::format("{}-{}", path_, files_.size()), pool_));
        isOpen_ = true;
      }
      files_.back()->output().append(str);
    }
  }
  if (close && isOpen_) {
    files_.back()->finishWrite();
    isOpen_ = false;
  }
}

void FileList::write(
    RowVectorPtr rows,
    const folly::Range<IndexRange*> indices) {
  if (!batch_) {
    batch_ = std::make_unique<VectorStreamGroup>(mappedMemory_);
    batch_->createStreamTree(
        std::static_pointer_cast<const RowType>(rows->type()), 1000);
  }
  batch_->append(rows, indices);
  flush(false);
}

// static
int32_t SpillState::compareSpilled(
    const VectorRow& left,
    const VectorRow& right,
    int32_t numKeys) {
  for (auto i = 0; i < numKeys; ++i) {
    auto leftDecoded = (*left.decoded)[i].get();
    auto rightDecoded = (*right.decoded)[i].get();
    auto result = leftDecoded->base()->compare(
        rightDecoded->base(),
        leftDecoded->index(left.index),
        rightDecoded->index(right.index));
    if (result) {
      return result;
    }
  }
  return 0;
}

void SpillState::setNumWays(int32_t numWays) {
  numWays_ = numWays;
  for (auto newWay = files_.size(); newWay < numWays_; ++newWay) {
    files_.push_back(std::make_unique<FileList>(
        type_,
        fmt::format("{}-{}", path_, newWay),
        1 << 20,
        targetFileSize_,
        pool_,
        mappedMemory_));
  }
}

void SpillState::write(
    uint16_t way,
    RowVectorPtr rows,
    const folly::Range<IndexRange*> indices) {
  files_[way]->write(rows, indices);
}

std::unique_ptr<TreeOfLosers<VectorRow, SpillStream>> SpillState::startMerge(
    uint16_t way,
    std::unique_ptr<SpillStream>&& extra) {
  VELOX_CHECK(way < files_.size());
  auto list = std::move(files_[way]);
  list->flush(true);
  auto files = list->files();
  std::vector<std::unique_ptr<SpillStream>> result;
  for (auto& file : files) {
    file->startRead();
    result.push_back(std::move(file));
  }
  if (extra) {
    result.push_back(std::move(extra));
  }
  return std::make_unique<TreeOfLosers<VectorRow, SpillStream>>(
      std::move(result));
}

} // namespace facebook::velox::exec
