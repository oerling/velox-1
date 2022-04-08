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

#include "velox/exec/Spiller.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::exec {

int SpillStream::ordinalCounter_;
std::mutex SpillState::mutex_;
uint64_t SpillState::sequence_ = 0;

SpillFile::~SpillFile() {
  if (path_[0] == '/') {
    if (unlink(path_.c_str()) != 0) {
      LOG(ERROR) << "Error deleting spill file " << path_
                 << " errno: " << errno;
    }
  }
}

std::shared_ptr<Config> spillConfig() {
  static auto config = std::make_shared<Config>();
  return config;
}

void FileList::flush(bool close) {
  std::string str;
  {
    if (batch_) {
      std::stringstream outStr;
      OutputStream out(&outStr);
      batch_->flush(&out);
      batch_.reset();
      str = outStr.str();
      if (!isOpen_ || (!close && files_.back()->size() > targetFileSize_)) {
        if (isOpen_) {
          files_.back()->finishWrite();
        }
        files_.push_back(std::make_unique<SpillFile>(
            type_, fmt::format("{}-{}", path_, files_.size()), pool_));
        isOpen_ = true;
      }
      files_.back()->output()->append(str);
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

void SpillState::setNumWays(int32_t numWays) {
  numWays_ = numWays;
  for (auto newWay = files_.size(); newWay < numWays_; ++newWay) {
    files_.push_back(std::make_unique<FileList>(
        type_,
        fmt::format("{}-{}", path_, newWay),
        1 << 20,
        targetSize_,
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
