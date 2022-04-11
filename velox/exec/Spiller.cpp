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

#include "velox/exec/Spiller.h"

#include "velox/common/base/AsyncSource.h"

#include <folly/ScopeGuard.h>

namespace facebook::velox::exec {

void Spiller::extractSpill(folly::Range<char**> rows, RowVectorPtr* resultPtr) {
  if (!*resultPtr) {
    *resultPtr = std::static_pointer_cast<RowVector>(
        BaseVector::create(rowType_, rows.size(), &pool_));
  } else {
    (*resultPtr)->prepareForReuse();
    (*resultPtr)->resize(rows.size());
  }
  auto result = resultPtr->get();
  auto& types = container_.columnTypes();
  for (auto i = 0; i < types.size(); ++i) {
    container_.extractColumn(rows.data(), rows.size(), i, result->childAt(i));
  }
  auto& aggregates = container_.aggregates();
  auto numKeys = types.size();
  for (auto i = 0; i < aggregates.size(); ++i) {
    aggregates[i]->extractAccumulators(
        rows.data(), rows.size(), &result->childAt(i + numKeys));
  }
}

namespace {
// A stream of ordered rows being read from the in memory
// container. This is the part of a spillable range that is not yet
// spilled when starting to produce output. This is only used for
// sorted spills since for hash join spilling we just use the data in
// the RowContainer as is.
class RowContainerSpillStream : public SpillStream {
 public:
  RowContainerSpillStream(
      RowTypePtr type,
      int32_t numSortingKeys,
      memory::MemoryPool& pool,
      std::vector<char*>&& rows,
      Spiller& spiller)
      : SpillStream(std::move(type), numSortingKeys, pool),
        rows_(std::move(rows)),
        spiller_(spiller) {
    if (!rows_.empty()) {
      nextBatch();
    }
  }

  uint64_t size() const override {
    return 0;
  }

 private:
  void nextBatch() override {
    // Extracts up to 64 rows at a time. Small batch size because may
    // have wide data and no gain in larger.when the caller will go
    // over aggregations row by row.
    static constexpr vector_size_t kMaxRows = 64;
    constexpr uint64_t kMaxBytes = 4 << 20;
    size_t bytes = 0;
    vector_size_t numRows = 0;
    auto limit = std::min<size_t>(rows_.size() - nextBatchIndex_, kMaxRows);
    assert(!rows_.empty());
    for (; numRows < limit; ++numRows) {
      bytes += spiller_.container().rowSize(rows_[nextBatchIndex_ + numRows]);
      if (bytes > kMaxBytes) {
        ++numRows;
        break;
      }
    }
    spiller_.extractSpill(
        folly::Range(&rows_[nextBatchIndex_], numRows), &rowVector_);
    nextBatchIndex_ += numRows;
    size_ = rowVector_->size();
    index_ = 0;
  }

  std::vector<char*> rows_;
  Spiller& spiller_;
  size_t nextBatchIndex_ = 0;
};
} // namespace

std::unique_ptr<SpillStream> Spiller::spillStreamOverRows(int32_t partition) {
  VELOX_CHECK(spillFinalized_);
  VELOX_CHECK_LT(partition, spillRuns_.size());
  ensureSorted(spillRuns_[partition]);
  return std::make_unique<RowContainerSpillStream>(
      rowType_,
      container_.keyTypes().size(),
      pool_,
      std::move(spillRuns_[partition].rows),
      *this);
}

void Spiller::ensureSorted(SpillRun& run) {
  if (!run.sorted) {
    std::sort(
        run.rows.begin(),
        run.rows.end(),
        [&](const char* left, const char* right) {
          return container_.compareRows(left, right) < 0;
        });
    run.sorted = true;
  }
}

std::unique_ptr<Spiller::SpillStatus> Spiller::writeSpill(
    int32_t partition,
    uint64_t maxBytes) {
  // Size Target size of a single vector of spilled content. One of
  // these will be materialized at a time for each stream of the
  // merge.
  constexpr int32_t kTargetBatchBytes = 1 << 20; // 1MB

  RowVectorPtr spillVector;
  auto& run = spillRuns_[partition];
  try {
    ensureSorted(run);
    int64_t totalBytes = 0;
    int32_t written = 0;
    while (written < run.rows.size()) {
      int32_t i = 0;
      int32_t limit = std::min<uint64_t>(128, run.rows.size() - written);
      int32_t bytes = 0;
      for (; i < limit; ++i) {
        bytes += container_.rowSize(run.rows[written + i]);
        if (bytes > kTargetBatchBytes) {
          ++i;
          break;
        }
      }
      folly::Range<char**> spilled(run.rows.data() + written, i);
      extractSpill(spilled, &spillVector);
      state_.appendToPartition(partition, spillVector);
      written += i;
      totalBytes += bytes;
      if (totalBytes > maxBytes) {
        break;
      }
    }
    return std::make_unique<SpillStatus>(partition, written, nullptr);
  } catch (const std::exception& e) {
    return std::make_unique<SpillStatus>(
        partition, 0, std::current_exception());
  }
}

void Spiller::advanceSpill(uint64_t maxBytes) {
  std::vector<std::shared_ptr<AsyncSource<SpillStatus>>> writes;
  for (auto partition = 0; partition < spillRuns_.size(); ++partition) {
    if (pendingSpillPartitions_.find(partition) ==
        pendingSpillPartitions_.end()) {
      continue;
    }
    writes.push_back(std::make_shared<AsyncSource<SpillStatus>>(
        [partition, this, maxBytes]() {
          return writeSpill(
              partition, maxBytes / pendingSpillPartitions_.size());
        }));
    if (executor_) {
      executor_->add([source = writes.back()]() { source->prepare(); });
    }
  }
  auto sync = folly::makeGuard([&]() {
    for (auto& write : writes) {
      try {
        write->move();
      } catch (const std::exception& e) {
      }
    }
  });

  for (auto& write : writes) {
    auto result = write->move();

    if (result->error) {
      std::rethrow_exception(result->error);
    }
    auto numWritten = result->numWritten;
    auto partition = result->partition;
    auto& run = spillRuns_[partition];
    auto spilled = folly::Range<char**>(run.rows.data(), numWritten);
    eraser_(spilled);
    if (!container_.numRows()) {
      // If the container became empty, free its memory.
      container_.clear();
    }
    run.rows.erase(run.rows.begin(), run.rows.begin() + numWritten);
    if (run.rows.empty()) {
      // Run ends, start with a new file next time.
      run.clear();
      state_.finishWrite(partition);
      pendingSpillPartitions_.erase(partition);
    }
  }
}

void Spiller::spill(
    uint64_t targetRows,
    uint64_t targetSpace,
    RowContainerIterator& iterator) {
  bool doneFullSweep = false;
  bool startedFullSweep = false;
  VELOX_CHECK(!spillFinalized_);
  if (!state_.numPartitions()) {
    state_.setNumPartitions(1);
  }
  for (;;) {
    auto rowsLeft = container_.numRows();
    auto spaceLeft = container_.stringAllocator().retainedSize() -
        container_.stringAllocator().freeSpace();
    if (!rowsLeft || (rowsLeft <= targetRows && spaceLeft < targetSpace)) {
      return;
    }
    if (!pendingSpillPartitions_.empty()) {
      advanceSpill(std::numeric_limits<uint64_t>::max());
      if (!pendingSpillPartitions_.empty()) {
        continue;
      }
    }
    if (doneFullSweep) {
      return;
    }
    for (auto newPartition = spillRuns_.size();
         newPartition < state_.maxPartitions();
         ++newPartition) {
      spillRuns_.emplace_back();
    }
    clearSpillRuns();
    iterator.reset();
    if (fillSpillRuns(
            iterator,
            targetSpace < state_.targetFileSize() ? RowContainer::kUnlimited
                                                  : state_.targetFileSize())) {
      // Arrived at end of the container. Add more spilled ranges if any left.
      if (state_.numPartitions() < state_.maxPartitions()) {
        state_.setNumPartitions(state_.numPartitions() + 1);
      } else {
        doneFullSweep = startedFullSweep;
        startedFullSweep = true;
      }
      iterator.reset();
    }
  }
}

std::vector<char*> Spiller::finishSpill() {
  VELOX_CHECK(!spillFinalized_);
  spillFinalized_ = true;
  clearSpillRuns();
  RowContainerIterator iterator;
  iterator.reset();
  std::vector<char*> rowsFromNonSpillingPartitions;
  fillSpillRuns(
      iterator, RowContainer::kUnlimited, &rowsFromNonSpillingPartitions);
  return rowsFromNonSpillingPartitions;
}

void Spiller::clearSpillRuns() {
  for (auto& run : spillRuns_) {
    run.clear();
  }
}

bool Spiller::fillSpillRuns(
    RowContainerIterator& iterator,
    uint64_t targetSize,
    std::vector<char*>* FOLLY_NULLABLE rowsFromNonSpillingPartitions) {
  // Number of rows to hash and divide into spill partitions at a time.
  constexpr int32_t kHashBatchSize = 1024;
  bool final = false;
  if (rowsFromNonSpillingPartitions) {
    final = true;
    VELOX_CHECK_EQ(
        targetSize,
        RowContainer::kUnlimited,
        "Retrieving rows of non-spilling partitions is only "
        "allowed if retrieving the whole container");
    final = true;
  } else if (targetSize == RowContainer::kUnlimited) {
    final = true;
  }
  std::vector<uint64_t> hashes(kHashBatchSize);
  std::vector<char*> rows(kHashBatchSize);
  for (;;) {
    auto numRows = container_.listRows(
        &iterator, rows.size(), RowContainer::kUnlimited, rows.data());

    // Calculate hashes for this batch of spill candidates.
    auto rowSet = folly::Range<char**>(rows.data(), numRows);
    for (auto i = 0; i < container_.keyTypes().size(); ++i) {
      container_.hash(i, rowSet, i > 0, hashes.data());
    }

    // Put each in its run.
    for (auto i = 0; i < numRows; ++i) {
      auto partition = bits_.partition(hashes[i], spillRuns_.size());
      if (partition == -1) {
        if (rowsFromNonSpillingPartitions) {
          rowsFromNonSpillingPartitions->push_back(rows[i]);
        }
        continue;
      }
      spillRuns_[partition].rows.push_back(rows[i]);
      spillRuns_[partition].size += container_.rowSize(rows[i]);
    }
    // The final phase goes through the whole container and makes runs for all
    // non-empty spilling partitions.
    if (final && numRows) {
      continue;
    }
    bool anyStarted = false;
    for (auto i = 0; i < spillRuns_.size(); ++i) {
      auto& run = spillRuns_[i];
      if (!run.rows.empty() && (run.size > targetSize || final)) {
        pendingSpillPartitions_.insert(i);
        anyStarted = true;
      }
    }
    if (final) {
      return true;
    }
    if (!numRows) {
      clearNonSpillingRuns();
      return true;
    }
    if (anyStarted) {
      clearNonSpillingRuns();
      return false;
    }
  }
}

void Spiller::clearNonSpillingRuns() {
  for (auto i = 0; i < spillRuns_.size(); ++i) {
    if (pendingSpillPartitions_.find(i) == pendingSpillPartitions_.end()) {
      spillRuns_[i].clear();
    }
  }
}

} // namespace facebook::velox::exec
