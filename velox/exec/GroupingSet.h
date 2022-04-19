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

#include "velox/exec/AggregationMasks.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Spiller.h"
#include "velox/exec/TreeOfLosers.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {

class Aggregate;

class GroupingSet {
 public:
  GroupingSet(
      std::vector<std::unique_ptr<VectorHasher>>&& hashers,
      std::vector<ChannelIndex>&& preGroupedKeys,
      std::vector<std::unique_ptr<Aggregate>>&& aggregates,
      std::vector<std::optional<ChannelIndex>>&& aggrMaskChannels,
      std::vector<std::vector<ChannelIndex>>&& channelLists,
      std::vector<std::vector<VectorPtr>>&& constantLists,
      std::vector<TypePtr>&& intermediateTypes,
      bool ignoreNullKeys,
      bool isPartial,
      bool isRawInput,
      OperatorCtx* operatorCtx);

  void addInput(const RowVectorPtr& input, bool mayPushdown);

  void noMoreInput();

  /// Typically, the output is not available until all input has been added.
  /// However, in case when input is clustered on some of the grouping keys, the
  /// output becomes available every time one of these grouping keys changes
  /// value. This method returns true if no-more-input message has been received
  /// or if some groups are ready for output because pre-grouped keys values
  /// have changed.
  bool hasOutput();

  /// Called if partial aggregation has reached memory limit or if hasOutput()
  /// returns true.
  bool getOutput(
      int32_t batchSize,
      RowContainerIterator* iterator,
      RowVectorPtr& result);

  uint64_t allocatedBytes() const;

  void resetPartial();

  const HashLookup& hashLookup() const;

  // Spills content until under 'targetRows' and under 'targetBytes'
  // of out of line data are left. If targetRows is 0, spills
  // everything and physically frees the data in the
  // 'table_->rows()'. This leaves 'table_' initialized and 'this'
  // ready to accumulate more input. This is called by ensureInputFits
  // or by external memory management. In the latter case, the Driver
  // of this will be in a paused state and off thread.
  void spill(int64_t targetRows, int64_t targetBytes);

  int64_t spilledBytes() const {
    return spiller_ ? spiller_->spilledBytes() : 0;
  }
  
 private:
  void addInputForActiveRows(const RowVectorPtr& input, bool mayPushdown);

  void addRemainingInput();

  void initializeGlobalAggregation();

  void addGlobalAggregationInput(const RowVectorPtr& input, bool mayPushdown);

  bool getGlobalAggregationOutput(
      int32_t batchSize,
      bool isPartial,
      RowContainerIterator* iterator,
      RowVectorPtr& result);

  void createHashTable();

  void populateTempVectors(int32_t aggregateIndex, const RowVectorPtr& input);

  // If the given aggregation has mask, the method returns reference to the
  // selectivity vector from the maskedActiveRows_ (based on the mask channel
  // index for this aggregation), otherwise it returns reference to activeRows_.
  const SelectivityVector& getSelectivityVector(size_t aggregateIndex) const;

  // Checks if input will fit in the existing memory and increases
  // reservation if not. If reservation cannot be increased, spills
  // enough to make 'input' fit.
  void ensureInputFits(const RowVectorPtr& input);

  void
  extractGroups(char** groups, int32_t numGroups, const RowVectorPtr& result);

  bool getOutputWithSpill(const RowVectorPtr& result);

  bool mergeNext(const RowVectorPtr& result);

  void initializeRow(SpillStream& keys, char* row);
  void updateRow(SpillStream& keys, char* row);
  void extractSpillResult(const RowVectorPtr& result);

  std::vector<ChannelIndex> keyChannels_;

  /// A subset of grouping keys on which the input is clustered.
  const std::vector<ChannelIndex> preGroupedKeyChannels_;

  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  const bool isGlobal_;
  const bool isPartial_;
  const bool isRawInput_;
  std::vector<std::unique_ptr<Aggregate>> aggregates_;
  AggregationMasks masks_;
  // Argument list for the corresponding element of 'aggregates_'.
  const std::vector<std::vector<ChannelIndex>> channelLists_;
  // Constant arguments to aggregates. Corresponds pairwise to
  // 'channelLists_'. This is used when channelLists_[i][j] ==
  // kConstantChannel.
  const std::vector<std::vector<VectorPtr>> constantLists_;
  const bool ignoreNullKeys_;
  memory::MappedMemory* const mappedMemory_;

  // Boolean indicating whether accumulators for a global aggregation (i.e.
  // aggregation with no grouping keys) have been initialized.
  bool globalAggregationInitialized_{false};

  std::vector<bool> mayPushdown_;

  // Place for the arguments of the aggregate being updated.
  std::vector<VectorPtr> tempVectors_;
  std::unique_ptr<BaseHashTable> table_;
  std::unique_ptr<HashLookup> lookup_;
  SelectivityVector activeRows_;

  // Used to allocate memory for a single row accumulating results of global
  // aggregation
  HashStringAllocator stringAllocator_;
  AllocationPool rows_;
  const bool isAdaptive_;

  core::ExecCtx& execCtx_;

  bool noMoreInput_{false};

  /// In case of partial streaming aggregation, the input vector passed to
  /// addInput(). A set of rows that belong to the last group of pre-grouped
  /// keys need to be processed after flushing the hash table and accumulators.
  RowVectorPtr remainingInput_;

  /// First row in remainingInput_ that needs to be processed.
  vector_size_t firstRemainingRow_;

  /// The value of mayPushdown flag specified in addInput() for the
  /// 'remainingInput_'.
  bool remainingMayPushdown_;

  uint64_t spillThreshold_ = 0;
  uint64_t maxBatchBytes_;

  // Intermediate types of aggregates. Used for spilling

  // Sum of the minimum variable length size for variable length
  // accumulators. This + serialized size of new values is a reasonable
  // cap for additional space usage when updating.
  int32_t minVariableWidthAccumulatorBytes_{0};
  std::vector<TypePtr> intermediateTypes_;

  // Filesystem path for spill files, empty if spilling is disabled.
  const std::string spillPath_;

  std::unique_ptr<Spiller> spiller_;
  std::unique_ptr<TreeOfLosers<SpillStream>> merge_;
  RowContainerIterator spillIterator_;

  // Container for materializing batches of output from spilling.
  std::unique_ptr<RowContainer> mergeRows_;

  // The row with the current merge state, allocated from 'mergeRow_'.
  char* mergeState_ = nullptr;

  // The currently running spill partition in producing spilld output.
  int32_t outputPartition_{-1};

  // Intermediate vector for passing arguments to aggregate in merging spill.
  std::vector<VectorPtr> mergeArgs_;

  // Indicates the element in mergeArgs_[0] that corresponds to the accumulator
  // to merge.
  SelectivityVector mergeSelection_;

  // True if 'merge_' indicates that the next key is the same as the current
  // one.
  bool nextKeyIsEqual_{false};

  // The set of rows that are outside of the spillable hash number
  // ranges. Used when producing output.
  std::optional<Spiller::SpillRows> nonSpilledRows_;

  // Index of first in 'nonSpilledRows_' that has not been added to output.
  size_t nonSpilledIndex_ = 0;
  // Pool of the OperatorCtx. Used for spilling.
  memory::MemoryPool& pool_;

  // Executor for spilling. If nullptr spilling writes on the Driver's thread.
  folly::Executor* FOLLY_NULLABLE const spillExecutor_;
};

} // namespace facebook::velox::exec
