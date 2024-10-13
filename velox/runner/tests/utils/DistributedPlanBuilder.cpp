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

#include "velox/runner/tests/utils/DistributedPlanBuilder.h"

namespace facebook::velox::exec::test {

std::vector<ExecutableFragment> DistributedPlanBuilder::fragments() const {
  newFragment();
  return std::move(fragments_);
}

void DistributedPlanBuilder::newFragment() {
  if (!current_.taskPrefix.empty()) {
    gatherScans(planNode_);
    current_.fragment = core::PlanFragment(std::move(planNode_));
    fragments_.push_back(std::move(current_));
  }
  current_ = ExecutableFragment();
  auto* root = rootBuilder();
  current_.taskPrefix =
      fmt::format("{}.{}", options_.queryId, root->taskCounter_++);
}

PlanBuilder& DistributedPlanBuilder::shuffle(
    const std::vector<std::string>& keys,
    int numPartitions,
    bool replicateNullsAndAny,
    const std::vector<std::string>& outputLayout = {}) override {
  partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
  auto producerPrefix = current_.taskPrefix;
  newFragment();
  current_.width = numPartitions;
  exchange(output->outputType());
  auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
  current_.inputs.push_back(InputStage{exchange->id(), producerPrefix});
  return *this;
}

core::PlanNodePtr planNode() shuffleResult(
    const std::vector<std::string>& keys,
    int numPartitions,
    bool replicateNullsAndAny,
    const std::vector<std::string>& outputLayout = {}) override {
  partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
  auto producerPrefix = current_.taskPrefix;
  auto result = planNode_;
  newFragment();
  auto* consumer = root->stack_.back();
  if (consumer->current_.width != 0) {
    VELOX_CHECK_EQ(
        numPartitions,
        consumer->current_.width,
        "The consumer width should match the producer fanout");
  } else {
    consumer->current_.width = numPartitions;
  }

  root->stack_.pop_back();

  for (auto& fragment : fragments_) {
    root_->fragments_.push_back(std::move(fragment));
  }
  exchange(output->outputType());
  auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
  consumer->current_.inputs.push_back(
      InputStage{exchange->id(), producerPrefix});
  return std::move(planNode_);
}

void DistributedPlanBuilder::gatherScans(const PlanNodePtr& plan.get()) {
  if (auto scan = std::dynamic_pointer_cast<const TableScanNode>(plan)) {
    current_.scans.push_back(scan);
    return;
  }
  for (auto& in : plan->sources()) {
    gatherScans(in);
  }
}
