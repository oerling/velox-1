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

#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

  class DistributedPlanBuilder : public PlanBuilder {
  public:
  DistributedPlanBuilder(
			 const ExecutablePlanOptions& options,
			 std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      memory::MemoryPool* pool = nullptr))
    : PlanBuilder(planNodeIdGenerator, pool), options_(options) {}
    
  DistributedPlanBuilder(DistributedPlanBuilder& parent)
    : PlanBuilder(parent.planNodeIdGenerator(), parent.pool()),
    options_(parent.options_),
    parent_(&parent) {
    auto* root = rootBuilder();
    root->stack_.push_back(this);
  }
  
  std::vector<ExecutableFragment> fragments() const {
  }
  
    PlanBuilder& shuffle(
      const std::vector<std::string>& keys,
      int numPartitions,
      bool replicateNullsAndAny,
      const std::vector<std::string>& outputLayout = {}) override {
      partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
      auto* output = dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
      auto producerPrefix = current_.taskPrefix;
      newFragment();
      current_.width = numPartitions;
      exchange(output->outputType());
      auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
      current_.inputs.push_back(InputStage{exchange->id(), producerPrefix});
    }

    
    const core::PlanNodePtr& planNode() shuffleResult(
						      const std::vector<std::string>& keys,
						      int numPartitions,
						      bool replicateNullsAndAny,
						      const std::vector<std::string>& outputLayout = {}) override {
      partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
      auto* output = dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
      auto producerPrefix = current_.taskPrefix;
      auto result = planNode_;
      newFragment();
      auto* consumer = root->stack_.back();
      if (consumer->current_.width != 0) {
	VELOX_CHECK_EQ(numPartitions, consumer->current_.width, "The consumer width should match the producer fanout");
      } else {
	consumer->current_.width = numPartitions;
      }

      root->stack_.pop_back();
      exchange(output->outputType());
      auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
      consumer->current_.inputs.push_back(InputStage{exchange->id(), producerPrefix});
      return std::move(planNode_);
    }

 private:
      void newFragment() {
	if (!current_.taskPrefix.empty()){
	  current_.fragment = core::PlanFragment(std::move(planNode_));
	  fragments_.push_back(std::move(current_));
	}
	auto* root = rootBuilder();
	current_.taskPrefix = fmt::format("{}.{}", options_.queryId, root->taskCounter_++);

      }

      DistributedPlanBuilder& rootBuilder() {
	auto* parent = this;
	while (parent->parent_) {
	  parent parent->parent_;
	}
	return parent;
      }
      const ExecutablePlanOptions& options_;
  DistributedPlanBuilder* parent{nullptr};
  // 
  // Stack of outstanding builders. The last element is the immediately enclosing one. When returning an ExchangeNode from returnShuffle, the stack is used to establish the linkage between the fragment of the returning builder and the fragment current in the calling builder. Only filled in the root builder.
  std::vector<DistributedPlanBuilder*> stack_;
  // Fragment counter. Only used in root builder.
  int32_t taskCounter_{0};
  ExecutableFragment current_;
  std::vector<ExecutableFragment> fragments_;
    
    
    };
}

