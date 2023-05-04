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

#include "velox/exec/Operator.h"


namespace facebook::velox::exec::test {


class TestingRebatchNode : public core::PlanNode {
 public:
  explicit TestingRebatchNode(core::PlanNodePtr input)
      : PlanNode("Rebatch"), sources_{input} {}

  TestingRebatchNode(const core::PlanNodeId& id, core::PlanNodePtr input)
      : PlanNode(id), sources_{input} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "Rebatch";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};



class TestingRebatch : public Operator {
 public:
  enum class Twist {kConstant, kLongFlat, kShortFlat, kDicts, kSameDict, kSameDoubleDict};

 TestingRebatch(
      DriverCtx* ctx,
      int32_t id,
      std::shared_ptr<const TestingPauserNode> node,
      DriverTest* test,
      int32_t sequence)
    : Operator(ctx, node->outputType(), id, node->id(), "Rebatch") {}

  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  void noMoreInput() override {
    Operator::noMoreInput();
  }

  RowVectorPtr getOutput() override; 

  BlockingReason isBlocked(ContinueFuture* future) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:

  // Counter deciding the next action in getOutput().
  int32_t counter_;

  // Next row of input to be sent to output.
  vector_size_t currentRow_{0};

  // Drop every second row of input. Used for introducing a predictable error to test drilldown into minimal breaking fuzziness.
  bool injectError_{false};
};
 
}
