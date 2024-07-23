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

#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"

DEFINE_int32(
    wave_stripe_rows,
    100000,
    "Stripe size for Wave recoding of test data");

namespace facebook::velox::wave::test {

class TestingWaveRecodeNode : public core::PlanNode {
 public:
  explicit TestingWaveRecodeNode(core::PlanNodePtr input)
      : PlanNode("WaveRecode"), sources_{input} {}

  TestingWaveRecodeNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr input,
      const std::string& tableName)
      : PlanNode(id), sources_{input}, tableName_(tableName) {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  const std::string& tableName() const {
    return tableName_;
  }

  std::string_view name() const override {
    return "WaveRecode";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
  std::string tableName_;
};

class WaveRecoder : public Operator {
 public:
  WaveRecoder(
      DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const TestingWaveRecodeNode> node)
      : Operator(ctx, node->outputType(), id, node->id(), "WaveRecoder"),
        tableName_(node->tableName()) {}

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override {
    if (!table_) {
      table_ = std::make_unique<wave::test::Writer>(FLAGS_wave_stripe_rows);
    }
    table_->append(input);
  }

  void noMoreInput() override {
    if (table_) {
      table_->finalize(tableName_);
    }
    Operator::noMoreInput();
  }

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_;
  }

 private:
  std::string tableName_;
  std::unique_ptr<wave::test::Writer> table_;
};

class WaveRecodeFactory : public Operator::PlanNodeTranslator {
 public:
  WaveRecodeFactory() {}

  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto recode =
            std::dynamic_pointer_cast<const TestingWaveRecodeNode>(node)) {
      return std::make_unique<WaveRecoder>(ctx, id, recode);
    }
    return nullptr;
  }

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override {
    return std::nullopt;
  }

 private:
};

TpchPlan TpchQueryBuilder::getWaveRecodePlan(
    const std::string& tableName) const {
  static bool inited = false;
  if (!inited) {
    inited = true;
    Operator::registerOperator(std::make_unique<WaveRecodeFactory>());
  }

  auto columns = getFileColumnNames(tableName);
  std::vector<std::string> names;
  for (auto& pair : columns) {
    names.push_back(pair.first);
  }

  const auto selectedRowType = getRowType(tableName, names);

  core::PlanNodeId scanPlanNodeId;
  std::unordered_map<std::string, std::string> aliases;
  for (auto& name : names) {
    aliases[name] = name;
  }
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(tableName, selectedRowType, aliases, {})
                  .capturePlanNodeId(scanPlanNodeId)
                  .addNode([&](std::string id, core::PlanNodePtr input) {
                    return std::make_shared<TestingWaveRecodeNode>(
                        id, input, tableName);
                  })
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[scanPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

} // namespace facebook::velox::wave::test
