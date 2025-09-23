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

#include "velox/exec/ProjectSequence.h"
#include "velox/core/Expressions.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::exec {
OperandIdx findInputOperand(
    const StageData& stage,
    const std::vector<int32_t>& path) {
  for (auto& assignment : stage.input) {
    if (assignment.path == path) {
      return assignment.operand;
    }
  }
  return kNoOperand;
}

void ProjectSequence::markExprFields(
    const core::TypedExprPtr& expr,
    OperandIdx target,
    StageData& state) {
  auto kind = expr->kind();
  if (kind == core::ExprKind::kFieldAccess ||
      kind == core::ExprKind::kDereference) {
    std::vector<int32_t> path;
    if (isField(expr, path)) {
      auto it = state.fieldToOperand.find(expr.get());
      OperandIdx inputIdx = kNoOperand;
      if (it == state.fieldToOperand.end()) {
        if (state.next) {
          inputIdx = findInputOperand(*state.next, path);
        }
        if (inputIdx == kNoOperand) {
          inputIdx = stateCounter_++;
          state.fieldToOperand[expr.get()] = inputIdx;
          state.input.emplace_back(path, inputIdx, state.inputSourceIdx);
        }
      } else {
        inputIdx = OperandIdx(it->second & ~kMultiple);
        it->second |= kMultiple;
      }
      if (target != kNoOperand) {
        state.exprForPath.push_back(nullptr);
        state.identityPaths.push_back({path, target});
      }
      return;
    }
  }
  if (target != kNoOperand) {
    state.exprForPath.push_back(expr);
  }
  for (auto i = 0; i < expr->inputs().size(); ++i) {
    markExprFields(expr->inputs()[i], kNoOperand, state);
  }
}

void ProjectSequence::makeExprStageData(
    const core::TypedExprPtr& expr,
    std::vector<int32_t>& path,
    StageData& state) {
  if (expression::utils::isCall(expr, "row_constructor")) {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    for (auto i = 0; i < call->inputs().size(); ++i) {
      path.push_back(i);
      makeExprStageData(call->inputs()[i], path, state);
      path.pop_back();
    }
    return;
  }
  auto destination = stateCounter_++;
  state.output.emplace_back(path, destination, state.outputSourceIdx);
  markExprFields(expr, destination, state);
}

void ProjectSequence::makeRowStageData(
    const std::vector<core::TypedExprPtr>& exprs,
    StageData& state) {
  std::vector<int32_t> path;

  for (auto i = 0; i < exprs.size(); ++i) {
    path.push_back(i);
    makeExprStageData(exprs[i], path, state);
    path.pop_back();
  }
}

ProjectSequence::ProjectSequence(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const ProjectVector& projects)
    : Operator(
          driverCtx,
          projects.back()->outputType(),
          operatorId,
          projects.front()->id(),
          "ProjectSequence"),
      projects_(projects),
      inputType_(projects_.front()->sources()[0]->outputType()) {}

void ProjectSequence::makeWorkUnits(int stageIdx) {
  const core::AbstractProjectNode* project = projects_[stageIdx].get();
  std::vector<std::vector<core::TypedExprPtr>> groups;
  std::vector<WorkUnit> units;
  if (auto* parallel =
          dynamic_cast<const core::ParallelProjectNode*>(project)) {
    groups = parallel->exprGroups();
  } else {
    groups.push_back(project->projections());
  }
  auto& stage = stages_[stageIdx];
  TranslateCtx ctx(stage, this);
  int exprIdx = 0;
  for (auto& group : groups) {
    units.emplace_back();
    auto& unit = units.back();
    unit.execCtx = std::make_unique<core::ExecCtx>(
        operatorCtx_->pool(),
        operatorCtx_->driverCtx()->task->queryCtx().get());
    unit.program = std::make_unique<ExprProgram>();

    for (auto i = 0; i < group.size(); ++i) {
      auto expr = stage.exprForPath[exprIdx];
      ++exprIdx;
      if (expr) {
        ctx.translateExpr(expr, *unit.program, stage.output[exprIdx].operand);
      }
    }
    ctx.noReuseOfTemp();
  }
  work_.push_back(std::move(units));
}

int32_t findPrefixIdx(
    const std::vector<std::vector<int32_t>>& paths,
    const std::vector<int32_t>& path) {
  std::vector<int32_t> prefix;
  prefix.insert(prefix.end(), path.begin(), path.end() - 1);
  auto it = std::find(paths.begin(), paths.end(), prefix);
  VELOX_CHECK(it != paths.end());
  return it - paths.begin();
}

void ProjectSequence::setLeafRow(std::vector<Assignment>& assignments) {
  for (auto& assignment : assignments) {
    auto sourceRow = assignment.sourceRow;
    auto& paths = allPaths_[assignment.sourceRow];
    if (assignment.path.size() == 1) {
      assignment.leafRow = pathGroupStart_[assignment.sourceRow];
    } else {
      auto idx = findPrefixIdx(allPaths_[sourceRow], assignment.path);
      assignment.leafRow = pathGroupStart_[sourceRow] + idx;
    }
  }
}

void ProjectSequence::initialize() {
  Operator::initialize();
  const auto& inputType = projects_[0]->sources()[0]->outputType();
  stages_.resize(projects_.size());
  for (int32_t level = projects_.size() - 1; level >= 0; --level) {
    if (level < projects_.size() - 1) {
      stages_[level].next = &stages_[level + 1];
    }
    makeRowStageData(projects_[level]->projections(), stages_[level]);
  }
  allPaths();
  tempRowIdx_ = 0;
  for (auto& paths : allPaths_) {
    tempRowIdx_ += paths.size();
  }
  for (auto i = 0; i < projects_.size(); ++i) {
    makeWorkUnits(i);
  }
  for (auto& stage : stages_) {
    setLeafRow(stage.input);
    setLeafRow(stage.output);
  }
  tempVectors_.resize(tempTypes_.size());
  resultRows_.push_back(&tempVectors_);
}

void ProjectSequence::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

bool ProjectSequence::isFinished() {
  return noMoreInput_ && !input_;
}

RowVectorPtr ProjectSequence::getOutput() {
  if (!input_) {
    return nullptr;
  }
  SCOPE_EXIT {
    input_.reset();
  };

  vector_size_t size = input_->size();
  LocalSelectivityVector localRows(*operatorCtx_->execCtx(), size);
  auto* rows = localRows.get();
  VELOX_DCHECK_NOT_NULL(rows);
  rows->setAll();
  setState();

  return results_.back();
}

void listRows(
    const RowType* row,
    std::vector<int32_t>& path,
    std::vector<std::vector<int32_t>>& result) {
  for (auto i = 0; i < row->size(); ++i) {
    auto child = row->childAt(i);
    if (child->kind() == TypeKind::ROW) {
      path.push_back(i);
      result.push_back(path);
      listRows(&child->as<TypeKind::ROW>(), path, result);
      path.pop_back();
    }
  }
}

void ProjectSequence::allPaths() {
  std::vector<int32_t> empty;
  allPaths_.emplace_back();
  listRows(inputType_.get(), empty, allPaths_.back());
  pathGroupStart_.push_back(0);
  pathGroupStart_.push_back(allPaths_.back().size());
  for (auto& project : projects_) {
    allPaths_.emplace_back();
    listRows(project->outputType().get(), empty, allPaths_.back());
    pathGroupStart_.push_back(pathGroupStart_.back() + allPaths_.back().size());
  }
}

std::vector<VectorPtr>* getRowAt(
    RowVector* row,
    const std::vector<int32_t> path) {
  for (auto idx : path) {
    row = row->childAt(idx)->as<RowVector>();
    VELOX_CHECK_EQ(row->encoding(), VectorEncoding::Simple::ROW);
  }
  return &row->children();
}

void ProjectSequence::initState(const std::vector<Assignment>& assignments) {
  if (assignments.empty()) {
    return;
  }
  for (auto& assignment : assignments) {
    state_[operandIdx(assignment.operand)] =
        &(*resultRows_[assignment.leafRow])[assignment.path.back()];
  }
}

void ProjectSequence::setState() {
  state_.resize(stateCounter_);
  if (results_.empty()) {
    for (auto& project : projects_) {
      results_.push_back(
			 BaseVector::create<RowVector>(std::static_pointer_cast<const Type>(project->outputType()), input_->size(), operatorCtx_->pool()));
    }
  }
  auto& inputPaths = allPaths_.front();
  for (auto i = 0; i < inputPaths.size(); ++i) {
    resultRows_[i] = getRowAt(input_.get(), inputPaths[i]);
  }
  int32_t fill = inputPaths.size();

  for (auto i = 0; i < stages_.size(); ++i) {
    auto& paths = allPaths_[i + 1];
    for (auto i = 0; i < paths.size(); ++i) {
      resultRows_[fill + i] = getRowAt(results_[i].get(), paths[i]);
    }
    fill += paths.size();
  }

  for (int32_t i = stages_.size() - 1; i >= 0; --i) {
    auto& stage = stages_[i];
    initState(stage.output);
    initState(stage.input);
  }
  for (auto i = 0; i < tempVectors_.size(); ++i) {
    state_[firstTempIdx_ + i] = &tempVectors_[i];
  }
}

} // namespace facebook::velox::exec
