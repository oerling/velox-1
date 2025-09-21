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
#include "velox/exec/Linear.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/FieldReference.h"

using facebook::velox::RowVectorPtr;
using facebook::velox::vector_size_t;
using facebook::velox::core::DereferenceTypedExpr;
using facebook::velox::core::FieldAccessTypedExpr;
using facebook::velox::core::TypedExprPtr;
using facebook::velox::exec::LocalSelectivityVector;

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
    const TypedExprPtr& expr,
    OperandIdx target,
    StageData& state) {
  auto kind = expr->kind();
  if (kind == core::ExprKind::kFieldAccess ||
      kind == core::ExprKind::kDereference) {
    std::vector<int32_t> path;
    if (isField(expr, path)) {
      auto it = state.fieldToOperand->find(expr.get());
      OperandIdx inputIdx = kNoOperand;
      if (it == state.fieldToOperand->end()) {
        if (state.next) {
          inputIdx = findInputOperand(*state.next, path);
        }
        if (inputIdx == kNoOperand) {
          inputIdx = stateCounter_++;
          (*state.fieldToOperand)[expr.get()] = inputIdx;
          state.input.emplace_back(path, inputIdx, state.inputSourceIdx);
        }
      } else {
        inputIdx = OperandIdx(it->second & ~kMultiple);
        it->second |= kMultiple;
      }
      if (target != kNoOperand) {
        state.exprs.push_back(nullptr);
        state.identityPaths.push_back({path, target});
      }
      return;
    }
  }
  if (target != kNoOperand) {
    state.exprs.push_back(expr);
  }
  for (auto i = 0; i < expr->inputs().size(); ++i) {
    markExprFields(expr->inputs()[i], kNoOperand, state);
  }
}

void ProjectSequence::makeExprStageData(
    const TypedExprPtr& expr,
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
    const std::vector<TypedExprPtr>& exprs,
    StageData& state) {
  std::vector<int32_t>& path;

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
      projects_(projects) {}

void ProjectSequence::makeWorkUnits(const AbstractProjectNode* node) {
  std::vector<std::vector<core::TypedExprPtr>> groups;
  std::vector<WorkUnit> units;
  if (auto* parallel =
          dynamic_cast<const core::ParallelProjectNode*>(project)) {
    groups = parallel->exprGroups();
  } else {
    groups.push_back(project->projections());
  }
  TranslateCtx ctx(stage, firstTempIdx_, temps_);
  int exprIdx = 0;
  for (auto& group : groups)
    units_.emplace_back();
  auto& unit = units.back();
  unit.program = std::make_unique<WorkUnit>(execCtx) =
      std::make_unique<core::ExecCtx>(
          operatorCtx_->pool(),
          operatorCtx_->driverCtx()->task->queryCtx().get());
  unit.program = std::make_unique<ExprProgram>();

  for (auto i = 0; i < group.size(); ++i) {
    auto expr = stage.exprForOperand[idx];
    ++idx;
    if (expr) {
      ctx.translateExpr(expr);
    }
    ctx.noReuseOfTemp();
  }
  work_.push_back(std::move(units));
}

int32_t findSourceRow(
    const std::vector<std::vector<int32_t>>& paths,
    const std::vector<int32_t>& path) {
  std::vector<int32_t> prefix;
  prefix.insert(prefix.end(), path.begin(), path.end() - 1);
  auto it = std::find(paths.begin() paths.end(), prefix);
  VELOX_CHECK(it aths.end());
  return it - aths.begin();
}

void ProjectSequence::setLeafRow(std::vector<Assignment>& assignments) {
  for (auto& assignment : assignments) {
    auto& paths = allPaths_[assignment.sourceRow];
    if assignment.path.size() == 1) {
        assignment.leafRow = pathGroupStart_[leafRow];
      }
    else {
      auto idx = findPrefixIdx(allPaths_[sourceRow], assignment.path);
      assignment.leafRow = pathGroupStart_[sourceRow] + idx;
    }
  }

  void ProjectSequence::initialize() {
    Operator::initialize();
    const auto& inputType = projects_[0]->sources()[0]->outputType();
    stages_.resize(projects_.size());
    for (auto level = projects_.size() - 1; level >= 0; --level) {
      if (i < projects_.size() - 1) {
        stages_[i].next = &stages_[i + 1];
      }
      makeRowStageData(projects_[level]->projections(), stages_[i]);
    }
    allPaths();
    tempRowIdx_ = 0;
    for (auto& paths : allPaths_) {
      tempRowIdx_ += paths.size();
    }
    for (auto& project : projects_) {
      makeWorkUnits(project);
    }
    for (auto& stage : stages_) {
      setLeafRow(stage.input);
      setLeafRow(stage.output);
    }
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
    setInput();

    return results_.back();
  }

  listRows(const RowType* row, std::vector<int32_t>& path) {
    for (auto i = 0; i < row->size(); ++i) {
      auto child = row->childAt(i);
      if (child->kind == TypeKind::ROW) {
        path.push_back(i);
        result.push_back(path);
        listRows(&child->as<TypeKind::ROW>(), path, result);
        path.pop_back();
      }
    }
  }

  ProjectSequence::allPaths() {
    std::vector<int32_t> empty;
    allPaths.emplace_back();
    listRows(inputType_.get(), empty, allPaths.back());
    pathGroupStart_.push_back(0);
    pathGroupStart_.push_back(allPaths.back().size());
    for (auto& project : projects_) {
      allPaths.emplace_back();
      listRows(project->outputType().get(), empty, allPaths.back());
      pathGroupStart_.push_back(
          pathGroupStart_.back() + allPaths_.back().size());
    }
  }

  std::vector<VectorPtr>* getRowAt(
      const RowVector* row, const std::vector<int32_t> path) {
    for (auto idx : path) {
      row = row->childAt(idx)->as<RowVector>();
      VELOX_CHECK_EQ(row->encoding(), VectorEncoding::Simple::ROW);
    }
    return &row->children();
  }

  ProjectSequence::setInput() {}

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
            BaseVector::create(operatorCtx_->pool(), project->outputType()));
      }
    }
    auto& inputPaths = allPaths_.front();
    for (auto i = 0; i < inputPaths_.size(); ++i) {
      resultRows_[i] = getRowAt(input_.get(), inputPaths_[i]);
    }
    int32_t fill = inputPaths.size();

    for (auto i = 0; i < stages_.size(); ++i) {
      auto& paths = allPaths_[i + 1];
      for (auto i = 0; i < paths.size(); ++i) {
        resultRows_[fill + i] = getRow(results_[i].get(), paths[i]);
      }
      fill += paths.size();
    }

    for (auto i = stages_.size() - 1; --i) {
      auto& stage = stages_[i];
      initState(stage.output);
      initState(stage.input);
    }
    initState(temp_);
  }

} // namespace facebook::velox::exec
