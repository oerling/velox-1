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

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"
#include "velox/expression/ExprToSubfieldFilter.h"

namespace facebook::verax {

using namespace facebook::velox;
using namespace facebook::velox::exec;

void filterUpdated(BaseTablePtr table) {
  auto optimization = queryCtx()->optimization();
  std::vector<core::TypedExprPtr> remainingConjuncts;
  connector::hive::SubfieldFilters subfieldFilters;
  for (auto filter : table->columnFilters) {
    auto typedExpr = optimization->toTypedExpr(filter);
    auto pair = velox::exec::toSubfieldFilter(typedExpr);
    if (!pair.second) {
      remainingConjuncts.push_back(std::move(typedExpr));
      continue;
    }
    subfieldFilters[std::move(pair.first)] = std::move(pair.second);
  }
  for (auto expr : table->filter) {
    remainingConjuncts.push_back(optimization->toTypedExpr(expr));
  }
  core::TypedExprPtr remainingFilter;
  for (auto conjunct : remainingConjuncts) {
    if (!remainingFilter) {
      remainingFilter = conjunct;
    } else {
      remainingFilter = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{remainingFilter, conjunct},
          "and");
    }
  }
  const char* connector = table->schemaTable->indices[0]
                              ->distribution()
                              .distributionType.locus->name();
  auto handle = std::make_shared<connector::hive::HiveTableHandle>(
      connector,
      table->schemaTable->name,
      true,
      std::move(subfieldFilters),
      remainingFilter);
  optimization->setLeafHandle(table->id(), handle);
}

RelationOpPtr addGather(RelationOpPtr op) {
  if (op->relType() == RelType::kOrderBy) {
    auto order = op->distribution();
    Distribution final = Distribution::gather(
        op->distribution().distributionType, order.order, order.orderType);
    Declare(Repartition, gather, op, final, op->columns());
    Declare(OrderBy, orderBy, gather, order.order, order.orderType);
    return orderBy;
  }
  Declare(
      Repartition,
      gather,
      op,
      Distribution::gather(op->distribution().distributionType),
      op->columns());
  return gather;
}

std::vector<ExecutableFragment> Optimization::toVeloxPlan(
    RelationOpPtr plan,
    const ExecutablePlanOptions& options) {
  options_ = options;
  std::vector<ExecutableFragment> stages;
  if (options_.numWorkers > 1) {
    plan = addGather(plan);
  }
  ExecutableFragment top;
  makeFragment(plan, top, stages);
  stages.push_back(std::move(top));
  return stages;
}

RowTypePtr Optimization::makeOutputType(const ColumnVector& columns) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < columns.size(); ++i) {
    names.push_back(columns[i]->name());
    types.push_back(toTypePtr(columns[i]->value().type));
  }
  return ROW(std::move(names), std::move(types));
}

core::TypedExprPtr Optimization::toAnd(const ExprVector& exprs) {
  core::TypedExprPtr result;
  for (auto expr : exprs) {
    auto conjunct = toTypedExpr(expr);
    if (!result) {
      result = conjunct;
    } else {
      result = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(), std::vector<core::TypedExprPtr>{result, conjunct}, "and");
    }
  }
  return result;
}

core::TypedExprPtr Optimization::toTypedExpr(ExprPtr expr) {
  switch (expr->type()) {
    case PlanType::kColumn: {
      return std::make_shared<core::FieldAccessTypedExpr>(
          toTypePtr(expr->value().type), expr->as<Column>()->name());
    }
    case PlanType::kCall: {
      std::vector<core::TypedExprPtr> inputs;
      auto call = expr->as<Call>();
      for (auto arg : call->args()) {
        inputs.push_back(toTypedExpr(arg));
      }
      return std::make_shared<core::CallTypedExpr>(
          toTypePtr(expr->value().type), std::move(inputs), call->name());
    }
    case PlanType::kLiteral: {
      auto literal = expr->as<Literal>();
      return std::make_shared<core::ConstantTypedExpr>(
          toTypePtr(literal->value().type), literal->literal());
    }

    default:
      VELOX_FAIL("Cannot translate {} to TypeExpr", expr->toString());
  }
}

core::PlanNodePtr Optimization::maybeProject(
    const ExprVector& exprs,
    core::PlanNodePtr source,
    std::vector<core::FieldAccessTypedExprPtr>& result) {
  bool anyNonColumn = false;
  for (auto expr : exprs) {
    result.push_back(
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
            toTypedExpr(expr)));
    if (expr->type() != PlanType::kColumn) {
      anyNonColumn = true;
    }
  }
  if (!anyNonColumn) {
    return source;
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<core::TypedExprPtr> projections;
  for (auto i = 0; i < source->outputType()->size(); ++i) {
    names.push_back(source->outputType()->nameOf(i));
    types.push_back(source->outputType()->childAt(i));
    projections.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        types.back(), names.back()));
  }
  for (auto i = 0; i < exprs.size(); ++i) {
    names.push_back(fmt::format("r{}", ++resultNameCounter_));
    types.push_back(toTypePtr(exprs[i]->value().type));
    projections.push_back(toTypedExpr(exprs[i]));
    result.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        types.back(), names.back()));
  }
  return std::make_shared<core::ProjectNode>(
      idGenerator_.next(), std::move(names), std::move(projections), source);
}

// Translates ExprPtrs to FieldAccessTypedExprs. Maintains a set of
// projections and produces a ProjectNode to evaluate distinct
// expressions for non-column Exprs given to toFieldref() and
// related functions.
class TempProjections {
 public:
  TempProjections(Optimization& optimization, const RelationOp& input)
      : optimization_(optimization), input_(input) {
    for (auto& column : input_.columns()) {
      exprChannel_[column] = nextChannel_++;
      names_.push_back(column->name());
      fieldRefs_.push_back(std::make_shared<core::FieldAccessTypedExpr>(
          optimization_.toTypePtr(column->value().type), column->name()));
    }
    exprs_.insert(exprs_.begin(), fieldRefs_.begin(), fieldRefs_.end());
  }

  core::FieldAccessTypedExprPtr toFieldRef(ExprPtr expr) {
    auto it = exprChannel_.find(expr);
    if (it == exprChannel_.end()) {
      VELOX_CHECK_NE(expr->type(), PlanType::kColumn);
      exprChannel_[expr] = nextChannel_++;
      exprs_.push_back(optimization_.toTypedExpr(expr));
      names_.push_back(fmt::format("__r{}", nextChannel_ - 1));
      fieldRefs_.push_back(std::make_shared<core::FieldAccessTypedExpr>(
          optimization_.toTypePtr(expr->value().type), names_.back()));
      return fieldRefs_.back();
    }
    return fieldRefs_[it->second];
  }

  template <typename Result = core::FieldAccessTypedExprPtr>
  std::vector<Result> toFieldRefs(const ExprVector& exprs) {
    std::vector<Result> result;
    for (auto expr : exprs) {
      result.push_back(toFieldRef(expr));
    }
    return result;
  }

  core::PlanNodePtr maybeProject(core::PlanNodePtr inputNode) {
    if (nextChannel_ == input_.columns().size()) {
      return inputNode;
    }
    return std::make_shared<core::ProjectNode>(
        optimization_.idGenerator().next(),
        std::move(names_),
        std::move(exprs_),
        inputNode);
  }

 private:
  Optimization& optimization_;
  const RelationOp& input_;
  int32_t nextChannel_;
  std::vector<core::FieldAccessTypedExprPtr> fieldRefs_;
  std::vector<std::string> names_;
  std::vector<core::TypedExprPtr> exprs_;
  std::map<ExprPtr, int32_t> exprChannel_;
};

core::PlanNodePtr Optimization::makeAggregation(
    Aggregation& op,
    ExecutableFragment& fragment,
    std::vector<ExecutableFragment>& stages) {
  auto input = makeFragment(op.input(), fragment, stages);
  TempProjections projections(*this, *op.input());

  std::vector<std::string> aggregateNames;
  std::vector<core::CallTypedExprPtr> aggregates;
  std::vector<core::FieldAccessTypedExprPtr> masks;
  bool isRawInput = op.step == core::AggregationNode::Step::kPartial ||
      op.step == core::AggregationNode::Step::kSingle;
  bool isIntermediateOutput =
      op.step == core::AggregationNode::Step::kPartial ||
      op.step == core::AggregationNode::Step::kIntermediate;
  for (auto i = 0; i < op.aggregates.size(); ++i) {
    aggregateNames.push_back(op.columns()[i + op.grouping.size()]->name());
    auto aggregate = op.aggregates[i];
    if (isRawInput) {
      if (aggregate->condition()) {
        masks.resize(i + 1);
        masks[i] = projections.toFieldRef(aggregate->condition());
      }
      aggregates.push_back(std::make_shared<core::CallTypedExpr>(
          toTypePtr(aggregate->value().type),
          projections.toFieldRefs<core::TypedExprPtr>(aggregate->args()),
          aggregate->name()));
    } else {
      aggregates.push_back(std::make_shared<core::CallTypedExpr>(
          toTypePtr(aggregate->value().type),
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(
                  toTypePtr(aggregate->value().type), fmt::format("a{}", i))},
          aggregate->name()));
    }
  }
  auto keys = projections.toFieldRefs(op.grouping);
  auto project = projections.maybeProject(input);
  auto r =new core::AggregationNode(
      idGenerator_.next(),
      op.step,
      keys,
      {},
      aggregateNames,
      aggregates,
      masks,
      false,
      project);
  core::PlanNodePtr ptr;
  ptr.reset(r);
  return ptr;
}
  
core::PlanNodePtr Optimization::makeFragment(
    RelationOpPtr op,
    ExecutableFragment& fragment,
    std::vector<ExecutableFragment>& stages) {
  switch (op->relType()) {
    case RelType::kProject: {
      auto input = makeFragment(op->input(), fragment, stages);
      auto project = op->as<Project>();
      std::vector<std::string> names;
      std::vector<core::TypedExprPtr> exprs;
      for (auto i = 0; i < project->exprs().size(); ++i) {
        names.push_back(project->columns()[i]->name());
        exprs.push_back(toTypedExpr(project->exprs()[i]));
      }
      return std::make_shared<core::ProjectNode>(
          idGenerator_.next(), std::move(names), std::move(exprs), input);
    }
    case RelType::kFilter: {
      auto filter = op->as<Filter>();
      return std::make_shared<core::FilterNode>(
          idGenerator_.next(),
          toAnd(filter->exprs()),
          makeFragment(filter->input(), fragment, stages));
    }
    case RelType::kAggregation: {
      return makeAggregation(*op->as<Aggregation>(), fragment, stages);
    }
    case RelType::kRepartition: {
      ExecutableFragment source;
      source.taskPrefix = fmt::format("stage{}", ++stageCounter_);
      source.fragment.planNode = makeFragment(op->input(), source, stages);
      stages.push_back(std::move(source));
      break;
    }
    case RelType::kTableScan: {
      auto scan = op->as<TableScan>();
      auto handle = leafHandle(scan->baseTable->id());
      if (!handle) {
        filterUpdated(scan->baseTable);
        handle = leafHandle(scan->baseTable->id());
        VELOX_CHECK(handle, "No table for scan {}", scan->toString(true, true));
      }
      std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          assignments;
      for (auto column : scan->columns()) {
        assignments[column->name()] =
            std::make_shared<connector::hive::HiveColumnHandle>(
                column->name(),
                connector::hive::HiveColumnHandle::ColumnType::kRegular,
                toTypePtr(column->value().type));
      }
      return std::make_shared<core::TableScanNode>(
          idGenerator_.next(),
          makeOutputType(scan->columns()),
          handle,
          assignments);
    }
    case RelType::kJoin: {
      auto join = op->as<Join>();
      auto left = makeFragment(op->input(), fragment, stages);
      auto right = makeFragment(op->input(), fragment, stages);
      std::vector<core::FieldAccessTypedExprPtr> leftKeys;
      left = maybeProject(join->leftKeys, left, leftKeys);
      std::vector<core::FieldAccessTypedExprPtr> rightKeys;
      right = maybeProject(join->rightKeys, right, rightKeys);
      return std::make_shared<core::HashJoinNode>(
          idGenerator_.next(),
          join->joinType,
          false,
          leftKeys,
          rightKeys,
          toAnd(join->filter),
          left,
          right,
          makeOutputType(join->columns()));
    }
    default:
      VELOX_FAIL("Unsupported RelationOp {}", op->relType());
  }
  return nullptr;
}

} // namespace facebook::verax
