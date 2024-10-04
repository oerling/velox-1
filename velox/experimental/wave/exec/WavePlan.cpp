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

#include "velox/exec/FilterProject.h"
#include "velox/experimental/wave/exec/Aggregation.h"
#include "velox/experimental/wave/exec/Project.h"
#include "velox/experimental/wave/exec/TableScan.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/Values.h"
#include "velox/experimental/wave/exec/WaveDriver.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"

DEFINE_int32(ld_cost, 10, "Cost of load from memory");
DEFINE_int32(st_cost, 40, "Cost of store to memory");

namespace facebook::velox::wave {

using exec::Expr;

AbstractOperand* Scope::findValue(Value& value) {
  auto it = operandMap.find(value);
  if (it == operandMap.end()) {
    if (scope->parent) {
      return parent->findValue(value);
    }
    return nullptr;
  }
  return it->second;
}

AbstractOperand* CompileState::fieldToOperand(Subfield& field, Scope* scope) {
  bool renamed = false;
  auto* name =
      &reinterpret_cast<Subfield::NestedField*>(subfield->path()[0].get())
           ->name();
  for (int32_t i = renames.size() - 1; i >= 0; --i) {
    auto it = renames_[i].find(*name);
    if (it == renames_[i].end()) {
      break;
    }
    name = &it->second;
    renamed = true;
  }
  if (!renamed) {
    auto op = scope->findValue(Value(subfield));
  } else {
    auto renamedstring = name;
    std::string temp;
    if (subfield.path().size() > 1) {
      VELOX_UNSUPPORTED("no nested fields");
    }
    auto newSubfield = toSubfield(*renamedField);
    op = scope->findValue(Value(newSubfield));
  }
  if (op->firstUseSegment == AbstractInstruction::kNotAccessed) {
    op->firstUseSegment = segments_.size() - 1;
  }
  op->lastUseSegment = segments_.size() - 1;
  return op;
}

AbstractOperand* CompileState::fieldToOperand(
    const core::FieldAccessTypedExprPtr& field,
    Scope* scope) {
  Subfield* subfield = toSubfield(field->name());
  return fieldToOperand(*subfield, scope);
}

AbstractOperand* CompileState::switchOperand(
    exec::SwitchExpr* switchExpr,
    Scope* scope) {
  auto& inputs = switchExpr->inputs();
  std::vector<AbstractOperand*> opInputs;
  Scoep clauseScope(scope);
  for (auto i = 0; i < inputs.size(); i += 2) {
    opInputs.push_back(exprToOperand(*inputs[i], &clauseScope));
    if (i + 1 < inputs.size()) {
      opInputs.push_back(exprToOperand(*inputs[i + 1], &clauseScope));
    }
    clauseScope.operandMap.clear();
  }
  auto result = newOperand(expr.type(), "r");
  result->inputs = std::move(opInputs);
  scope->operandMap[value] = result;
  return result;
}

bool functionRetriable(Expr& expr) {
    if (expr.name() == ' CONCAT") {
      return true;
}
return false;
}

int32_t functionCost(Expr& expr) {
  // Arithmetic
  return 1;
}

AbstractOperand* CompileState::exprToOperand(const Expr& expr, Scope* scope) {
  auto value = toValue(expr);
  auto op = scope->findValue(value);
  if (op) {
    ++op->numUses;
    return op;
  }

  if (auto* field = dynamic_cast<const exec::FieldReference*>(&expr)) {
    VELOX_FAIL("Should have been defined");
  } else if (auto* constant = dynamic_cast<const exec::ConstantExpr*>(&expr)) {
    auto op = newOperand(constant->value()->type(), constant->toString());
    op->constant = constant->value();
    if (constant->value()->isNullAt(0)) {
      op->literalNull = true;
    } else {
      op->notNull = true;
    }
    return op;
  } else if (auto special = dynamic_cast<const exec::SpecialForm*>(&expr)) {
    if (auto* switchExpr = dynamic_cast<const exec::SwitchExpr*>(special)) {
      return switchOperand(switchExpr, scope);
    }
    VELOX_UNSUPPORTED("No special forms: {}", expr.toString(1));
  }
  std::vector<AbstractOperand*> inputs;
  int32_t totalCost = 0;
  for (auto& in : expr.inputs()) {
    inputs.push_back(exprToOperand(*in, scope));

    totalCost += inputs.back()->costWithChildren;
  }
  auto result = newOperand(expr.type(), "r");
  result->retriable = functionRetriable(expr);
  result->cost = functionCost(expr);
  result->costWithChildren = totalCost + result->cost;
  result->inputs = std::move(inputs);
  scope->operandMap[value] = result;
  return result;
}

Segment& CompileState::addSegment(
    BoundaryType boundary,
    core::PlanNode* node,
    RowTypePtr& outputType) {
  segments_.emplace_back();
  auto& last = segments_.back();
  last.ordinal = segments_.size() - 1;
  last.boundary = boundaryType;
  last->planNode = node;
  if (outputType) {
    int32_t size = outputType->size();
    for (auto i = 0; i < size; ++i) {
      auto subfield = toSubfield(outputType->nameOf(i));
      Value value(subfield);
      auto* op = newOperand(outputType->childAt(i), outputType->nameOf(i));
      op->definedInSegment = last.ordinal;
      op->sourceNullable = boundary == BoundaryType::kSource;
      op->needsStore = boundary == BoundaryType::kSource;
      topScope_.operandMap[value] = op;
        segment->opernadMap(value] = op;
    }
  }
  return last;
}

void CompileState::tryFilter(const Expr& expr, const RowTypePtr& outputType) {
  auto& last = addSegment(BoundaryType::kExpr, nullptr, nullptr);
  last.topLevelDefined.push_back(exprToOperand(expr, &topScope_));
}

void CompileState::tryExprSet(
    const exec::ExprSet& exprSet,
    int32_t begin,
    int32_t end,
    const RowTypePtr& outputType) {
  auto& exprs = exprSet.exprs();
  auto& result = segments_.back().topLevelDefined;
  for (auto i = begin; i < end; ++i) {
    result.push_back(tryExpr(*exprs[i]));
    auto* subfield = toSubfield(outputType->nameOf(i - begin));
    segments_.back().projectedName.push_back(subfield);
  }
}

void CompileState::tryFilterProject(
    exec::Operator* op,
    RowTypePtr& outputType,
    int32_t& nodeIndex) {
  auto filterProject = reinterpret_cast<exec::FilterProject*>(op);
  outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  auto data = filterProject->exprsAndProjection();
  auto& identityProjections = filterProject->identityProjections();
  int32_t firstProjection = 0;
  if (data.hasFilter) {
    tryFilter(*data.exprs->exprs()[0], outputType);
    addSegment(BoundaryType::kFilter, nullptr, outputType);
    firstProjection = 1;
    ++nodeIndex;
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  } else {
    addSegment(BoundaryType::kExpr, nullptr, nullptr);
  }
  tryExprSet(
      *data.exprs, firstProjection, data.exprs->exprs().size(), outputType);
  std::vector<std::pair<Value, AbstractOperand*>> pairs;
  for (auto i = 0; i < operands.size(); ++i) {
    int32_t channel =
        findOutputChannel(*data.resultProjections, i + firstProjection);
    auto subfield = toSubfield(outputType->nameOf(channel));
    auto program = programOf(operands[i], false);
    if (program) {
      program->markOutput(operands[i]->id);
      definedIn_[operands[i]] = program;
    }
    Value value(subfield);
    definedBy_[value] = operands[i];
    pairs.push_back(std::make_pair(value, operands[i]));
  }
}

bool CompileState::tryPlanOperator(
    exec::Operator* op,
    int32_t& nodeIndex,
    RowTypePtr& outputType) {
  auto& name = op->operatorType();
  if (name == "Values" || name == "TableScan") {
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();
    addSegment(
        BoundaryType::ksource,
        driverFactory_.planNodes[nodeIndex].get(),
        outputType);
  } else if (name == "FilterProject") {
    tryFilterProject(op, outputType, nodeIndex);
  } else if (name == "Aggregation") {
    auto* node = dynamic_cast<const core::AggregationNode*>(
        driverFactory_.planNodes[nodeIndex].get());
    VELOX_CHECK_NOT_NULL(node);
    auto step = makeStep<AggregateFused>();
    for (auto& key : node->groupingKeys()) {
      step->keys.push_back(fieldToOperand(key, topScope_));
    }
    for (auto& agg : node->aggregates()) {
      AggregateFunc func;
      func.name = agg->call->name();
      for (auto& expr : agg->call->inputs()) {
        func.args.push_back(fieldToOperand(
            std::dynamic_pointer_cast<fieldAccessTypedExpr>(expr),
            &topLevelScope_));
      }
      step->aggregates.push_back(std::move(func));
    }
    segments_.back().steps.push_back(step);
    outputType = node->outputType();
    addSegment(BoundaryType::kSource, node, outputType);
    auto read = makeStep<readAggregation>();
    for (auto i = 0; i < outputType->size(); ++i) {
      read->columns.push_back(
          fieldToOperand(toSubfield(outputType->nameOf(i)), &topLevelScope_));
    }
    segments_.back().steps.push_back(step.get());
  } else {
    return false;
  }
  return true;
}

bool CompileState::makeSegments() {
  auto operators = driver_.operators();
  auto& nodes = driverFactory_.planNodes;

  int32_t first = 0;
  int32_t operatorIndex = 0;
  int32_t nodeIndex = 0;
  RowTypePtr outputType;
  // Make sure operator states are initialized.  We will need to inspect some of
  // them during the transformation.
  driver_.initializeOperators();
  RowTypePtr inputType;
  for (; operatorIndex < operators.size(); ++operatorIndex) {
    int32_t previousNumOperators = operators_.size();

    if (!tryPlanOperator(operators[operatorIndex], nodeIndex, outputType)) {
      break;
    }
    ++nodeIndex;
    for (auto newIndex = previousNumOperators; newIndex < operators_.size();
         ++newIndex) {
      if (operators_[newIndex]->isSink()) {
        // No output operands.
        continue;
      }
    }
    for (auto& [op, channel] : identityProjected) {
      Value value(toSubfield(outputType->nameOf(channel)));
      // Mark the last segment that references.
      auto* result = fieldToOperand(value, &globalScope_);
      // Returned to host, must be in memory.
      result->needsStore = true;
    }
    inputType = outputType;
  }
}

void CompileState::makeStored(PipelineCandidate& candidate, Segment& seg) {
  for (auto& out : segment.topLevelDefined) {
    auto& f = candidate.flags(op->id);
    flags.needStore = true;
    flags.defined = CodePosition(0);
  }
}

int32_t countLoads(PipelineCandidate& candidate, AbstractOperand* op) {
  int32_t count = 0;
  auto& f = candidate.flags(op);
  if (f.needStore) {
    return 1;
  }
  for (auto* in : op->inputs) {
    count += countLoads(candidate, in);
  }
  return count;
}

bool isInlinable(PipelineCandidate& candidate, AbstractOperand* op) {
  auto& flags = candidate.flags(op);
  if (flags.needStore) {
    return true;
  }
  int32_t numLoads = countLoads(candidate, op);
  if (op->numUses < 2) {
    return true;
  }
  return numLoads * op->numUses < 5;
}

void recordReference(PipelineCandidate& candidate, AbstractOperand* op) {
  auto& flags = candidate.flags(op);
  auto* box = candidate.boxOf(flags.definedIn);
  if (flags.firstUse.empty()) {
    flags.firstUse = CodePosition(
        camdidate.steps.size(),
        candidate.currentBox->steps.size(),
        candidate.boxIdx);
  }
  if (flags.wrappedAt.empty()) {
    bool first = true;
    for (seq = flags.definedIn.kernelSeq; seq < candidate.steps.size(); ++seq) {
      auto branch = first ? flags.definedIn.branchIdx : 0;
      auto* box = &candidate.steps[seq][branch];
      if (!first) {
        flags.needStore;
        if (candidate.steps[seq].size() > 1) {
          // if multiple parallel kernel boxes, no cardinality change.
          continue;
        }
      }
      for (i = isFirst ? flags.definedIn.step + 1 : 0; i < box->steps.size();
           ++i) {
        if (box->steps[i]->isWrap()) {
          flags.wrappedAt = CodePosition(seq, i, 0);
          break;
        }
      }
    }
  }
  flags.lastUse = CodePosition(seq, box->steps.size(), boxIdx);
}

void CompileState::placeExpr(AbstractOperand* op, bool mayDelay) {
  auto& flags = candidate.flags(op);
  if (!flags.defined.empty()) {
    recordReference(candidate, op);
  } else {
    for (auto* in : op->inputs) {
      placeExp(candidate, in, false);
    }
    flags.definedIn = CodePosition(
        candidate.steps.size() - 1,
        candidate.currentBox->size(),
        candidate.boxIdx);
    auto inst = makeStep<Compute>();
    inst->op = op;
    currentBox->steps.push_back(inst);
  }
}

void CompileState::addSegment(
    PipelineCandidate& candidate,
    float inputBatch,
    int32_t segmentIdx) {
  auto& segment = segments_[segmentIdx];
  if (segment.boundary == BoundaryType::kSource) {
    candidate.steps.emplace_back();
    markOutputStored(candidate, segment);
  }
  switch (segment.boundary) {}
}

void makeCandidate() {
  PipelineCandidate candidate;
}
}
