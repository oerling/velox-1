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

using common::Subfield;
using exec::Expr;

AbstractOperand* markUse(AbstractOperand* op) {
  ++op->numUses;
  return op;
}

AbstractOperand* Scope::findValue(const Value& value) {
  auto it = operandMap.find(value);
  if (it == operandMap.end()) {
    if (parent) {
      return parent->findValue(value);
    }
    return nullptr;
  }
  return it->second;
}

AbstractOperand* CompileState::fieldToOperand(Subfield& field, Scope* scope) {
  auto* op = scope->findValue(Value(&field));
  if (op) {
    return markUse(op);
  }
  auto* name =
      &reinterpret_cast<common::Subfield::NestedField*>(field.path()[0].get())
           ->name();
  for (int32_t i = renames_.size() - 1; i >= 0; --i) {
    auto it = renames_[i].find(*name);
    if (it == renames_[i].end()) {
      VELOX_FAIL("Can't resolve {}", *name);
    }
    name = &it->second;
    auto* temp = toSubfield(*name);
    auto* def = topScopes_[i].findValue(Value(temp));
    if (def) {
      return markUse(def);
    }
  }
  VELOX_FAIL("Unresolved {}", *name);
}

AbstractOperand* CompileState::fieldToOperand(
    const core::FieldAccessTypedExpr& field,
    Scope* scope) {
  Subfield* subfield = toSubfield(field.name());
  return fieldToOperand(*subfield, scope);
}

AbstractOperand* CompileState::switchOperand(
    const exec::SwitchExpr& switchExpr,
    Scope* scope) {
  auto& inputs = switchExpr.inputs();
  std::vector<AbstractOperand*> opInputs;
  Scope clauseScope(scope);
  for (auto i = 0; i < inputs.size(); i += 2) {
    opInputs.push_back(exprToOperand(*inputs[i], &clauseScope));
    if (i + 1 < inputs.size()) {
      opInputs.push_back(exprToOperand(*inputs[i + 1], &clauseScope));
    }
    clauseScope.operandMap.clear();
  }
  auto result = newOperand(switchExpr.type(), "r");
  result->inputs = std::move(opInputs);
  scope->operandMap[Value(&switchExpr)] = result;
  return result;
}

bool functionRetriable(const Expr& expr) {
  if (expr.name() == "CONCAT") {
    return true;
  }
  return false;
}

int32_t functionCost(const Expr& expr) {
  // Arithmetic
  return 1;
}

AbstractOperand* CompileState::exprToOperand(const Expr& expr, Scope* scope) {
  auto value = toValue(expr);
  auto op = scope->findValue(value);
  if (op) {
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
      return switchOperand(*switchExpr, scope);
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
    const core::PlanNode* node,
    RowTypePtr outputType) {
  segments_.emplace_back();
  auto& last = segments_.back();
  last.ordinal = segments_.size() - 1;
  last.boundary = boundary;
  last.planNode = node;
  if (outputType) {
    int32_t size = outputType->size();
    for (auto i = 0; i < size; ++i) {
      auto* subfield = toSubfield(outputType->nameOf(i));
      Value value(subfield);
      auto* op = newOperand(outputType->childAt(i), outputType->nameOf(i));
      op->definingSegment = last.ordinal;
      op->sourceNullable = boundary == BoundaryType::kSource;
      op->needsStore = boundary == BoundaryType::kSource;
      topScope_.operandMap[value] = op;
    }
  }
  return last;
}

void CompileState::tryFilter(const Expr& expr, const RowTypePtr& outputType) {
  auto& last = addSegment(BoundaryType::kExpr, nullptr, nullptr);
  last.topLevelDefined.push_back(exprToOperand(expr, &topScope_));
}

std::vector<AbstractOperand*> CompileState::tryExprSet(
    const exec::ExprSet& exprSet,
    int32_t begin,
    int32_t end,
    const RowTypePtr& outputType) {
  auto& exprs = exprSet.exprs();
  auto& result = segments_.back().topLevelDefined;
  for (auto i = begin; i < end; ++i) {
    result.push_back(exprToOperand(*exprs[i], &topScope_));
    auto* subfield = toSubfield(outputType->nameOf(i - begin));
    segments_.back().projectedName.push_back(subfield);
  }
  return result;
}

std::unordered_map<std::string, std::string> makeRenames(
    const std::vector<exec::IdentityProjection>& identities,
    const RowTypePtr inputType,
    const RowTypePtr& outputType) {
  std::unordered_map<std::string, std::string> map;
  for (auto p : identities) {
    map[outputType->nameOf(p.outputChannel)] =
        inputType->nameOf(p.inputChannel);
  }
  return map;
}

void CompileState::tryFilterProject(
    exec::Operator* op,
    RowTypePtr& outputType,
    int32_t& nodeIndex) {
  auto filterProject = reinterpret_cast<exec::FilterProject*>(op);
  outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  auto data = filterProject->exprsAndProjection();
  auto& identityProjections = filterProject->identityProjections();
  auto inputType = outputType;
  int32_t firstProjection = 0;
  if (data.hasFilter) {
    tryFilter(*data.exprs->exprs()[0], outputType);
    auto filterOp = segments_.back().topLevelDefined[0];
    addSegment(BoundaryType::kFilter, nullptr, outputType);
    auto filterStep = makeStep<Filter>();
    filterStep->flag = filterOp;
    segments_.back().steps.push_back(filterStep);
    firstProjection = 1;
    ++nodeIndex;
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  } else {
    addSegment(BoundaryType::kExpr, nullptr, nullptr);
  }

  auto operands = tryExprSet(
      *data.exprs, firstProjection, data.exprs->exprs().size(), outputType);
  if (!identityProjections.empty()) {
    renames_.push_back(makeRenames(identityProjections, inputType, outputType));
    topScopes_.push_back(std::move(topScope_));
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
        BoundaryType::kSource,
        driverFactory_.planNodes[nodeIndex].get(),
        outputType);
  } else if (name == "FilterProject") {
    tryFilterProject(op, outputType, nodeIndex);
  } else if (name == "Aggregation") {
    auto* node = dynamic_cast<const core::AggregationNode*>(
        driverFactory_.planNodes[nodeIndex].get());
    VELOX_CHECK_NOT_NULL(node);
    addSegment(BoundaryType::kAggregation, node, nullptr);
    auto step = makeStep<AggregateProbe>();
    auto* state = newState(StateKind::kGroupBy, node->id(), "");
    auto aggregationStep = node->step();
    step->state = state;
    step->rows = newOperand(BIGINT(), "rows");
    for (auto& key : node->groupingKeys()) {
      step->keys.push_back(fieldToOperand(*key, &topScope_));
    }
    std::vector<AggregateUpdate*> allUpdates;
    for (auto& agg : node->aggregates()) {
      std::vector<AbstractOperand*> args;
      for (auto& expr : agg.call->inputs()) {
        args.push_back(fieldToOperand(
            *std::dynamic_pointer_cast<core::FieldAccessTypedExpr>(expr),
            &topScope_));
      }

      auto* func = makeStep<AggregateUpdate>();
      func->step = aggregationStep;
      func->name = agg.call->name();
      func->rows = step->rows;
      func->args = std::move(args);
      func->result = fieldToOperand(
          *toField(output->nameOf(i + step->keys.size())), &topScope_);
      allUpdates.push_back(func);
    }
    segments_.back().steps.push_back(step);
    outputType = node->outputType();
    addSegment(BoundaryType::kSource, node, outputType);
    auto read = makeStep<ReadAggregation>();
    read->state = state;
    for (auto i = 0; i < agg->keys.size(); ++i) {
      read->columns.push_back(
          fieldToOperand(*toSubfield(outputType->nameOf(i)), &topScope_));
    }
    read->updates = std::move(allUpdates);
    segments_.back().steps.push_back(read);
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
  }
  for (auto i = 0; i < outputType->size(); ++i) {
    auto* result =
        fieldToOperand(*toSubfield(outputType->nameOf(i)), &topScope_);
    // Returned to host, must be in memory.
    result->needsStore = true;
  }
  return true;
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
        candidate.steps.size(),
        candidate.currentBox->steps.size(),
        candidate.boxIdx);
  }
  if (flags.wrappedAt.empty()) {
    bool first = true;
    for (auto seq = flags.definedIn.kernelSeq; seq < candidate.steps.size();
         ++seq) {
      auto branch = first ? flags.definedIn.branchIdx : 0;
      auto* box = &candidate.steps[seq][branch];
      if (!first) {
        flags.needStore = true;
        if (candidate.steps[seq].size() > 1) {
          // if multiple parallel kernel boxes, no cardinality change.
          continue;
        }
      }
      for (auto i = first ? flags.definedIn.step + 1 : 0; i < box->steps.size();
           ++i) {
        if (box->steps[i]->isWrap()) {
          flags.wrappedAt = CodePosition(seq, i, 0);
          break;
        }
      }
      first = false;
    }
  }
  flags.lastUse = CodePosition(
      candidate.steps.size() - 1, box->steps.size(), candidate.boxIdx);
}

void CompileState::placeExpr(
    PipelineCandidate& candidate,
    AbstractOperand* op,
    bool mayDelay) {
  auto& flags = candidate.flags(op);
  if (!flags.definedIn.empty()) {
    recordReference(candidate, op);
  } else {
    for (auto* in : op->inputs) {
      placeExpr(candidate, in, false);
    }
    flags.definedIn = CodePosition(
        candidate.steps.size() - 1,
        candidate.currentBox->steps.size(),
        candidate.boxIdx);
    auto inst = makeStep<Compute>();
    inst->operand = op;
    candidate.currentBox->steps.push_back(inst);
  }
}

void CompileState::markOutputStored(
    PipelineCandidate& candidate,
    Segment& segment) {
  auto& RowTypePtr type = segment.outputType;
  for (auto i = 0; i < type->size(); ++i) {
    auto* op = fieldToOperand(*toSubfield(type->nameOf(i)), &topScope_);
    candidate.flags(op).needStore = true;
  }
}

void newKernel(PipelineCandidate& candidate) {
  candidate.steps.emplace_back();
  candidate.steps.back().emplace_back();
  candidate.currentBox = &candidate.steps.back()[0];
  candidate.boxIdx = 0;
}

void CompileState::recordCandidate(PipelineCandidate& candidate) {
  candidates_.push_back(std::move(candidate));
}

void CompileState::planSegment(
    PipelineCandidate& candidate,
    float inputBatch,
    int32_t segmentIdx) {
  auto& segment = segments_[segmentIdx];
  switch (segment.boundary) {
    case BoundaryType::kSource: {
      if (!candidate.steps.size() > 1 || !candidate.currentBox->steps.empty()) {
        // A pipeline barrier.
        recordCandidate(candidate);
        return;
      }
      auto* node = segment.planNode;
      if (auto* scan = dynamic_cast<const core::TableScanNode>(node)) {
        auto step = makeStep<TableScanStep>();
        step->node = scan;
        candidate.currentBox->steps.push_back(step);
        newKernel(candidate);
      } else if (
          auto* read = dynamic_cast<const core::AggregationNode*>(node)) {
        auto* step = segment.steps[0];
        candidate.currentBox->steps.push_back(step);
      }
      markOutputStored(candidate, segment);
      break;
    }
    case BoundaryType::kExpr: {
      for (auto i = 0; i < segment.topLevelDefined.size(); ++i) {
        placeExpr(candidate, segment.topLevelDefined[i], true);
      }
      break;
      break;
    }
    case BoundaryType::kFilter: {
      placeExpr(candidate, segment.topLevelDefined[0], false);
      auto filter = reinterpret_cast<Filter*>(segment.steps[0]);
      candidate.currentBox->steps.push_back(filter);
      break;
    }
    case BoundaryType::kAggregation: {
      if (candidate.steps.back().size() > 1) {
        newKernel(candidate);
      }
      candidate->currentBox->steps.push_back(segment.steps[]);
      break;
    }
  }
  if (segmentIdx == segments_.size() - 1 {
        recordCandidate(candidate);
        return;
      })
    ;
  planSegment(candidate, inputBatch, segmentIdx + 1);
}

void CompileState::planPipelines() {
  int32_t startIdx = 0;
  for (;;) {
    PipelineCandidate candidate;
    newKernel(candidate);
    planSegment(candidate, 100000, startIdx);
    pickBest();
    bool found = false;
    for (auto i = startIdx + 1; i < segments_.size(); ++i) {
      if (segments_[i].boundary == BoundaryType::kSource) {
        startIdx = i;
        found = true;
        break;
      }
    }
    if (!found) {
      break;
    }
  }
  makeDriver();
}

ProgramKey makeKey(PipelineCandidate& candidate, int32_t kernelIdx) {
  std::vector<AbstractParameter*> input;
  std::vector<AbstractParameter*> output;
  std::stringstream out;
  auto& level = candidate.steps[stepIdx];
  folly::F14FastMap<int32_t, int32_t> renamed;
  for (auto programIdx = 0; programIdx < level.size(); ++programIdx) {
    auto& box = level[programIdx];
    for (auto stepIdx = 0; stepIdx < box.steps.size(); ++stepIdx) {
      auto renamedId = [&](AbstractOperand* op) -> int32_t {
        auto it = renamed.find(op->id);
        if (it == renamed.end()) {
          return renamed[op->id] = renamed.size();
        }
        return it->second;
      };

      auto markOutput = [&](AbstractOperand* op) {
        auto& flags = candidate.flags(op);
        if (flags.lastUse.kernelSeq > kernelIdx) {
          out << fmt::format("<O {} {}>=", output.size(), op->type->toString());
          output.push_back(op);
        } else {
          out << fmt::format("<T {} {}>=", renamedId(op), op->type->toString());
        }
      };

      auto markInput = [&](AbstractOperand* op) {
        auto& flags = candidate.flags(op);
        if (flags.definedIn.kernelSeq < kernelIdx) {
          out << fmt::format("<I {} {}>", input.size(), op->type->toString());
          input.push_back(op);
        } else {
          out << fmt::format("<T {} {}>", renamedId(op), op->type->toString());
        }
      };

      switch (step->kind()) {
        case StepKind::kOperand: {
          auto& compute = step.as<Compute>();
          markOutput(op);
          out << op->expr->name();
          out << "(";
          for (auto* in : op->inputs) {
            markInput(in);
          }
          out << ")\n";
          break;
        }
        case StepKind::kFilter: {
          auto& filter = step->as<Filter>();
          out << "filter(";
          markInput(filter.flag);
          out << ")\n";
          break;
        }
        case StepKind::kAggregateProbe: {
          auto& agg = step.as<AggregateProbe>();
          out << "Aggregate(";
          for (auto& k : agg.keys) {
            markInput(k);
          }
          out << ") =";
          markOutput(agg.rows);
          out << "\n";
          break;
        }
        case StepKind::kAggregateUpdate: {
          auto& func = step->as<AggregateUpdate>();
          out << "update " << step->name << "(";
          markInput(func->rows);
          for (auto& op : func->args) {
            markInput(op);
          }
          out << ")\n";
          break;
        }
        case StepKind::kReadAggregation: {
          auto& read = step.as<ReadAggregation>();
          out << "readAgg " << static_cast<int32_t>(read->step) << "(";
          for (auto* key : read->columns) {
            markOutput(key);
          }
          for (auto i = 0; i < read.updates.size(); ++i) {
            out << fmt::format("A:{} ", i);
            markOutput(read.updates[i]->result);
          }
          out << ")\n";
          break;
        }
      }
    }
  }
  return ProgramKey{
      .text = out.str(),
      .input = std::move(input),
      .output = std::move(output)};
}

void CompileState::makeDriver() {
  makeSegments();
  planPipelines();
}

} // namespace facebook::velox::wave
