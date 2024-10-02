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

#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/exec/FilterProject.h"
#include "velox/experimental/wave/exec/Aggregation.h"
#include "velox/experimental/wave/exec/Project.h"
#include "velox/experimental/wave/exec/TableScan.h"
#include "velox/experimental/wave/exec/Values.h"
#include "velox/experimental/wave/exec/WaveDriver.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"

DEFINE_int64(velox_wave_arena_unit_size, 1 << 30, "Per Driver GPU memory size");

namespace facebook::velox::wave {

using exec::Expr;





void CompileState::addFilter(const Expr& expr, const RowTypePtr& outputType) {
  int32_t numPrograms = allPrograms_.size();
  auto condition = addExpr(expr);
  auto indices = newOperand(INTEGER(), "indices");
  indices->notNull = true;
  auto program = programOf(condition);
  program->addLabel(expr.toString(true));
  program->markOutput(indices->id);
  program->add(std::make_unique<AbstractFilter>(condition, indices));
  auto wrapUnique = std::make_unique<AbstractWrap>(indices, wrapCounter_++);
  auto wrap = wrapUnique.get();
  program->add(std::move(wrapUnique));
  auto levels = makeLevels(numPrograms);
  operators_.push_back(
      std::make_unique<Project>(*this, outputType, levels, wrap));
}

void CompileState::addFilterProject(
    exec::Operator* op,
    RowTypePtr& outputType,
    int32_t& nodeIndex) {
  auto filterProject = reinterpret_cast<exec::FilterProject*>(op);
  outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  auto data = filterProject->exprsAndProjection();
  auto& identityProjections = filterProject->identityProjections();
  int32_t firstProjection = 0;
  if (data.hasFilter) {
    addFilter(*data.exprs->exprs()[0], outputType);
    firstProjection = 1;
    ++nodeIndex;
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  }
  int32_t numPrograms = allPrograms_.size();
  auto operands =
      addExprSet(*data.exprs, firstProjection, data.exprs->exprs().size());
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
  auto levels = makeLevels(numPrograms);
  operators_.push_back(std::make_unique<Project>(*this, outputType, levels));
  for (auto& [value, operand] : pairs) {
    operators_.back()->defined(value, operand);
  }
}

bool CompileState::reserveMemory() {
  if (arena_) {
    return true;
  }
  auto* allocator = getAllocator(getDevice());
  arena_ =
      std::make_unique<GpuArena>(FLAGS_velox_wave_arena_unit_size, allocator);
  return true;
}

const std::shared_ptr<aggregation::AggregateFunctionRegistry>&
CompileState::aggregateFunctionRegistry() {
  if (!aggregateFunctionRegistry_) {
    aggregateFunctionRegistry_ =
        std::make_shared<aggregation::AggregateFunctionRegistry>(
            getAllocator(getDevice()));
    Stream stream;
    aggregateFunctionRegistry_->addAllBuiltInFunctions(stream);
    stream.wait();
  }
  return aggregateFunctionRegistry_;
}

void CompileState::setAggregateFromPlan(
    const core::AggregationNode::Aggregate& planAggregate,
    AbstractAggInstruction& agg) {
  agg.op = AggregateOp::kSum;
}

void CompileState::makeAggregateLayout(AbstractAggregation& aggregate) {
  // First key nulls, then key wirds. Then accumulator nulls, then accumulators.
  int32_t numKeys = aggregate.keys.size();
  int32_t startOffset = bits::roundUp(numKeys + 4, 8) + 8 * numKeys;
  int32_t accNullOffset = startOffset;
  auto numAggs = aggregate.aggregates.size();
  int32_t accOffset = accNullOffset + bits::roundUp(numAggs, 8);
  for (auto i = 0; i < numAggs; ++i) {
    auto& agg = aggregate.aggregates[i];
    agg.nullOffset = accNullOffset + i;
    agg.accumulatorOffset = accOffset + i * sizeof(int64_t);
  }
}

void CompileState::makeAggregateAccumulate(const core::AggregationNode* node) {
  auto* state = newState(StateKind::kGroupBy, node->id(), "");
  std::vector<AbstractOperand*> keys;
  folly::F14FastSet<AbstractOperand*> uniqueArgs;
  folly::F14FastSet<Program*> programs;
  std::vector<AbstractOperand*> allArgs;
  std::vector<AbstractAggInstruction> aggregates;
  int numPrograms = allPrograms_.size();
  for (auto& key : node->groupingKeys()) {
    auto arg = findCurrentValue(key);
    allArgs.push_back(arg);
    keys.push_back(arg);
    if (auto source = definedIn_[arg]) {
      programs.insert(source);
    }
  }
  auto numKeys = node->groupingKeys().size();
  for (auto& planAggregate : node->aggregates()) {
    aggregates.emplace_back();
    std::vector<PhysicalType> argTypes;
    auto& aggregate = aggregates.back();
    setAggregateFromPlan(planAggregate, aggregate);
    auto i = numKeys + aggregates.size() - 1;
    aggregate.result = newOperand(
        node->outputType()->childAt(i), node->outputType()->nameOf(i));
    auto subfield = toSubfield(node->outputType()->nameOf(i));
    definedBy_[Value(subfield)] = aggregate.result;
    for (auto& arg : planAggregate.call->inputs()) {
      argTypes.push_back(fromCpuType(*arg->type()));
      auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(arg);
      auto op = findCurrentValue(field);
      aggregate.args.push_back(op);
      bool isNew = uniqueArgs.insert(op).second;
      if (isNew) {
        allArgs.push_back(op);
        if (auto source = definedIn_[op]) {
          programs.insert(source);
        }
      }
    }
#if 0
    auto func =
        functionRegistry_->getFunction(aggregate.call->name(), argTypes);
    VELOX_CHECK_NOT_NULL(func);
#endif
  }
  auto instruction = std::make_unique<AbstractAggregation>(
      nthContinuable_++,
      std::move(keys),
      std::move(aggregates),
      state,
      node->outputType());
  if (!instruction->keys.empty()) {
    instruction->maxReadStreams = FLAGS_max_streams_per_driver * 10;
  }

  makeAggregateLayout(*instruction);
  std::vector<Program*> sourceList;
  if (programs.empty()) {
    sourceList.push_back(newProgram());
  } else if (programs.size() == 1) {
    sourceList.push_back(*programs.begin());
  } else {
    for (auto& s : programs) {
      sourceList.push_back(s);
    }
  }
  instruction->reserveState(instructionStatus_);
  allStatuses_.push_back(instruction->mutableInstructionStatus());
  auto aggInstruction = instruction.get();
  addInstruction(std::move(instruction), nullptr, sourceList);
  if (allPrograms_.size() > numPrograms) {
    makeProject(numPrograms, node->outputType());
  }
  numPrograms = allPrograms_.size();
  auto reader = newProgram();
  reader->add(std::make_unique<AbstractReadAggregation>(
      nthContinuable_++, aggInstruction));

  makeProject(numPrograms, node->outputType());
  auto project = reinterpret_cast<Project*>(operators_.back().get());
  for (auto i = 0; i < node->groupingKeys().size(); ++i) {
    std::string name = aggInstruction->keys[i]->label;
    operators_.back()->defined(
        Value(toSubfield(name)), aggInstruction->keys[i]);
    definedIn_[aggInstruction->keys[i]] = reader;
  }
  for (auto i = 0; i < aggInstruction->aggregates.size(); ++i) {
    std::string name = aggInstruction->aggregates[i].result->label;
    operators_.back()->defined(
        Value(toSubfield(name)), aggInstruction->aggregates[i].result);
    definedIn_[aggInstruction->aggregates[i].result] = reader;
    // project->definesSubfield(*this,
    // aggInstruction->aggregates[i].result->type, name, false);
  }
}

void CompileState::makeProject(int firstProgram, RowTypePtr outputType) {
  auto levels = makeLevels(firstProgram);
  operators_.push_back(
      std::make_unique<Project>(*this, outputType, std::move(levels)));
}

bool CompileState::addOperator(
    exec::Operator* op,
    int32_t& nodeIndex,
    RowTypePtr& outputType) {
  auto& name = op->operatorType();
  if (name == "Values") {
    if (!reserveMemory()) {
      return false;
    }
    operators_.push_back(std::make_unique<Values>(
        *this,
        *reinterpret_cast<const core::ValuesNode*>(
            driverFactory_.planNodes[nodeIndex].get())));
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();
  } else if (name == "FilterProject") {
    if (!reserveMemory()) {
      return false;
    }
    addFilterProject(op, outputType, nodeIndex);
  } else if (name == "Aggregation") {
    if (!reserveMemory()) {
      return false;
    }
    auto* node = dynamic_cast<const core::AggregationNode*>(
        driverFactory_.planNodes[nodeIndex].get());
    VELOX_CHECK_NOT_NULL(node);
    makeAggregateAccumulate(node);
#if 0
    operators_.push_back(std::make_unique<Aggregation>(
        *this, *node, aggregateFunctionRegistry()));
#endif
    outputType = node->outputType();
  } else if (name == "TableScan") {
    if (!reserveMemory()) {
      return false;
    }
    auto scan = reinterpret_cast<const core::TableScanNode*>(
        driverFactory_.planNodes[nodeIndex].get());
    outputType = driverFactory_.planNodes[nodeIndex]->outputType();

    operators_.push_back(
        std::make_unique<TableScan>(*this, operators_.size(), *scan));
    outputType = scan->outputType();
  } else {
    return false;
  }
  return true;
}

bool isProjectedThrough(
    const std::vector<exec::IdentityProjection>& projectedThrough,
    int32_t i,
    int32_t& inputChannel) {
  for (auto& projection : projectedThrough) {
    if (projection.outputChannel == i) {
      inputChannel = projection.inputChannel;
      return true;
    }
  }
  return false;
}

bool CompileState::compile() {
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
    auto& identity = operators[operatorIndex]->identityProjections();
    // The columns that are projected through are renamed. They may also get an
    // indirection after the new operator is placed.
    std::vector<std::pair<AbstractOperand*, int32_t>> identityProjected;
    for (auto& projection : identity) {
      identityProjected.push_back(std::make_pair(
          findCurrentValue(
              Value(toSubfield(inputType->nameOf(projection.inputChannel)))),
          projection.outputChannel));
    }
    if (!addOperator(operators[operatorIndex], nodeIndex, outputType)) {
      break;
    }
    ++nodeIndex;
    for (auto newIndex = previousNumOperators; newIndex < operators_.size();
         ++newIndex) {
      if (operators_[newIndex]->isSink()) {
        // No output operands.
        continue;
      }
      for (auto i = 0; i < outputType->size(); ++i) {
        auto& name = outputType->nameOf(i);
        Value value = Value(toSubfield(name));
        int32_t inputChannel;
        if (isProjectedThrough(identity, i, inputChannel)) {
          continue;
        }
        auto operand = operators_[newIndex]->defines(value);
        if (!operand &&
            (operators_[newIndex]->isSource() ||
             !operators_[newIndex]->isStreaming())) {
          operand = operators_[newIndex]->definesSubfield(
              *this, outputType->childAt(i), name, newIndex == 0);
        }
        if (operand) {
          operators_[newIndex]->addOutputId(operand->id);
          definedBy_[value] = operand;
          operandOperatorIndex_[operand] = operators_.size() - 1;
        }
      }
    }
    for (auto& [op, channel] : identityProjected) {
      Value value(toSubfield(outputType->nameOf(channel)));
      auto newOp = addIdentityProjections(op);
      projectedTo_[value] = newOp;
    }
    inputType = outputType;
  }
  if (operators_.empty()) {
    return false;
  }
  std::vector<OperandId> resultOrder;
  for (auto i = 0; i < outputType->size(); ++i) {
    auto operand = findCurrentValue(Value(toSubfield(outputType->nameOf(i))));
    auto source = programOf(operand, false);
    // Operands produced by programs, when projected out of Wave, must
    // be marked as output of their respective programs. Some
    // operands, e.g. table scan results are not from programs.
    if (source) {
      source->markOutput(operand->id);
    }
    resultOrder.push_back(operand->id);
  }
  for (auto& op : operators_) {
    op->finalize(*this);
  }
  instructionStatus_.gridStateSize = instructionStatus_.gridState;
  for (auto* status : allStatuses_) {
    status->gridStateSize = instructionStatus_.gridState;
  }
  auto waveOpUnique = std::make_unique<WaveDriver>(
      driver_.driverCtx(),
      outputType,
      operators[first]->planNodeId(),
      operators[first]->operatorId(),
      std::move(arena_),
      std::move(operators_),
      std::move(resultOrder),
      std::move(subfields_),
      std::move(operands_),
      std::move(operatorStates_),
      instructionStatus_);
  auto waveOp = waveOpUnique.get();
  waveOp->initialize();
  std::vector<std::unique_ptr<exec::Operator>> added;
  added.push_back(std::move(waveOpUnique));
  auto replaced = driverFactory_.replaceOperators(
      driver_, first, operatorIndex, std::move(added));
  waveOp->setReplaced(std::move(replaced));
  return true;
}

  
  
}
