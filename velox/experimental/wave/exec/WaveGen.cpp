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

#include "velox/experimental/wave/exec/Project.h"
#include "velox/experimental/wave/exec/TableScan.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/Values.h"

namespace facebook::velox::wave {

const std::string typeName(const Type& type) {
  switch (type.kind()) {
    case TypeKind::BIGINT:
      return "int64_t ";
    default:
      VELOX_UNSUPPORTED("No gen for type {}", type.toString());
  }
}

bool KernelStep::references(AbstractOperand* op) {
  bool found = false;
  visitReferences([&](AbstractOperand* referenced) {
    if (found) {
      return;
    }
    if (op == referenced) {
      found = true;
    }
  });
  return found;
}

int32_t CompileState::ordinal(const AbstractOperand& op) {
  auto& params = selectedPipelines_[pipelineIdx_].levelParams[kernelSeq_];
  if (params.input.contains(op.id)) {
    return params.input.ordinal(op.id);
  }
  if (params.local.contains(op.id)) {
    return params.input.size() + params.local.ordinal(op.id);
  }
  if (params.output.contains(op.id)) {
    return params.input.size() + params.local.size() +
        params.output.ordinal(op.id);
  }
  VELOX_UNREACHABLE();
}

int32_t CompileState::declareVariable(const AbstractOperand& op) {
  auto ord = ordinal(op);
  generated_ << fmt::format("{} r{};", typeName(*op.type), ord);
  return ord;
}

void EndNullCheck::generateMain(CompileState& state) {
  auto ord = state.ordinal(*result);
  state.generated() << fmt::format("goto skip{};\n", label)
                    << fmt::format("end{}: \n", label);
  auto flags = state.flags(*result);
  fmt::format("setRegisterNull(nulls{}, {});\n", ord / 32, ord & 31, true);
  if (flags.needStore) {
    state.generated() << fmt::format(
        "setNull(operands, {}, blockBase, true);n", ord);
  }
  state.generated() << fmt::format("skip{}: ;\n", label);
  state.setInsideNullPropagating(false);
}

bool CompileState::hasMoreReferences(AbstractOperand* op, int32_t pc) {
  for (auto i = pc; i < currentBox_->steps.size(); ++i) {
    if (!currentBox_->steps[i]->preservesRegisters()) {
      return false;
    }
    if (currentBox_->steps[i]->references(op)) {
      return true;
    }
  }
  return false;
}

void CompileState::clearInRegister() {
  for (auto& op : operands_) {
    op->inRegister = false;
    op->registerNullBit = AbstractOperand::kNoNullBit;
  }
}

void NullCheck::generateMain(CompileState& state) {
  std::vector<AbstractOperand*> lastUse;
  bool isFirst = true;
  state.setInsideNullPropagating(true);

  for (auto* op : operands) {
    if (!op->inRegister && state.hasMoreReferences(op, endIdx)) {
      if (isFirst) {
        state.generated() << fmt::format("bool anyNull{} = false;\n", label);
        isFirst = false;
      }
      auto& flags = state.flags(*op);
      bool mayWrap = flags.wrappedAt.empty() ||
          flags.wrappedAt.isBefore(state.currentPosition());
      auto ordinal = state.declareVariable(*op);
      state.generated() << fmt::format(
          "anyNull{} |= setRegisterNull(nulls{}, {}, valueOrNull<{}>(operands, {}, blockBase, r{}));\n",
          label,
          ordinal / 32,
          ordinal & 31,
          mayWrap ? "true" : "false",
          ordinal);
    } else {
      lastUse.push_back(op);
    }
  }
  if (!isFirst) {
    state.generated() << fmt::format(
        "if (anyNull{}) {{ goto end{};}}\n", label, label);
  }
  for (auto* op : lastUse) {
    if (op->inRegister) {
      auto ord = state.ordinal(*op);
      state.generated() << fmt::format(
          "if (isRegisterNull(nulls{}, {})) {{goto end{};}}\n",
          ord / 32,
          ord & 31,
          label);
      continue;
    }
    auto& flags = this->flags(op);

    bool mayWrap = flags.wrappedAt.empty() ||
        flags.wrappedAt.isBefore(state.currentPosition());
    auto ord = state.declareVariable(*op);
    state.generated() << fmt::format(
        "if (!valueOrNull<{}>(operands, {}, blockBase, r{})) {goto end{};}\n",
        mayWrap ? "true" : "false",
        ord,
        ord,
        label);
  }
}

void CompileState::generateOperand(const AbstractOperand& op) {
  if (op.inRegister && insideNullPropagating_) {
    generated_ << fmt::format(" r{} ", ordinal(op));
    return;
  }
  if (op.notNull || insideNullPropagating_) {
    auto& flags = this->flags(op);
    bool mayWrap =
        flags.wrappedAt.empty() || flags.wrappedAt.isBefore(currentPosition());
    generated_ << fmt::format(
        "nonNullOperand<{}, {}>(operands, {}, blockBase)",
        typeName(*op.type),
        mayWrap,
        ordinal(op));
  }
}

void Compute::generateMain(CompileState& state) {
  VELOX_CHECK_NOT_NULL(operand->expr);
  auto& flags = state.flags(*operand);
  auto ord = state.declareVariable(*operand);
  state.generated() << fmt::format("r{} = {}(", ord, operand->expr->name());
  for (auto i = 0; i < operand->inputs.size(); ++i) {
    state.generateOperand(*operand->inputs[i]);
    if (i < operand->inputs.size() - 1) {
      state.generated() << ", ";
    }
  }
  state.generated() << ");\n";
  operand->inRegister = true;
  if (flags.needStore) {
    state.generated() << fmt::format(
        "flatValue(operands, {}, blockBase) = r{};\n", ord, ord);
  }
}

std::string CompileState::generateIsTrue(const AbstractOperand& op) {
  auto ord = ordinal(op);
  if (op.inRegister) {
    if (op.notNull) {
      generated_ << fmt::format("bool flag{} = r{}", ord, ord);
    } else {
      generated_ << fmt::format(
          "bool flag{} = {} && !isRegisterNull(nulls{}, {});\n",
          ord,
          ord / 32,
          ord & 31);
    }
  } else {
    auto& flags = this->flags(op);
    bool mayWrap =
        flags.wrappedAt.empty() || flags.wrappedAt.isBefore(currentPosition());
    if (op.notNull || insideNullPropagating_) {
      generated_ << fmt::format(
          "bool flag{} = nonNullOperand<bool, {}>(operands, {}, blockBase)",
          mayWrap,
          ord);
    } else {
      generated_ << fmt::format("bool flag{};\n", ord);
      generated_ << fmt::format(
          "if (!valueOrNull<{}, bool>(operands, {}, blockBase, flags{})) {{ flags{} = false; }};\n",
          mayWrap ? "true" : "false",
          ord,
          ord,
          ord);
    }
  }
  return fmt::format("flag{}", ord);
}

int32_t CompileState::nextWrapId() {
  return ++wrapId_;
}

int32_t CompileState::wrapLiteral(int32_t id) {
  // We take one Operand of each group of Operands that shares a wrappedAt such
  // that the Operand's lifetime crosses the filter.
  CodePosition filter(kernelSeq_, stepIdx_, 0);
  std::unordered_set<CodePosition> wraps;
  std::vector<OperandIndex> ops;
  for (auto& op : operands_) {
    auto& flags = currentCandidate_->flags(op.get());
    if (filter.isBefore(flags.lastUse) && flags.definedIn.isBefore(filter)) {
      auto& wrappedAt = flags.wrappedAt;
      if (wraps.count(wrappedAt)) {
        continue;
      }
      wraps.insert(wrappedAt);
      ops.push_back(op->id);
    }
  }
  generated_ << fmt::format("const OperandIndex wraps{}[] = {", id);
  for (auto i = 0; i < ops.size(); ++i) {
    generated_ << i;
    if (i < ops.size() - 1) {
      generated_ << ", ";
    }
  }
  generated_ << "};\n";
  return ops.size();
}

void Filter::generateMain(CompileState& state) {
  auto flagValue = state.generateIsTrue(*flag);
  state.generated() << fmt::format(
      "filterKernel({}, operands, {}, blockBase, shared, laneStatus);\n",
      flagValue,
      state.ordinal(*indices));
  auto id = state.nextWrapId();
  auto numWraps = state.wrapLiteral(id);
  state.generated() << fmt::format(
      "wrapKernel(wraps{}, {}, {}, operands, blockBase, shared);\n",
      id,
      numWraps,
      state.ordinal(*indices));
}

void AggregateProbe::generateMain(CompileState& state) {}

void AggregateUpdate::generateMain(CompileState& state) {}

void writeDebugFile(const KernelSpec& spec) {
  try {
    std::ofstream out(
        fmt::format("/tmp/{}", spec.filePath),
        std::ios_base::out | std::ios_base::trunc);
    out << spec.code;
    out.close();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error saving compiled file /tmp/" << spec.filePath << " "
               << e.what();
  }
}

ProgramKey CompileState::makeLevelText(
    int32_t pipelineIdx,
    int32_t kernelSeq,
    KernelSpec& spec) {
  std::lock_guard<std::mutex> l(generateMutex_);
  insideNullPropagating_ = false;
  currentCandidate_ = &selectedPipelines_[pipelineIdx];
  pipelineIdx_ = pipelineIdx;
  kernelSeq_ = kernelSeq;
  auto& level = selectedPipelines_[pipelineIdx_].steps[kernelSeq_];
  VELOX_CHECK_EQ(1, level.size(), "Only one program per level supported");
  std::stringstream head;
  auto kernelName = fmt::format("wavegen{}", ++kernelCounter_);
  std::vector<std::string> entryPoints = {kernelName};
  head << fmt::format(
      "#include \"velox/experimental/wave/exec/WaveCore.cuh\"\n"
      "void __global__ __launch_bounds__(1024) {}(KernelParams params) {{\n",
      kernelName);

  generated_ << "  GENERATED_PREAMBLE(0);\n";
  for (branchIdx_ = 0; branchIdx_ < level.size(); ++branchIdx_) {
    auto& box = level[branchIdx_];
    currentBox_ = &box;
    clearInRegister();
    bool anyRetry = false;
    for (stepIdx_ = 0; stepIdx_ < box.steps.size(); ++stepIdx_) {
      if (box.steps[stepIdx_]->hasContinue()) {
        if (!anyRetry) {
          anyRetry = true;
          generated_ << "if (shared->isContinue) {\n"
                     << "switch(entryPoint) {n";
        }
        generated_ << fmt::format(
            "case {}: goto continue{};\n", stepIdx_, stepIdx_);
      }
      if (anyRetry) {
        generated_ << "}}\n}\n";
      }
    }
    for (stepIdx_ = 0; stepIdx_ < box.steps.size(); ++stepIdx_) {
      // Generate the  code for first execution.
      auto step = box.steps[stepIdx_];
      if (step->hasContinue()) {
        generated_ << fmt::format("enter{}: \n", stepIdx_);
      }
      step->generateMain(*this);
    }
  }

  generated_ << " PROGRAM_EPILOGUE()\n}";
  auto& params = currentCandidate_->levelParams[kernelSeq_];
  int32_t numRegs =
      params.input.size() + params.local.size() + params.output.size();
  for (auto i = 0; i < numRegs; i += 32) {
    head << fmt::format(" uint32_t nulls{} = ~0;\n", i / 32);
  }
  head << generated_.str();

  std::vector<AbstractOperand*> input;
  std::vector<AbstractOperand*> local;
  std::vector<AbstractOperand*> output;
  params.input.forEach(
      [&](int32_t id) { input.push_back(operands_[id].get()); });

  params.local.forEach(
      [&](int32_t id) { local.push_back(operands_[id].get()); });
  params.output.forEach(
      [&](int32_t id) { output.push_back(operands_[id].get()); });

  spec.code = head.str();
  spec.entryPoints = std::move(entryPoints);
  spec.filePath = fmt::format("/tmp/{}.cu", kernelName);
#ifndef NDEBUG
  // Write the geneerated code to a file for debugger.
  writeDebugFile(spec);
#endif
  return ProgramKey{
      head.str(), std::move(input), std::move(local), std::move(output)};
}

void CompileState::makeLevel(std::vector<KernelBox>& level) {
  VELOX_CHECK_EQ(1, level.size(), "Only one program per level supported");
  auto key = makeKey();
  auto sharedState = shared_from_this();
  // The generator function captures a shared 'this'. The
  // code generation and compilation are on an executor and run after
  // the plan transformation has returned.
  auto kernel = CompiledKernel::getKernel(
      key.text,
      [sharedState, pipelineIdx = pipelineIdx_, kernelSeq = kernelSeq_]() {
        KernelSpec spec;
        sharedState->makeLevelText(pipelineIdx, kernelSeq, spec);
        return spec;
      });
  auto& params = currentCandidate_->levelParams[kernelSeq_];
  auto program = std::make_shared<Program>(
      params.input, params.local, params.output, operands_, std::move(kernel));
  for (branchIdx_ = 0; branchIdx_ < level.size(); ++branchIdx_) {
    currentBox_ = &level[branchIdx_];
    for (stepIdx_ = 0; stepIdx_ < currentBox_->steps.size(); ++stepIdx_) {
      currentBox_->steps[stepIdx_]->addInstruction(*this, *program);
    }
  }
  programs_.push_back(std::move(program));
}

bool emptyLevel(std::vector<KernelBox> level) {
  return level.empty() || level[0].steps.empty();
}

void CompileState::generatePrograms() {
  for (pipelineIdx_ = 0; pipelineIdx_ < selectedPipelines_.size();
       ++pipelineIdx_) {
    currentCandidate_ = &selectedPipelines_[pipelineIdx_];
    auto& firstStep = currentCandidate_->steps[0][0].steps.front();
    int32_t start = 0;
    if (firstStep->kind() == StepKind::kTableScan) {
      operators_.push_back(std::make_unique<TableScan>(
          *this, operators_.size(), *firstStep->as<TableScanStep>().node));
      start = 1;
    }
    if (firstStep->kind() == StepKind::kValues) {
      operators_.push_back(
          std::make_unique<Values>(*this, *firstStep->as<ValuesStep>().node));
      start = 1;
    }

    for (kernelSeq_ = start; kernelSeq_ < currentCandidate_->steps.size();
         ++kernelSeq_) {
      if (emptyLevel(currentCandidate_->steps[kernelSeq_])) {
        continue;
      }
      makeLevel(currentCandidate_->steps[kernelSeq_]);
    }
    std::vector<std::vector<ProgramPtr>> levels;
    for (auto& program : programs_) {
      levels.emplace_back();
      levels.back().push_back(std::move(program));
    }
    if (levels.empty()) {
      return;
    }
    operators_.push_back(std::make_unique<Project>(
        *this,
        selectedPipelines_[pipelineIdx_].outputType,
        std::move(levels),
        nullptr));
  }
}

} // namespace facebook::velox::wave
