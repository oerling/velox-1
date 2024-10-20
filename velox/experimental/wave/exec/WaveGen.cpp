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
        "if (anyNull{}) { goto end{};}\n", label, label);
  }
  for (auto* op : lastUse) {
    if (op->inRegister) {
      auto ord = state.ordinal(*op);
      state.generated() << fmt::format(
          "if (isRegisterNull(nulls{}, {})) {goto end{};}\n",
          ord / 32,
          ord & 31,
          label);
      continue;
    }
    auto ord = state.declareVariable(*op);
    state.generated() << fmt::format(
        "if (!valueOrNull(operands, {}, blockBase, r{})) {goto end{};}\n",
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
  state.generated() << ")\n";
  operand->inRegister = true;
  if (flags.needStore) {
    state.generated() << fmt::format(
        "flatValue(operands, {}, blockBase) = r{};\n", ord, ord);
  }
}

void Filter::generateMain(CompileState& state) {}

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

ProgramKey CompileState::makeLevelText(KernelSpec& spec) {
  auto& level = selectedPipelines_[pipelineIdx_].steps[kernelSeq_];
  VELOX_CHECK_EQ(1, level.size(), "Only one program per level supported");
  std::stringstream head;
  auto kernelName = fmt::format("wavegen{}", ++kernelCounter_);
  std::vector<std::string> entryPoints = {kernelName};
  head << fmt::format(
      "void __global__ __launch_bounds__(1024) {}(KernelParams params) {\n",
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
        generated_ << "}\n}\n";
      }
      // Generate the  code for first execution.
      for (auto i = 0; i < box.steps.size(); ++i) {
        auto step = box.steps[i];
        if (step->hasContinue()) {
          generated_ << fmt::format("enter{}: \n", i);
        }
        step->generateMain(*this);
      }
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
  KernelSpec spec;
  auto kernel = CompiledKernel::getKernel(key.text, [&]() {
    makeLevelText(spec);
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
    for (kernelSeq_ = start; kernelSeq_ < currentCandidate_->steps.size();
         ++kernelSeq_) {
      makeLevel(currentCandidate_->steps[kernelSeq_]);
    }
    std::vector<std::vector<ProgramPtr>> levels;
    for (auto& program : programs_) {
      levels.emplace_back();
      levels.back().push_back(std::move(program));
    }
    operators_.push_back(std::make_unique<Project>(
        *this,
        selectedPipelines_[pipelineIdx_].outputType,
        std::move(levels),
        nullptr));
  }
}

} // namespace facebook::velox::wave
