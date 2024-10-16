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

namespace facebook::velox::wave {

const std::string typeName(Type& type) {
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
		      if (found) {return;}
		      if (op == referenced) {
			found = true;
		      }});
    return found;
  }

  void CompileState::declareVariable(const AbstractOperand& op, bool create) {
    generated_ << fmt::format("{} r{}", typeName(op.type), op.id);
    if (create) {
      generated_ << " = ";
    } else {
      generated_ << fmt::format(
				"r{} = operand(shared->operands, {};\n", op.id, op.id);
    }
  }
  
  bool CompileState::hasMoreReferences(AbstractOperand* op, int32_t pc) {
    for (auto i = pc; i < ccurrentBox_->steps.size(); ++i) {
      if (!currentBox_->steps[i]->preservesRegisters()) {
	return false;
      }
      if (currentBox_->steps[i]->references(op)) {
	return true;
      }
    }
    return false;
  }

  std::string isRegisterNull(AbstractOperand* op) {
    VELOX_CHECK_NE(op->registerNullBit, AbstractOperand::kNoNullFlag);
    return fmt::format("(0 != (nullReg{} & (1 << {})))", op->registerNullBit / 32, registerNullBit & 31);
  }
  
  void CompileState::clearInRegister() {
    for (auto& op : operands_) {
      op->inRegister = false;
      op->registerNullBit = AbstractOperand::kNoNullBit;
    }
  }

  void NullCheck::generateMain(CompileState& state) {
    state.generated() << fmt::format("bool anyNull{} = false;\n", label);
    for (auto* op : operators) {
      if (op->inRegister) {
	if (op->notNull) {
	  continue;
	} else {
	  state.generated() << fmt::format("if (isRegNull(nulls{}, {}) { null{} = true; }", op->registerNullBit / 32, op->registerNullBit & 31, label, label);
	}
      } else {
	state.declareVariable(op);
	state.generated() << "if (isOperandNull(operands, {}) { null{s} |= (1 << {}); }", operandIdx(op), op->nullReg / 32, op->nullReg & 31)
      }
    }
  }
  
int32_t CompileState::generateOperand(const AbstractOperand& op) {
  if (op->inRegister && insideNullPropagating_) {
    generated << fmt::format(" r{} ", op->id);
    return;
  }
  if (op->notNull) {
    generated_ << fmt::format(
        "{} v{};\noperand(shared, {}, v{});",
        typeDecl(op->type),
        op->id,
        op->id,
        op->id);
    return op->id;
  }
}

void CompileState::makeOperand(AbstractOperand& op) {
  int32_t v;
  if (op->inlineExpr) {
    v = generateInline(op->expr);
  } else {
    v = generateOperand(op);
  }
}

void CompileState::makeComparison(
    Type& type,
    AbstractOperand& left,
    AbstractOperand& right,
    bool nullEq) {}

void CompileState::makeComparison(
    AbstractField& left,
    AbstractOperand& right,
    bool nullEq) {}

  void NullCheck::generateMain(CompileState& state) {

  }
  
  void Compute::generateMain(CompileState& state) {
    auto& flags = state.candidate().flags(operand);
    auto* name = operand->expr ? &operand->expr->name() : nullptr;
    auto& out = state.generated();
    out << varDecl(operand);
  }
  
  void AggregateProbe::generateMain(CompileState& state);


  CompileState::makeLevel(std::vector<KernelBox>& level) {
    VELOX_CHECK(1, level.size(), "Only one program per level supported");
    generated_ << "void __global__ __launch_bounds__(1024) waveGen" << ++kernelCounter
	       << "(KernelParams params) {\n";
    generated_ << "  GENERATED_PREAMBLE(0);\n";
    for (auto boxIdx = 0; boxIdx < level.size(); ++boxIdx) {
      auto& box = level[boxIdx];
      currentBox_ = &box;
      clearInRegister();
      bool anyRetry = false;
      for (auto i = 0; i < box.steps.size(); ++i) {
	if (box.steps[i]->hasContinue()) {
	  if (!anyRetry) {
	    anyRetry = true;
	    generated_ << "if (shared->isContinue) {\n" 
		       << "switch(entryPoint) {n";
	  }
	  generated_ << fmt::format("case {}: goto continue{};\n", i, i);
	}
	if (anyRetry) {
	  generated_ << "}\n}\n";
	}
	// Generate the  code for first execution.
	for (auto i = 0; i < box.steps.size()) {
	  auto step = steps[i];
	  if (step->hasContinue) {
	    generated_ << fmt::format("enter{}: \n", i);
	  }
	  step->generateMain(this);
	}
      }
      }
  }

  void CompileState::generatePrograms() {
    for (stageIdx = 0; stageIdx < selectedPipelines_.size(); ++stageIdx) {
      currentCandidate_ = &selectedPipelines_[stageIdx];
      auto& firstStep = currentCandidate_->steps[0][0].steps.front();
      int32_t start = 0;
      if (firstStep.stepKind() == StepKind::kTableScan) {
    operators_.push_back(
			 std::make_unique<TableScan>(*this, operators_.size(), *firstStep.as<TableScanStep>().node));
    start = 1;
	}
      for (auto i = start; i < currentCandidate_->steps.size(); ++i)  {
	makeLevel(currentCandidate_->steps[i]);
      }
      operators_.push_back(std::make_unique<Project>(this, std::move(programs), nullptr));
    }	  
  }

    }

  

  



