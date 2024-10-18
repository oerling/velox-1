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

#pragma once

#include "velox/exec/Operator.h"
#include "velox/experimental/wave/exec/Accumulators.h"
#include "velox/experimental/wave/exec/AggregateFunctionRegistry.h"
#include "velox/experimental/wave/exec/WaveOperator.h"
#include "velox/expression/Expr.h"
#include "velox/expression/SwitchExpr.h"

namespace facebook::velox::wave {

using SubfieldMap =
    folly::F14FastMap<std::string, std::unique_ptr<common::Subfield>>;

/// Branch targets when generating device code.
struct Branches {
  int32_t trueLabel;
  int32_t falseLabel;
  int32_t errorLabel;
  int32_t nullLabel;
  int32_t nextLabel;
};

struct Scope {
  Scope() = default;
  Scope(Scope* parent) : parent(parent) {}

  AbstractOperand* findValue(const Value& value);

  DefinesMap operandMap;
  Scope* parent{nullptr};
};

enum class StepKind : int8_t {
  kOperand,
  kNullCheck,
  kEndNullCheck,
  kTableScan,
  kFilter,
  kAggregateProbe,
  kAggregateUpdate,
  kReadAggregation,
  kJoinBuild,
  kJoinProbe,
  kJoinExpand
};

class CompileState;

struct KernelStep {
  virtual ~KernelStep() = default;
  virtual StepKind kind() const = 0;
  virtual bool isWrap() const {
    return false;
  }
  virtual bool hasContinue() const {
    return false;
  }

  virtual bool preservesRegisters() const {
    return !isWrap();
  }
  
  virtual void generateMain(CompileState& state) {
    VELOX_NYI();
  }

  virtual void generateContinue(CompileState& state){};

  virtual void visitReferences(std::function<void(AbstractOperand*)> visitor){};

  virtual void visitResults(std::function<void(AbstractOperand*)> visitor){};

  bool references(AbstractOperand* op);

  template <typename T>
  T& as() {
    return *reinterpret_cast<T*>(this);
  }
};

struct TableScanStep : public KernelStep {
  StepKind kind() const override {
    return StepKind::kTableScan;
  }
  const core::TableScanNode* node;
};

struct NullCheck : public KernelStep {
  StepKind kind() const override {
    return StepKind::kNullCheck;
  }

  void generateMain(CompileState& state) override;

  std::vector<AbstractOperand*> operands;
  AbstractOperand* result;
  int32_t label;
  int32_t endIdx{-1};
};

struct EndNullCheck : public KernelStep {
  StepKind kind() const override {
    return StepKind::kEndNullCheck;
  }
  void generateMain(CompileState& state) override;
  
  AbstractOperand* result;
  int32_t label;
};

struct Compute : public KernelStep {
  StepKind kind() const override {
    return StepKind::kOperand;
  }
  bool hasContinue() const override {
    return operand->retriable;
  }

  void generateMain(CompileState& state) override;

  AbstractOperand* operand;
};

struct Filter : public KernelStep {
  StepKind kind() const override {
    return StepKind::kFilter;
  }

  bool isWrap() const override {
    return true;
  }

  void generateMain(CompileState& state) override;

  AbstractOperand* flag;
  AbstractOperand* indices;
  int32_t nthWrap{-1};
};

struct AggregateUpdate : public KernelStep {
  StepKind kind() const override {
    return StepKind::kAggregateUpdate;
  }

  void generateMain(CompileState& state) override;

  std::string name;
  AbstractOperand* rows;
  core::AggregationNode::Step step;
  int32_t accumulatorIdx;
  std::vector<AbstractOperand*> args;
  AbstractOperand* condition{nullptr};
  bool distinct{false};
  std::vector<AbstractOperand*> sort;
  AbstractOperand* pushdownColumn;
  std::optional<int32_t> restartNumber;
  AbstractOperand* result;
};

struct AggregateProbe : public KernelStep {
  StepKind kind() const override {
    return StepKind::kAggregateProbe;
  }

  bool hasContinue() const override {
    return true;
  }

  void generateMain(CompileState& state) override;

  AbstractState* state;
  std::vector<AbstractOperand*> keys;
  AbstractOperand* rows;
};

struct ReadAggregation : public KernelStep {
  StepKind kind() const override {
    return StepKind::kReadAggregation;
  }
  core::AggregationNode::Step step;
  AbstractState* state;
  std::vector<AbstractOperand*> keys;
  std::vector<AggregateUpdate*> funcs;
};

struct JoinBuild : public KernelStep {
  StepKind kind() const override {
    return StepKind::kJoinBuild;
  }
  AbstractState* state;
  std::vector<AbstractOperand*> keys;
  std::vector<AbstractOperand*> dependent;
};

struct JoinProbe : public KernelStep {
  StepKind kind() const override {
    return StepKind::kJoinProbe;
  }
  bool isWrap() const override {
    return true;
  }

  AbstractState* state;
  std::vector<AbstractOperand*> keys;
  AbstractOperand* hits;
};

struct JoinExpand : public KernelStep {
  StepKind kind() const override {
    return StepKind::kJoinExpand;
  }
  bool isWrap() const override {
    return true;
  }

  AbstractOperand* hits;
  std::vector<int32_t> columns;
  std::vector<AbstractOperand*> extract;
};

struct KernelBox {
  std::vector<KernelStep*> steps;
  // Number of consecutive wraps (filter, join, unnest...).
  int32_t numWraps{0};
};

// Position of a definition or use of data in a pipeline grid.
struct CodePosition {
  static constexpr uint16_t kNone = ~0;

  CodePosition() = default;
  CodePosition(uint16_t s) : kernelSeq(s) {}
  CodePosition(uint16_t s, uint16_t step) : kernelSeq(s), step(step) {}
  CodePosition(uint16_t s, uint16_t step, uint16_t branchIdx)
      : kernelSeq(s), step(step), branchIdx(branchIdx) {}

  bool empty() const {
    return kernelSeq == kNone;
  }

  bool isBefore(const CodePosition& other) {
    if (kernelSeq == other.kernelSeq && branchIdx != other.branchIdx) {
      VELOX_FAIL(
          "Bad comparison of CodePosition in between parallel  kernel boxes");
    }
    return kernelSeq < other.kernelSeq ||
        (kernelSeq == other.kernelSeq && step < other.step);
  }

  // Index of kernelBox in PipelineCandidate.
  uint16_t kernelSeq{kNone};
  // Position of program in KernelBox.
  uint16_t step{kNone};
  // If many kernelBoxes each with an independent program overlap, index of the
  // program.
  uint16_t branchIdx{kNone};
};

struct OperandFlags {
  CodePosition definedIn;
  CodePosition firstUse;
  CodePosition lastUse;
  CodePosition wrappedAt;
  bool needStore{0};
};

/// Contains input/local/output param sets for each level of a
/// PipelineCandidate.
struct LevelParams {
  OperandSet input;
  OperandSet local;
  OperandSet output;
};

struct PipelineCandidate {
  OperandFlags& flags(const AbstractOperand* op) {
    if (op->id >= operandFlags.size()) {
      operandFlags.resize(op->id + 10);
    }
    return operandFlags[op->id];
  }

  void makeOperandSets(int32_t kernelSeq);

  void markParams(KernelBox& box, int32_t kernelSeq, LevelParams& params);

  KernelBox* boxOf(CodePosition pos) {
    return &steps[pos.kernelSeq][pos.branchIdx];
  }

  std::vector<OperandFlags> operandFlags;
  std::vector<std::vector<KernelBox>> steps;

  /// Params for each vector of KernelBox.
  std::vector<LevelParams> levelParams;
  KernelBox* currentBox{nullptr};
  int32_t boxIdx{0};

  RowTypePtr outputType;
};

/// Describes the operation at the start of a segment.
enum class BoundaryType {
  // Table scan, values, exchange
  kSource,
  // Expressions. May or may not produce named projected columns. May be
  // generated at place of use or generated in place and written to memory.
  kExpr,
  // Filter in join or standalone
  kFilter,
  // n:Guaranteed 1 join, e.g, semi/antijoin.
  kReducingJoin,
  // Join that can produce multiple hits
  kJoin,

  // Filter associated to non-inner join.
  kJoinFilter,
  kAggregation
};

/// Describes the space between cardinality changes in an operator pipeline.
struct Segment {
  BoundaryType boundary;

  int32_t ordinal;

  const core::PlanNode* planNode{nullptr};

  // Operands defined here. These can be referenced by subsequent segments.
  // Local intermediates like ones created inside conditionals or lambdas are
  // not included. If this is a filter, this is the bool filter  value.
  std::vector<AbstractOperand*> topLevelDefined;

  // If this projects out columns, these are the column names, 1:1 to
  // topLevelDefined.
  std::vector<common::Subfield*> projectedName;

  // intermediates that are unconditionally computed and could be referenced
  // from subsequent places for optimization, e.g. dedupping. Does not include
  // intermediates inside conditional branches.
  std::vector<AbstractOperand*> definedIntermediate;

  // Aggregation, read aggregation, join, ... References planned operands via
  // AbstractOperand.
  std::vector<KernelStep*> steps;

  // Cardinality change. 0.5 means that half the input passes.
  float fanout{1};

  // Projected top level columns if this is not a sink.
  RowTypePtr outputType;
};

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Replaces sequences of Operators in the Driver given at construction with
  // Wave equivalents. Returns true if the Driver was changed.
  bool compile();

  common::Subfield* toSubfield(const exec::Expr& expr);

  common::Subfield* toSubfield(const std::string& name);

  AbstractOperand* newOperand(AbstractOperand& other);

  AbstractOperand* newOperand(
      const TypePtr& type,
      const std::string& label = "");

  Program* newProgram();

  Value toValue(const exec::Expr& expr);

  Value toValue(const core::FieldAccessTypedExpr& field);

  AbstractOperand* addIdentityProjections(AbstractOperand* source);
  AbstractOperand* findCurrentValue(Value value);

  AbstractOperand* findCurrentValue(
      const std::shared_ptr<const core::FieldAccessTypedExpr>& field) {
    Value value = toValue(*field);
    return findCurrentValue(value);
  }

  AbstractOperand* addExpr(const exec::Expr& expr);

  void addInstruction(
      std::unique_ptr<AbstractInstruction> instruction,
      AbstractOperand* result,
      const std::vector<Program*>& inputs);

  std::vector<AbstractOperand*>
  addExprSet(const exec::ExprSet& set, int32_t begin, int32_t end);
  std::vector<std::vector<ProgramPtr>> makeLevels(int32_t startIndex);

  GpuArena& arena() const {
    return *arena_;
  }

  int numOperators() const {
    return operators_.size();
  }

  GpuArena& arena() {
    return *arena_;
  }

  std::stringstream& generated() {
    return generated_;
  }

  PipelineCandidate& candidate() {
    return *currentCandidate_;
  }

  CodePosition currentPosition() {
    return CodePosition(kernelSeq_, branchIdx_, stepIdx_);
  }

  int32_t declareVariable(const AbstractOperand& op);

    int32_t ordinal(const AbstractOperand& op);

  OperandFlags& flags(const AbstractOperand& op) const {
    return currentCandidate_->flags(&op);
  }

  bool hasMoreReferences(AbstractOperand* op, int32_t pc);

  void generateOperand(const AbstractOperand& op);

  
 private:
  bool
  addOperator(exec::Operator* op, int32_t& nodeIndex, RowTypePtr& outputType);

  void addFilter(const exec::Expr& expr, const RowTypePtr& outputType);

  AbstractState* newState(
      StateKind kind,
      const std::string& idString,
      const std::string& label);

  void addFilterProject(
      exec::Operator* op,
      RowTypePtr& outputType,
      int32_t& nodeIndex);

  /// Adds a projection operator containing programs starting at 'firstProgram'
  /// for the rest of 'allPrograms_'..
  void makeProject(int32_t firstProgram, RowTypePtr outputType);

  void makeAggregateLayout(AbstractAggregation& aggregate);

  void setAggregateFromPlan(
      const core::AggregationNode::Aggregate& planAggregate,
      AbstractAggInstruction& agg);

  void makeAggregateAccumulate(const core::AggregationNode* node);

  bool reserveMemory();

  // Adds 'instruction' to the suitable program and records the result
  // of the instruction to the right program. The set of programs
  // 'instruction's operands depend is in 'programs'. If 'instruction'
  // depends on all immutable programs, start a new one. If all
  // dependences are from the same open program, add the instruction
  // to that. If Only one of the programs is mutable, ad the
  // instruction to that.
  void addInstruction(
      std::unique_ptr<Instruction> instruction,
      const AbstractOperand* result,
      const std::vector<Program*>& inputs);

  void setConditionalNullable(AbstractBinary& binary);

  // Adds 'op->id' to 'nullableIf' if not already there.
  void addNullableIf(
      const AbstractOperand* op,
      std::vector<OperandId>& nullableIf);

  Program* programOf(AbstractOperand* op, bool create = true);

  const std::shared_ptr<aggregation::AggregateFunctionRegistry>&
  aggregateFunctionRegistry();

  template <typename T>
  T* makeStep() {
    auto unq = std::make_unique<T>();
    auto* ptr = unq.get();
    allSteps_.push_back(std::move(unq));
    return ptr;
  }

  AbstractOperand* fieldToOperand(common::Subfield& field, Scope* scope);

  AbstractOperand* fieldToOperand(
      const core::FieldAccessTypedExpr& field,
      Scope* scope);

  AbstractOperand* switchOperand(
      const exec::SwitchExpr& switchExpr,
      Scope* scope);

  AbstractOperand* exprToOperand(const exec::Expr& expr, Scope* scope);

  Segment& addSegment(
      BoundaryType boundary,
      const core::PlanNode* node,
      RowTypePtr outputType);

  std::vector<AbstractOperand*> tryExprSet(
      const exec::ExprSet& exprSet,
      int32_t begin,
      int32_t end,
      const RowTypePtr& outputType);

  void tryFilter(const exec::Expr& expr, const RowTypePtr& outputType);

  void tryFilterProject(
      exec::Operator* op,
      RowTypePtr& outputType,
      int32_t& nodeIndex);

  bool tryPlanOperator(
      exec::Operator* op,
      int32_t& nodeIndex,
      RowTypePtr& outputType);

  void
  placeExpr(PipelineCandidate& candidate, AbstractOperand* op, bool mayDelay);

  NullCheck* addNullCheck(AbstractOperand* op);

  void markOutputStored(PipelineCandidate& candidate, Segment& segment);

  bool makeSegments();

  void recordCandidate(PipelineCandidate& candidate, int32_t lastSegmentIdx);

  void planSegment(
      PipelineCandidate& candidate,
      float inputBatch,
      int32_t segmentIdx);

  void planPipelines();

  void pickBest();

  void generatePrograms();

  void makeLevel(std::vector<KernelBox>& level);

  ProgramKey makeKey(PipelineCandidate& candidate, int32_t kernelIdx);

  void makeDriver();

  int32_t declareVariable(const AbstractOperand& op, bool create);

  void clearInRegister();

  std::unique_ptr<GpuArena> arena_;
  // The operator and output operand where the Value is first defined.
  DefinesMap definedBy_;

  // The Operand where Value is available after all projections placed to date.
  DefinesMap projectedTo_;

  // Index of WaveOperator producing the operand.
  folly::F14FastMap<AbstractOperand*, int32_t> operandOperatorIndex_;

  folly::F14FastMap<AbstractOperand*, Program*> definedIn_;

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
  SubfieldMap subfields_;

  std::vector<ProgramPtr> allPrograms_;

  std::vector<std::vector<ProgramPtr>> pendingLevels_;

  // All AbstractOperands. Handed off to WaveDriver after plan conversion.
  std::vector<std::unique_ptr<AbstractOperand>> operands_;
  std::vector<std::unique_ptr<AbstractState>> operatorStates_;

  // The Wave operators generated so far.
  std::vector<std::unique_ptr<WaveOperator>> operators_;

  // The program being generated.
  std::shared_ptr<Program> currentProgram_;

  // Sequence number for operands.
  int32_t operandCounter_{0};
  int32_t wrapCounter_{0};
  int32_t stateCounter_{0};
  InstructionStatus instructionStatus_;

  // All InstructionStatus records in instructions that have them. Used for
  // patching the final grid size when this is known.
  std::vector<InstructionStatus*> allStatuses_;

  int32_t nthContinuable_{0};
  std::shared_ptr<aggregation::AggregateFunctionRegistry>
      aggregateFunctionRegistry_;
  folly::F14FastMap<std::string, std::shared_ptr<exec::Expr>> fieldToExpr_;

  //  Text of the kernel being generated.
  std::stringstream generated_;
  bool insideNullPropagating_{false};
  int32_t labelCounter_{0};

  PipelineCandidate* currentCandidate_{nullptr};
  KernelBox* currentBox_{nullptr};

  // The programs generated for a kernel.
  std::vector<ProgramPtr> programs_;

  // Process wide counter for kernels.
  static std::atomic<int32_t> kernelCounter_;

  Branches branches_;
  std::vector<Segment> segments_;
  Scope topScope_;

  // Owns the steps of pipeline candidates.
  std::vector<std::unique_ptr<KernelStep>> allSteps_;

  // The number of the pipeline being generated.
  int32_t pipelineIdx_{0};

  // The sequence number of the kernel in the pipeline being generated.
  int32_t kernelSeq_;

  int32_t branchIdx_;

  int32_t stepIdx_;
  
  // Candidates being considered for a pipeline.
  std::vector<PipelineCandidate> candidates_;

  // Selected candidates for all stages, e.g. from scan to agg and from agg to
  // end. These are actually generated.
  std::vector<PipelineCandidate> selectedPipelines_;

  // Renames of columns introduced by project nodes that rename a top level
  // column to something else with no expression.
  std::vector<std::unordered_map<std::string, std::string>> renames_;

  // For each in 'renames_', a copy of 'topScope' before the rename was
  // installed.
  std::vector<Scope> topScopes_;
};

/// Registers adapter to add Wave operators to Drivers.
void registerWave();

} // namespace facebook::velox::wave
