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
#include "velox/experimental/wave/exec/WaveDriver.h"
#include "velox/experimental/wave/exec/WaveOperator.h"
#include "velox/experimental/wave/exec/WaveRegistry.h"
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

  std::string toString() const;
};

enum class StepKind : int8_t {
  kOperand,
  kNullCheck,
  kEndNullCheck,
  kValues,
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
  virtual int32_t isWrap() const {
    return AbstractOperand::kNoWrap;
  }

  virtual bool isSink() const {
    return false;
  }

  virtual bool hasContinue() const {
    return false;
  }

  virtual bool preservesRegisters() const {
    return isWrap() == AbstractOperand::kNoWrap;
  }

  /// Returns true if contains a __syncthreads() so all lanes, including
  /// inactive must hit.
  virtual bool isBarrier() const {
    return false;
  }

  virtual std::string toString() const {
    return fmt::format("step {} ", static_cast<int32_t>(kind()));
  }

  /// Returns the dynamic shared memory needed by 'this'.
  virtual int32_t sharedMemorySize() const {
    return sizeof(WaveShared);
  }

  virtual void generateMain(CompileState& state) {
    VELOX_NYI();
  }

  virtual void generateContinue(CompileState& state){};

  virtual void visitReferences(std::function<void(AbstractOperand*)> visitor){};

  virtual void visitResults(std::function<void(AbstractOperand*)> visitor){};

  bool references(AbstractOperand* op);

  /// Adds the AbstractInstruction to the current Program to interpret return
  /// state and hold OperatorStates. Only steps with retry or operator state add
  /// an instruction.
  virtual void addInstruction(CompileState& state, Program& program) {}

  template <typename T>
  T& as() {
    return *reinterpret_cast<T*>(this);
  }
};

struct ValuesStep : public KernelStep {
  StepKind kind() const override {
    return StepKind::kValues;
  }

  void visitResults(std::function<void(AbstractOperand*)> visitor) override;

  const core::ValuesNode* node;
  std::vector<AbstractOperand*> results;
};

struct TableScanStep : public KernelStep {
  StepKind kind() const override {
    return StepKind::kTableScan;
  }

  void visitResults(std::function<void(AbstractOperand*)> visitor) override;

  const core::TableScanNode* node;
  std::vector<AbstractOperand*> results;
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
  void visitReferences(std::function<void(AbstractOperand*)> visitor) override;

  void visitResults(std::function<void(AbstractOperand*)> visitor) override;

  void generateMain(CompileState& state) override;

  AbstractOperand* operand;
};

struct Filter : public KernelStep {
  StepKind kind() const override {
    return StepKind::kFilter;
  }

  int32_t isWrap() const override {
    return nthWrap;
  }

  bool isBarrier() const override {
    return true;
  }

  int32_t sharedMemorySize() const {
    return sizeof(WaveShared) + (kBlockSize / 32) * sizeof(int32_t);
  }

  void visitReferences(std::function<void(AbstractOperand*)> visitor) override {
    visitor(flag);
  }

  void visitResults(std::function<void(AbstractOperand*)> visitor) override {
    visitor(indices);
  }

  void generateMain(CompileState& state) override;

  AbstractOperand* flag;
  AbstractOperand* indices;
  int32_t nthWrap{-1};
};

  struct AggregateUpdate;
  struct AggregateProbe;
  
  /// Functions for generating different pieces of code for aggregates. Retrieved from registry based on function name and signature. The functions receive all details in 'update'.
class AggregateGenerator {
public:
  AggregateGenerator(bool needSync)
    : updateNeedsSync_(needSync) {}

  // True if  caller must ensure exclusive access to the row.

  /// Adds includes that may be needed by 'probe' or 'update'. May be called several times and should add the uncludes only once.
  virtual generateInclude(CompileState* state, const AggregateProbe* probe, const AggregateUpdate* update) {}

  /// Adds inline definitions that may be needed by 'probe' or 'update'. May be called several times and should add the uncludes only once.
  virtual generateInline(CompileState* state, const AggregateProbe* probe, const AggregateUpdate* update) {} {}

    /// Generates a declaration for the accumulator as part of a row.
    virtual void generateAccumulator(CompileState* state, const AggregateProbe* probe, const AggregateUpdate* update) = 0;

    /// Generates an update.
    virtual void generateUpdate(CompileState* state, const AggregateProbe* probe, const AggregateUpdate* update) const = 0;

    virtual void generateExtract(CompileState* state, const AggregateProbe* probe, const AggregateUpdate* update) const = 0;

  bool updateNeedsSync() const {
    return updateNeedsSync_;
  }
  
private:
  const bool updateNeedsSync_;
};

  ///
class AggregateRegistry {
public:
  const AggregateGenerator* getGenerator(const AggregateUpdate* update);

  bool registerGenerator(std::string aggregateName, std::unique_ptr<AggregateGenerator> generator);

private:
  std::unordered_map<std::string, std::unique_ptr<AggregateGenerator>> generators_;
};
  
struct AggregateUpdate : public KernelStep {
  StepKind kind() const override {
    return StepKind::kAggregateUpdate;
  }

  bool isSink() const {
    return true;
  }

  void generateMain(CompileState& state) override;

  std::string name;
  core::AggregationNode::Step step;
  /// The original argument types. Identifies the aggregate.
  std::vector<TypePtr> signature;
  AbstractOperand* rows;
  int32_t accumulatorIdx;
  // The arguments of the function. Types may differ from 'signature' for steps that take accumulators.
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

  /// Operand for accessing the row with accumulators in an update.
  AbstractOperand* rows;

  /// All accumulator updates. Needed to generate the definition for the group row.
  std::vector<const AggregateUpdate*> updates;

  /// Accumulator updates that are inlined inside the probe code. Updates can also be in a separate, wider kernel that runs different accumulators in a different TB.
  std::vector<AggregateUpdate*> inlinedUpdates;
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
  int32_t isWrap() const override {
    return nthWrap;
  }

  AbstractState* state;
  std::vector<AbstractOperand*> keys;
  AbstractOperand* hits;
  int32_t nthWrap{-1};
};

struct JoinExpand : public KernelStep {
  StepKind kind() const override {
    return StepKind::kJoinExpand;
  }
  int32_t isWrap() const override {
    return nthWrap;
  }

  AbstractOperand* hits;
  std::vector<int32_t> columns;
  std::vector<AbstractOperand*> extract;
  int32_t nthWrap{-1};
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
  CodePosition(uint16_t s, uint16_t branchIdx, uint16_t step)
      : kernelSeq(s), branchIdx(branchIdx), step(step) {}

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

  bool operator==(const CodePosition& other) const {
    return kernelSeq == other.kernelSeq && branchIdx == other.branchIdx &&
        step == other.step;
  }

  std::string toString() const;

  // Index of kernelBox in PipelineCandidate.
  uint16_t kernelSeq{kNone};

  // If many kernelBoxes each with an independent program overlap, index of the
  // program.
  uint16_t branchIdx{kNone};

  // Position of program in KernelBox.
  uint16_t step{kNone};
};

struct OperandFlags {
  CodePosition definedIn;
  CodePosition firstUse;
  CodePosition lastUse;
  int32_t wrappedAt{AbstractOperand::kNoWrap};
  bool needStore{false};

  std::string toString() const;
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

  void markParams(
      KernelBox& box,
      int32_t kernelSeq,
      int32_t branchIdx,
      std::vector<LevelParams>& params);

  /// marks 'op' as producing the output operands of steps from 'begin' to
  /// 'end'.
  void setOutputIds(
      CompileState* state,
      WaveOperator* op,
      int32_t begin,
      int32_t end);

  std::string toString() const;

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

  std::string toString() const;
};

class CompileState : public std::enable_shared_from_this<CompileState> {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver);

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

  /// Returns a key for kernel cache lookup for the step at
  /// 'pipelineIdx_', 'kernelSeq_'. Updates 'sharedSize' to the
  /// maximum dynamic shared memory needed by steps in the level.
  ProgramKey makeKey(int32_t& sharedSize);

  /// Makes the source text for kernels for the level of 'pipelineIdx',
  /// 'kernelSeq'.
  ProgramKey
  makeLevelText(int32_t pipelineIdx, int32_t kernelSeq, KernelSpec& spec);

  std::string generateIsTrue(const AbstractOperand& op);

  int32_t nextWrapId();

  // Generates an array of operands to wrap. Returns the number of distinct
  // wraps. 'id' is a sequence number from nextWrapId().
  int32_t wrapLiteral(int32_t id);

  void setInsideNullPropagating(bool flag) {
    insideNullPropagating_ = flag;
  }

  Scope* topScope() {
    return &topScope_;
  }

  AbstractOperand* fieldToOperand(common::Subfield& field, Scope* scope);
  static WaveRegistry& registry() {
    return registry_;
  }

  static AggregateRegistry& aggregateRegistry() {
    return aggregateRegistry_;
  }
  
  void functionReferenced(const AbstractOperand* op);

  std::string segmentString() const;

  /// The nthWrap of the last wrap generated into kernel code when emitting the
  /// code. Protected by 'mutex_'
  int32_t& lastPlacedWrap() {
    return lastPlacedWrap_;
  }

  /// Returns true if an access to an Operand wrapped at 'nthWrap' needs to
  /// check for wrap. Used during emitting code.
  bool mayWrap(int32_t nthWrap) {
    return nthWrap != AbstractOperand::kNoWrap && nthWrap <= lastPlacedWrap_;
  }

  /// Marks that register contents have to be reloaded, e.g. after
  /// cardinality change. Used during emitting code.
  void clearInRegister();


  
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

  /// Makes an array of AbstractOperands to correspond to the fields of
  /// 'rowType' in the top level scope.
  std::vector<AbstractOperand*> rowTypeToOperands(const RowTypePtr& rowType);

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
      const std::vector<exec::IdentityProjection>* resultProjections,
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

  std::string literalText(const AbstractOperand& op);

  void
  placeExpr(PipelineCandidate& candidate, AbstractOperand* op, bool mayDelay);

  NullCheck* addNullCheck(AbstractOperand* op);

  void markOutputStored(PipelineCandidate& candidate, Segment& segment);

  // Partitions the Driver's Operators into segments, one per cardinality
  // change. 'operatorIndex' is the index of the first considered operator and
  // is set to one after the last converted operator.
  bool makeSegments(int32_t& operatorIndex);

  void recordCandidate(PipelineCandidate& candidate, int32_t lastSegmentIdx);

  void planSegment(
      PipelineCandidate& candidate,
      float inputBatch,
      int32_t segmentIdx);

  void planPipelines();

  // Marks the operands for the output type of the last segment as copied to
  // host.
  void markHostOutput();

  void pickBest();

  void generatePrograms();

  void makeLevel(std::vector<KernelBox>& level);

  // Return true if 'nthWrap' is the wrappedAt of any of 'params'.
  bool isWrapInParams(int32_t nthWrap, const LevelParams& params);

  void setOperandByCandidate(PipelineCandidate& candidate);

  void fillExtraWrap(OperandSet& extraWrap);

  // Transforms the leading operators into WaveOperators with codegen.
  // 'operatorIndex' is set to 1 after the index of the last transformed
  // operator inde the original Driver.
  RowTypePtr makeOperators(int32_t& operatorIndex);

  int32_t declareVariable(const AbstractOperand& op, bool create);

  // Generates a check for lane active.
  void generateSkip();

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
  std::shared_ptr<WaveRuntimeObjects> runtime_;
  SubfieldMap& subfields_;

  std::vector<std::vector<ProgramPtr>> pendingLevels_;

  // All AbstractOperands. Handed off to WaveDriver after plan conversion.
  std::vector<std::unique_ptr<AbstractOperand>>& operands_;
  std::vector<std::unique_ptr<AbstractState>>& operatorStates_;

  // The Wave operators generated so far.
  std::vector<std::unique_ptr<WaveOperator>> operators_;

  // The program being generated.
  std::shared_ptr<Program> currentProgram_;

  // Sequence number for operands.
  int32_t operandCounter_{0};
  int32_t wrapCounter_{0};
  int32_t lastPlacedWrap_{-1};
  int32_t stateCounter_{0};
  InstructionStatus instructionStatus_;

  // All InstructionStatus records in instructions that have them. Used for
  // patching the final grid size when this is known.
  std::vector<InstructionStatus*> allStatuses_;

  int32_t nthContinuable_{0};
  std::shared_ptr<aggregation::AggregateFunctionRegistry>
      aggregateFunctionRegistry_;
  folly::F14FastMap<std::string, std::shared_ptr<exec::Expr>> fieldToExpr_;

  // Distinct includes pulled in by functions called from the generated kernel.
  folly::F14FastSet<std::string> includes_;

  // Text of the #include section for the generated kernel.
  std::stringstream includeText_;

  // Concatenated text of inlineable definitions of functions called from the
  // kernel.
  std::stringstream inlines_;

  //  Text of the kernel being generated.
  std::stringstream generated_;
  bool insideNullPropagating_{false};
  int32_t labelCounter_{0};
  int32_t nextSyncLabel_{0};

  thread_local static PipelineCandidate* currentCandidate_;
  thread_local static KernelBox* currentBox_;

  // Distinct functions inlined in kernel.
  folly::F14FastSet<FunctionKey> functions_;

  // The programs generated for a kernel.
  std::vector<ProgramPtr> programs_;

  // All programs for the interpreted generation.
  std::vector<ProgramPtr> allPrograms_;

  // Process wide counter for kernels.
  static std::atomic<int32_t> kernelCounter_;

  Branches branches_;
  std::vector<Segment> segments_;
  Scope topScope_;

  // Owns the steps of pipeline candidates.
  std::vector<std::unique_ptr<KernelStep>> allSteps_;

  // The number of the pipeline being generated.
  thread_local static int32_t pipelineIdx_;

  // The sequence number of the kernel in the pipeline being generated.
  thread_local static int32_t kernelSeq_;

  thread_local static int32_t branchIdx_;

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

  // Counter for making names for wraps.
  int32_t wrapId_{0};

  // Operands that have a declaration. Set when emitting code.
  OperandSet declared_;
  
  // Mutex serializing the background code generation after missing kernel
  // cache.
  std::mutex generateMutex_;

  static WaveRegistry registry_;
  static AggregateRegistry aggregateRegistry_;
};

void registerWaveFunctions();

const std::string cudaTypeName(const Type& type);

inline WaveRegistry& waveRegistry() {
  return CompileState::registry();
}

inline aggregateRegistry& waveRegistry() {
  return CompileState::aggregateRegistry();
}

  
/// Registers adapter to add Wave operators to Drivers.
void registerWave();

} // namespace facebook::velox::wave

namespace std {
template <>
struct hash<::facebook::velox::wave::CodePosition> {
  size_t operator()(const ::facebook::velox::wave::CodePosition pos) const {
    return (1 + pos.kernelSeq) * 211 + pos.branchIdx * 31 + pos.step * 29;
  }
};
} // namespace std
