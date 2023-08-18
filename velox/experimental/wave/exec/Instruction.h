

#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/experimental/wave/exec/ExprKernel.h"


namespace facebook::velox::wave {
/// Abstract representation of Wave instructions. These translate to a device
/// side ThreadBlockProgram right before execution.

struct AbstractOperand {
  AbstractOperand(int32_t id, const TypePtr& type, std::string label)
      : id(id), type(type), label(label) {}

  const int32_t id;

  //Operand type.
  TypePtr type;

  // Label for debugging, e.g. column name or Expr::toString output.
  std::string label;

  // Vector with constant value, else nullptr.
  VectorPtr constant;
};

struct AbstractInstruction {
  OpCode opCode;
};

struct AbstractFilter : public AbstractInstruction {
  AbstractOperand* flags;
  AbstractOperand* indices;
};

struct AbstractWrap : public AbstractInstruction {
  AbstractOperand indices;
  std::vector<AbstractOperand*> source;
  std::vector<AbstractOperand*> target;

  void addWrap(AbstractOperand* sourceOp, AbstractOperand* targetOp = nullptr) {
    if (std::find(source.begin(), source.end(), sourceOp) != source.end()) {
      return;
    }
    source.push_back(sourceOp);
    target.push_back(targetOp ? targetOp : sourceOp);
  }
};

struct AbstractBinary : public AbstractInstruction {
  AbstractOperand* left;
  AbstractOperand* right;
  AbstractOperand* result;
};

} // namespace facebook::velox::wave
