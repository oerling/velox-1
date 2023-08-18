

#include "velox/experimental/wave/exec/ExprKernel.h"

namespace facebook::velox::wave {
/// Abstract representation of Wave instructions. These translate to a device
/// side ThreadBlockProgram right before execution.

  struct AbstractOperand {
    AbstractOperand(int32_t id, TypePtr type, std::string label)
      : id(id), type(type), label(label) {}

    TypePtr type;

    // Label for debugging, e.g. column name or Expr::toString output.
    std::string label;
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
  std::vector<AbstractOperand> source;
  std::vector<AbstractOperand> target;
};

struct AbstractBinary : public AbstractInstruction {
  AbstractOperand* left;
  AbstractOperand* right;
  AbstractOperand* result;
};


} // namespace facebook::velox::wave
