
struct GenCtx {
  std::stringstream code;
  int32_t numNullFlags{0};
};

struct Branches {
  int32_t trueLabel;
  int32_t falseLabel;
  int32_t errorLabel;
  int32_t nullLabel;
  int32_t nextLabel;
};

const std::string typeName(Type& type) {
  switch (type.kind()) {
  case TypeKind::BIGINT: return "int64_t ";
  default: VELOX_UNSUPPORTED("No gen for type {}", type.toString());
  }
}

void CompileState::varDecl(AbstractOperand& op) {
  generated_ << fmt::format("{} v{};\n", typeName(op.type), op.id)
  generated_ <<  fmt::format("v{} = operand(shared->operands, {};\n", op.id);
}





int32_t CompileState::generateOperand(const AbstractOperand& op) {
  if (op->notNull) {
    generated_ << fmt::format("{} v{};\noperand(shared, {}, v{});", typeDecl(op->type), op->id, op->id, op->id);
    return op->id;
  }
}



void CompileState::makeOperand(AbstractOperand& op, ) {
  int32_t v;
  if (op->inlineExpr) {
    v = generateInline(op->expr);
  }  else {
    v = generateOperand(op);
  }
}

void CompileState::makeComparison(Type& type, AbstractOperand* left, AbstractOperand* right, bool nullEq) {
  
}

void CompileState::generateGroupBy(AbstractAggregationInstruction& inst) {
  
  
}



