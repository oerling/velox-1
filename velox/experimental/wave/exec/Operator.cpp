




Operator::Operator(CompileState& state, const TypePtr& type)
  : outputType_(type) {
  


  Operator::definesSubfields(CompileState& state, const TypePtr& type, const std::string& parantPath) {
    switch (type->kind()) {
    case TypeKind::ROW: {
      auto& row = type->as<TypeKind::ROW>();
      for (auto i = 0; i < type->size(); ++i) {
	auto& child = row.childAt(i);
	auto name = row->nameOf(i);
	auto field = state.toSubfield(name);
	Value value = state.toValue(field);
	defines_[value] = state.newOperand(child->type(), name);
      }
    }
    default: {
      auto field = std::make_unique<
    }
    }
  }

}
