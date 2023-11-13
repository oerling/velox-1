

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch);

template <TypeKind Kind>
void estimateFlatSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto valueSize = vector->type()->cppSizeInBytes();
  const auto numRows = rows.size();
  if (vector->mayHaveNulls()) {
    auto rawNulls = vector->rawNulls();
    ScratchPtr<uint64_t> nullsHolder(scratch);
    ScratchPtr<int32_t> nonNullsHolder(scratch);
    auto nulls = nullsHolder.get(bits::nwords(numRows));
    gatherBits(rawNulls, rows, nulls);
    auto nonNulls = nonNullsHolder.get(numRows);
    const auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);
    for (int32_t i = 0; i < numNonNull; ++i) {
      *sizes[nonNulls[i]] += valueSize;
    }
  } else {
    VELOX_UNREACHABLE("Non null fixed width case handled before this");
  }
}

void estimateFlatSerializedSizeVarcharOrVarbinary(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto numRows = rows.size();
  auto strings = static_cast<const FlatVector<StringView>*>(vector);
  auto rawNulls = strings->rawNulls();
  auto rawValues = strings->rawValues();
    if (!rawNulls) {
      for (auto i = 0; i < rows.size(); ++i) {
	*sizes[i] += rawValues[rows[i]].size();
      }
    } else {
      ScratchPtr<uint64_t> nullsHolder(scratch);
      ScratchPtr<int32_t> nonNullsHolder(scratch);
      auto nulls = nullsHolder.get(bits::nwords(numRows));
      gatherBits(rawNulls, rows, nulls);
      auto* nonNulls = nonNullsHolder.get(numRows);
      auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);
      
      for (int32_t i = 0; i < numNonNull; ++i) {
	*sizes[nonNulls[i]] += rawValues[rows[nonNulls[i]]].size();
      }
    }
}

template <>
void estimateFlatSerializedSize<TypeKind::VARCHAR>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, rows, sizes, scratch);
}

template <>
void estimateFlatSerializedSize<TypeKind::VARBINARY>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, rows, sizes, scratch);
}

void estimateBiasedSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  auto valueSize = vector->type()->cppSizeInBytes();
  VELOX_UNSUPPORTED();
}

void estimateWrapperSerializedSize(
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    const BaseVector* wrapper,
    Scratch& scratch) {
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  ScratchPtr<vector_size_t*> innerSizesHolder(scratch);
  const int32_t numRows = rows.size();
  int32_t numInner = 0;
  auto innerRows = innerRowsHolder.get(numRows);
  vector_size_t** innerSizes = innerSizesHolder.get(numRows);
  const BaseVector* wrapped;
  if (wrapper->encoding() == VectorEncoding::Simple::DICTIONARY &&
      !wrapper->rawNulls()) {
    // Dictionary with no nulls.
    auto* indices = wrapper->wrapInfo()->as<vector_size_t>();
    wrapped = wrapper->valueVector().get();
    simd::translate(indices, rows, innerRows);
  } else {
    wrapped = wrapper->wrappedVector();
    for (int32_t i = 0; i < rows.size(); ++i) {
      if (!wrapper->isNullAt(rows[i])) {
        innerRows[numInner] = wrapper->wrappedIndex(rows[i]);
        innerSizes[numInner] = sizes[i];
        ++numInner;
      }
    }
  }
  estimateSerializedSizeInt(
      wrapped,
      folly::Range<const vector_size_t*>(innerRows, numInner),
      innerSizes,
      scratch);
}

template <TypeKind Kind>
void estimateConstantSerializedSize(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  VELOX_CHECK(vector->encoding() == VectorEncoding::Simple::CONSTANT);
  using T = typename KindToFlatVector<Kind>::WrapperType;
  auto constantVector = vector->as<ConstantVector<T>>();
  int32_t elementSize = sizeof(T);
  if (constantVector->isNullAt(0)) {
    elementSize = 1;
  } else if (vector->valueVector()) {
    auto values = constantVector->wrappedVector();
    vector_size_t* sizePtr = &elementSize;
    vector_size_t singleRow = constantVector->wrappedIndex(0);
    estimateSerializedSizeInt(
        values,
        folly::Range<const vector_size_t*>(&singleRow, 1),
        &sizePtr,
        scratch);
  } else if (std::is_same_v<T, StringView>) {
    auto value = constantVector->valueAt(0);
    auto string = reinterpret_cast<const StringView*>(&value);
    elementSize = string->size();
  }
  for (int32_t i = 0; i < rows.size(); ++i) {
    *sizes[i] += elementSize;
  }
}
void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto numRows = rows.size();
  if (vector->type()->isFixedWidth() && !vector->mayHaveNullsRecursive()) {
    const auto elementSize = vector->type()->cppSizeInBytes();
    for (auto i = 0; i < numRows; ++i) {
      *sizes[i] += elementSize;
    }
    return;
  }
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT: {
            VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
					       estimateFlatSerializedSize,
          vector->typeKind(),
          vector,
          rows,
          sizes,
          scratch);
      break;
    }
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateConstantSerializedSize,
          vector->typeKind(),
          vector,
          rows,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      estimateWrapperSerializedSize(rows, sizes, vector, scratch);
      break;
    case VectorEncoding::Simple::BIASED:
      estimateBiasedSerializedSize(vector, rows, sizes, scratch);
      break;
    case VectorEncoding::Simple::ROW: {
      ScratchPtr<vector_size_t> innerRowsHolder(scratch);
      ScratchPtr<vector_size_t*> innerSizesHolder(scratch);
      ScratchPtr<uint64_t> nullsHolder(scratch);
      auto* innerRows = rows.data();
      auto* innerSizes = sizes;
      const auto numRows = rows.size();
      int32_t numInner = numRows;
      if (vector->mayHaveNulls()) {
        auto nulls = nullsHolder.get(bits::nwords(numRows));
        gatherBits(vector->rawNulls(), rows, nulls);
        auto mutableInnerRows = innerRowsHolder.get(numRows);
        numInner = simd::indicesOfSetBits(nulls, 0, numRows, mutableInnerRows);
        innerSizes = innerSizesHolder.get(numInner);
        for (auto i = 0; i < numInner; ++i) {
          innerSizes[i] = sizes[mutableInnerRows[i]];
        }
	simd::translate(rows.data(), folly::Range<const vector_size_t*>(mutableInnerRows, numInner), mutableInnerRows);
        innerRows = mutableInnerRows;
      }
      auto rowVector = vector->as<RowVector>();
      auto children = rowVector->children();
      for (auto& child : children) {
        if (child) {
          estimateSerializedSizeInt(
              child.get(),
              folly::Range(innerRows, numInner),
              innerSizes,
              scratch);
        }
      }
      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto mapVector = vector->asUnchecked<MapVector>();
      ScratchPtr<IndexRange> rangeHolder(scratch);
      ScratchPtr<vector_size_t*> sizesHolder(scratch);
      const auto numRanges = rowsToRanges(
          rows,
          mapVector->rawNulls(),
          mapVector->rawOffsets(),
          mapVector->rawSizes(),
          sizes,
          rangeHolder,
          &sizesHolder,
          nullptr,
          scratch);
      if (numRanges == 0) {
	return;
      }
      estimateSerializedSizeInt(
          mapVector->mapKeys().get(),
          folly::Range<const IndexRange*>(rangeHolder.get(), numRanges),
          sizesHolder.get(),
          scratch);
      estimateSerializedSizeInt(
          mapVector->mapValues().get(),
          folly::Range<const IndexRange*>(rangeHolder.get(), numRanges),
          sizesHolder.get(),
          scratch);
      break;
    }
    case VectorEncoding::Simple::ARRAY: {
      auto arrayVector = vector->as<ArrayVector>();
      ScratchPtr<IndexRange> rangeHolder(scratch);
      ScratchPtr<vector_size_t*> sizesHolder(scratch);
      const auto numRanges = rowsToRanges(
          rows,
          arrayVector->rawNulls(),
          arrayVector->rawOffsets(),
          arrayVector->rawSizes(),
          sizes,
          rangeHolder,
          &sizesHolder,
          nullptr,
          scratch);
      if (numRanges == 0) {
	return;
      }
      estimateSerializedSizeInt(
          arrayVector->elements().get(),
          folly::Range<const IndexRange*>(rangeHolder.get(), numRanges),
          sizesHolder.get(),
          scratch);
      estimateSerializedSizeInt(
          arrayVector->elements().get(),
          folly::Range<const IndexRange*>(rangeHolder.get(), numRanges),
          sizesHolder.get(),
          scratch);
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeInt(vector->loadedVector(), rows, sizes, scratch);
      break;
    default:
      VELOX_CHECK(false, "Unsupported vector encoding {}", vector->encoding());
  }
}
