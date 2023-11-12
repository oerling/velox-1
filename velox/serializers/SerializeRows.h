

void getNulls(
    const uint64_t* bits,
    folly::Range<const vector_size_t*> rows,
    uint64_t* result) {
  auto size = rows.size();
  auto indices = rows.data();
  uint8_t* resultPtr = reinterpret_cast<uint8_t*>(result);
  if (LIKELY(size < 5)) {
    uint8_t smallResult = 0;
    for (auto i = 0; i < size; ++i) {
      smallResult |= static_cast<uint8_t>(bits::isBitSet(bits, indices[i]))
          << i;
    }
    *resultPtr = smallResult;
    return;
  }
  int32_t i = 0;
  for (; i + 8 < size; i += 8) {
    *(resultPtr++) =
        simd::gather8Bits(bits, xsimd::load_unaligned(indices + i), 8);
  }
  auto bitsLeft = size - i;
  if (bitsLeft > 0) {
    *resultPtr =
        simd::gather8Bits(bits, xsimd::load_unaligned(indices + i), bitsLeft);
  }
}

// Returns ranges for the non-null rows of an array  or map. 'rows' gives the
// rows. nulls is the nulls of the array/map or nullptr if no nulls. 'offsets'
// and 'sizes' are the offsets and sizes of the array/map.Returns the number of
// index ranges. Obtains the ranges from 'rangesHolder'. If 'sizesPtr' is
// non-null, gets returns  the sizes for the inner ranges in 'sizesHolder'. If
// 'stream' is non-null, writes the lengths and nulls for the array/map into
// 'stream'.
int32_t rowsToRanges(
    folly::Range<const vector_size_t*> rows,
    const uint64_t* rawNulls,
    const vector_size_t* offsets,
    const vector_size_t* sizes,
    vector_size_t** sizesPtr,
    ScratchPtr<IndexRange>& rangesHolder,
    ScratchPtr<vector_size_t*>* sizesHolder,
    VectorStream* stream,
    Scratch& scratch) {
  auto numRows = rows.size();
  auto* innerRows = rows.data();
  int32_t numInner = rows.size();
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  if (rawNulls) {
    ScratchPtr<uint64_t> nullsHolder(scratch);
    auto* nulls = nullsHolder.get(rows.size());
    getNulls(rawNulls, rows, nulls);
    if (stream) {
      stream->appendLengths(nulls, rows, [&](auto row) { return sizes[row]; });
    }
    auto* mutableInnerRows = innerRowsHolder.get(numRows);
    numInner = simd::indicesOfSetBits(nulls, 0, numRows, mutableInnerRows);
    innerRows = mutableInnerRows;
  } else if (stream) {
    stream->appendNonNull(rows.size());
    for (auto i = 0; i < rows.size(); ++i) {
      stream->appendLength(sizes[rows[i]]);
    }
  }
  vector_size_t** sizesOut = nullptr;
  if (sizesPtr) {
    sizesOut = sizesHolder->get(numRows);
  }
  auto ranges = rangesHolder.get(numInner);
  for (auto i = 0; i < numInner; ++i) {
    if (sizesOut) {
      sizesOut[i] = sizesPtr[innerRows[i]];
    }
    ranges[i].begin = offsets[innerRows[i]];
    ranges[i].size = sizes[innerRows[i]];
  }
  return numInner;
}

template <typename T, int32_t extraScale = 1>
void copyWords(
    T* destination,
    const int32_t* indices,
    int32_t numIndices,
    const T* values) {
  for (auto i = 0; i < numIndices; ++i) {
    destination[i] = values[indices[i * extraScale]];
  }
}

template <>
void copyWords<int64_t, 1>(
    int64_t* destination,
    const int32_t* indices,
    int32_t numIndices,
    const int64_t* values) {
  constexpr int32_t kBatch = xsimd::batch<int64_t>::size;
  int32_t i = 0;
  for (; i + kBatch < numIndices; i += kBatch) {
    simd::gather<int64_t, int32_t>(
        values, simd::loadGatherIndices<int64_t, int32_t>(indices + i))
        .store_unaligned(destination + i);
  }
  auto mask = simd::leadingMask<int64_t>(numIndices - i);
  auto last = simd::maskGather<int64_t, int32_t>(
      xsimd::broadcast<int64_t>(0),
      mask,
      values,
      simd::loadGatherIndices<int64_t, int32_t>(indices + i));
  simd::storeLeading(last, mask, numIndices - i, destination + i);
}

template <int32_t extraScale = 1>
void copyWords(
    int32_t* destination,
    const int32_t* indices,
    int32_t numIndices,
    const int32_t* values) {
  int32_t i = 0;
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  for (; i + kBatch < numIndices; i += kBatch) {
    simd::gather<int32_t, int32_t>(
        values, simd::loadGatherIndices<int32_t, int32_t>(indices + i))
        .store_unaligned(destination + i);
  }
  auto mask = simd::leadingMask<int32_t>(numIndices - i);
  auto last = simd::maskGather<int32_t, int32_t>(
      xsimd::broadcast<int32_t>(0),
      mask,
      values,
      simd::loadGatherIndices<int32_t, int32_t>(indices + i));
  simd::storeLeading(last, mask, numIndices - i, destination + i);
}

template <typename T>
void appendNonNull(
    ByteStream& out,
    const uint64_t* nulls,
    folly::Range<const vector_size_t*> rows,
    const T* values,
    Scratch& scratch) {
  auto numRows = rows.size();
  ScratchPtr<int32_t> temp(scratch);
  vector_size_t localRows[32];
  const int32_t* nonNullIndices;
  int32_t numNonNull;
  if (LIKELY(numRows <= 8)) {
    uint8_t nullsByte = *reinterpret_cast<const uint8_t*>(nulls);
    nonNullIndices = simd::byteSetBits(nullsByte);
    numNonNull = __builtin_popcount(nullsByte);
  } else {
    auto mutableIndices = (numRows <= sizeof(localRows) / sizeof(localRows[0]))
        ? localRows
        : temp.get(numRows);
    numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, mutableIndices);
    nonNullIndices = mutableIndices;
  }
  if constexpr (sizeof(T) == 8) {
    constexpr int32_t kBatch = xsimd::batch<int64_t>::size;
    AppendWindow<int64_t> window(out, scratch);
    int64_t* output = window.get(numNonNull);
    copyWords(
        output,
        nonNullIndices,
        numNonNull,
        reinterpret_cast<const int64_t*>(values));
  } else if constexpr (sizeof(T) == 4) {
    constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
    AppendWindow<int32_t> window(out, scratch);
    int32_t* output = window.get(numNonNull);
    copyWords(
        output,
        nonNullIndices,
        numNonNull,
        reinterpret_cast<const int32_t*>(values));
  }
}

//*** Serialization functions on range of indices
template <TypeKind kind>
void serializeFlatVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  using T = typename TypeTraits<kind>::NativeType;
  auto flatVector = reinterpret_cast<const FlatVector<T>*>(vector);
  auto rawValues = flatVector->rawValues();
  if (!flatVector->mayHaveNulls()) {
    stream->appendNonNull(rows.size());
    AppendWindow<T> window(stream->values(), scratch);
    T* output = window.get(rows.size());
    copyWords(output, rows.data(), rows.size(), rawValues);
  } else {
    uint64_t tempNulls[16];
    ScratchPtr<uint64_t> scratchPtr(scratch);
    uint64_t* nulls = rows.size() <= sizeof(tempNulls) * 8
        ? tempNulls
        : scratchPtr.get(bits::nwords(rows.size()));
    getNulls(vector->rawNulls(), rows, nulls);
    stream->nulls().appendBits(nulls, 0, rows.size());
    appendNonNull(stream->values(), nulls, rows, rawValues, scratch);
  }
}

template <>
void serializeFlatVector<TypeKind::BOOLEAN>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& ranges,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

void serializeWrapped(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

template <>
void serializeFlatVector<TypeKind::UNKNOWN>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& ranges,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

template <>
void serializeFlatVector<TypeKind::OPAQUE>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& ranges,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

template <>
void serializeFlatVector<TypeKind::VARCHAR>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& ranges,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

template <>
void serializeFlatVector<TypeKind::VARBINARY>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& ranges,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
}

void serializeTimestampWithTimeZone(
    const RowVector* rowVector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI()
#if 0
  auto timestamps = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
  auto timezones = rowVector->childAt(1)->as<SimpleVector<int16_t>>();
  for (const auto& range : ranges) {
    for (auto i = range.begin; i < range.begin + range.size; ++i) {
      if (rowVector->isNullAt(i)) {
        stream->appendNull();
      } else {
        stream->appendNonNull();
        stream->appendOne(packTimestampWithTimeZone(
            timestamps->valueAt(i), timezones->valueAt(i)));
      }
    }
  }
#endif
}

void serializeRowVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  auto rowVector = reinterpret_cast<const RowVector*>(vector);
  ScratchPtr<int32_t> scratchPtr(scratch);
  vector_size_t* childRows;
  int32_t numChildRows = 0;
  if (isTimestampWithTimeZoneType(vector->type())) {
    serializeTimestampWithTimeZone(rowVector, rows, stream, scratch);
    return;
  }
  ScratchPtr<uint64_t> nullsHolder(scratch);
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  auto innerRows = rows.data();
  auto numInnerRows = rows.size();
  if (auto rawNulls = vector->rawNulls()) {
    auto nulls = nullsHolder.get(rows.size());
    getNulls(rawNulls, rows, nulls);
    stream->appendLengths(nulls, rows, [](int32_t) { return 1; });
    auto* mutableInnerRows = innerRowsHolder.get(rows.size());
    numInnerRows =
        simd::indicesOfSetBits(nulls, 0, rows.size(), mutableInnerRows);
    innerRows = mutableInnerRows;
  }
  for (int32_t i = 0; i < rowVector->childrenSize(); ++i) {
    serializeColumn(
        rowVector->childAt(i).get(),
        folly::Range<const vector_size_t*>(innerRows, numInnerRows),
        stream->childAt(i),
        scratch);
  }
}

void serializeArrayVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  auto arrayVector = reinterpret_cast<const ArrayVector*>(vector);

  ScratchPtr<IndexRange> rangesHolder(scratch);
  int32_t numRanges = rowsToRanges(
      rows,
      arrayVector->rawNulls(),
      arrayVector->rawOffsets(),
      arrayVector->rawSizes(),
      nullptr,
      rangesHolder,
      nullptr,
      stream,
      scratch);

  serializeColumn(
      arrayVector->elements().get(),
      folly::Range<const IndexRange*>(rangesHolder.get(), numRanges),
      stream->childAt(0));
}

void serializeMapVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
#if 0
  auto mapVector = dynamic_cast<const MapVector*>(vector);
  auto rawSizes = mapVector->rawSizes();
  auto rawOffsets = mapVector->rawOffsets();
  std::vector<IndexRange> childRanges;
  childRanges.reserve(ranges.size());
  for (int32_t i = 0; i < ranges.size(); ++i) {
    int32_t begin = ranges[i].begin;
    int32_t end = begin + ranges[i].size;
    for (int32_t offset = begin; offset < end; ++offset) {
      if (mapVector->isNullAt(offset)) {
        stream->appendNull();
      } else {
        stream->appendNonNull();
        auto size = rawSizes[offset];
        stream->appendLength(size);
        if (size > 0) {
          childRanges.emplace_back<IndexRange>({rawOffsets[offset], size});
        }
      }
    }
  }
  serializeColumn(mapVector->mapKeys().get(), childRanges, stream->childAt(0));
  serializeColumn(
      mapVector->mapValues().get(), childRanges, stream->childAt(1));

#endif
}

template <TypeKind kind>
void serializeConstantVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI();
#if 0
  using T = typename KindToFlatVector<kind>::WrapperType;
  auto constVector = dynamic_cast<const ConstantVector<T>*>(vector);
  if (constVector->valueVector()) {
    serializeWrapped(constVector, ranges, stream);
    return;
  }
  int32_t count = rangesTotalSize(ranges);
  if (vector->isNullAt(0)) {
    for (int32_t i = 0; i < count; ++i) {
      stream->appendNull();
    }
    return;
  }

  T value = constVector->valueAtFast(0);
  for (int32_t i = 0; i < count; ++i) {
    stream->appendNonNull();
    stream->appendOne(value);
  }
#endif
}

template <typename T>
void serializeBiasVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  VELOX_NYI()
#if 0
  auto biasVector = dynamic_cast<const BiasVector<T>*>(vector);
  if (!vector->mayHaveNulls()) {
    for (int32_t i = 0; i < ranges.size(); ++i) {
      stream->appendNonNull(ranges[i].size);
      int32_t end = ranges[i].begin + ranges[i].size;
      for (int32_t offset = ranges[i].begin; offset < end; ++offset) {
        stream->appendOne(biasVector->valueAtFast(offset));
      }
    }
  } else {
    for (int32_t i = 0; i < ranges.size(); ++i) {
      int32_t end = ranges[i].begin + ranges[i].size;
      for (int32_t offset = ranges[i].begin; offset < end; ++offset) {
        if (biasVector->isNullAt(offset)) {
          stream->appendNull();
          continue;
        }
        stream->appendNonNull();
        stream->appendOne(biasVector->valueAtFast(offset));
      }
    }
  }
#endif
}

void serializeColumn(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          serializeFlatVector,
          vector->typeKind(),
          vector,
          rows,
          stream,
          scratch);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          serializeConstantVector,
          vector->typeKind(),
          vector,
          rows,
          stream,
          scratch);
      break;
    case VectorEncoding::Simple::BIASED:
      VELOX_UNSUPPORTED();
#if 0
      switch (vector->typeKind()) {
        case TypeKind::SMALLINT:
          serializeBiasVector<int16_t>(vector, rows, stream, scratch);
          break;
        case TypeKind::INTEGER:
          serializeBiasVector<int32_t>(vector, rows, stream, scratch);
          break;
      case TypeKind::BIGINT:
          serializeBiasVector<int64_t>(vector, rows, stream, scratch);
          break;
        default:
          throw std::invalid_argument("Invalid biased vector type");
      }
      break;
#endif

    case VectorEncoding::Simple::ROW:
      serializeRowVector(vector, rows, stream, scratch);
      break;
    case VectorEncoding::Simple::ARRAY:
      serializeArrayVector(vector, rows, stream, scratch);
      break;
    case VectorEncoding::Simple::MAP:
      serializeMapVector(vector, rows, stream, scratch);
      break;
    case VectorEncoding::Simple::LAZY:
      serializeColumn(vector->loadedVector(), rows, stream, scratch);
      break;
    default:
      serializeWrapped(vector, rows, stream, scratch);
  }
}
