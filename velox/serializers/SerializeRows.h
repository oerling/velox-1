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

void gatherBits(
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
  auto* nonNullRows = innerRows;
  int32_t numInner = rows.size();
  ScratchPtr<vector_size_t> nonNullHolder(scratch);
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  if (rawNulls) {
    ScratchPtr<uint64_t> nullsHolder(scratch);
    auto* nulls = nullsHolder.get(rows.size());
    gatherBits(rawNulls, rows, nulls);
    auto* mutableNonNullRows = nonNullHolder.get(numRows);
    auto* mutableInnerRows = innerRowsHolder.get(numRows);
    numInner = simd::indicesOfSetBits(nulls, 0, numRows, mutableNonNullRows);
    if (stream) {
      stream->appendLengths(
          nulls, rows, numInner, [&](auto row) { return sizes[row]; });
    }
    simd::translate(
        rows.data(),
        folly::Range<const vector_size_t*>(mutableNonNullRows, numInner),
        mutableInnerRows);
    nonNullRows = mutableNonNullRows;
    innerRows = mutableInnerRows;
  } else if (stream) {
    stream->appendNonNull(rows.size());
    for (auto i = 0; i < rows.size(); ++i) {
      stream->appendLength(sizes[rows[i]]);
    }
  }
  vector_size_t** sizesOut = nullptr;
  if (sizesPtr) {
    sizesOut = sizesHolder->get(numInner);
  }
  auto ranges = rangesHolder.get(numInner);
  for (auto i = 0; i < numInner; ++i) {
    if (sizesOut) {
      sizesOut[i] = sizesPtr[nonNullRows[i]];
    }
    ranges[i].begin = offsets[innerRows[i]];
    ranges[i].size = sizes[innerRows[i]];
  }
  return numInner;
}

template <typename T>
void copyWords(
    T* destination,
    const int32_t* indices,
    int32_t numIndices,
    const T* values) {
  for (auto i = 0; i < numIndices; ++i) {
    destination[i] = values[indices[i]];
  }
}

template <>
void copyWords(
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

template <>
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
void copyWordsWithRows(
    T* destination,
    const int32_t* rows,
    const int32_t* indices,
    int32_t numIndices,
    const T* values) {
  for (auto i = 0; i < numIndices; ++i) {
    destination[i] = values[rows[indices[i]]];
  }
}

template <>
void copyWordsWithRows(
    int64_t* destination,
    const int32_t* rows,
    const int32_t* indices,
    int32_t numIndices,
    const int64_t* values) {
  constexpr int32_t kBatch = xsimd::batch<int64_t>::size;
  constexpr int32_t kDoubleBatch = xsimd::batch<int32_t>::size;
  int32_t i = 0;
  for (; i + kDoubleBatch < numIndices; i += kDoubleBatch) {
    auto indexVector = simd::gather<int32_t, int32_t>(
        rows, simd::loadGatherIndices<int32_t, int32_t>(indices + i));
    simd::gather<int64_t, int32_t>(
        values,
        simd::loadGatherIndices<int64_t, int32_t>(
            reinterpret_cast<int32_t*>(&indexVector)))
        .store_unaligned(destination + i);
    simd::gather<int64_t, int32_t>(
        values,
        simd::loadGatherIndices<int64_t, int32_t>(
            reinterpret_cast<int32_t*>(&indexVector) + kBatch))
        .store_unaligned(destination + i + kBatch);
  }
  int32_t numLeft = numIndices - i;
  auto indexMask = simd::leadingMask<int32_t>(numLeft);
  auto indexVector = simd::maskGather<int32_t, int32_t>(
      xsimd::broadcast<int32_t>(0),
      indexMask,
      rows,
      simd::loadGatherIndices<int32_t, int32_t>(indices + i));
  int32_t indexVectorOffset = 0;
  if (numLeft >= kBatch) {
    simd::gather<int64_t, int32_t>(
        values,
        simd::loadGatherIndices<int64_t, int32_t>(
            reinterpret_cast<int32_t*>(&indexVector)))
        .store_unaligned(destination + i);
    numLeft -= kBatch;
    i += kBatch;
    indexVectorOffset = kBatch;
  }

  if (numLeft > 0) {
    auto mask = simd::leadingMask<int64_t>(numLeft);
    auto last = simd::maskGather<int64_t, int32_t>(
        xsimd::broadcast<int64_t>(0),
        mask,
        values,
        simd::loadGatherIndices<int64_t, int32_t>(
            reinterpret_cast<int32_t*>(&indexVector) + indexVectorOffset));
    simd::storeLeading(last, mask, numLeft, destination + i);
  }
}

template <>
void copyWordsWithRows(
    int32_t* destination,
    const int32_t* rows,
    const int32_t* indices,
    int32_t numIndices,
    const int32_t* values) {
  int32_t i = 0;
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  for (; i + kBatch < numIndices; i += kBatch) {
    auto indexVector = simd::gather<int32_t, int32_t>(
        rows, simd::loadGatherIndices<int32_t, int32_t>(indices + i));
    simd::gather<int32_t, int32_t>(values, indexVector)
        .store_unaligned(destination + i);
  }
  auto mask = simd::leadingMask<int32_t>(numIndices - i);
  auto indexVector = simd::maskGather<int32_t, int32_t>(
      xsimd::broadcast<int32_t>(0),
      mask,
      rows,
      simd::loadGatherIndices<int32_t, int32_t>(indices + i));
  auto last = simd::maskGather<int32_t, int32_t>(
      xsimd::broadcast<int32_t>(0), mask, values, indexVector);
  simd::storeLeading(last, mask, numIndices - i, destination + i);
}

template <typename T>
void appendNonNull(
    VectorStream* stream,
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
  stream->appendNulls(nulls, 0, rows.size(), numNonNull);
  ByteStream& out = stream->values();
  if constexpr (sizeof(T) == 8) {
    constexpr int32_t kBatch = xsimd::batch<int64_t>::size;
    AppendWindow<int64_t> window(out, scratch);
    int64_t* output = window.get(numNonNull);
    copyWordsWithRows(
        output,
        rows.data(),
        nonNullIndices,
        numNonNull,
        reinterpret_cast<const int64_t*>(values));
  } else if constexpr (sizeof(T) == 4) {
    constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
    AppendWindow<int32_t> window(out, scratch);
    int32_t* output = window.get(numNonNull);
    copyWordsWithRows(
        output,
        rows.data(),
        nonNullIndices,
        numNonNull,
        reinterpret_cast<const int32_t*>(values));
  } else {
    AppendWindow<T> window(out, scratch);
    T* output = window.get(numNonNull);
    copyWordsWithRows(output, rows.data(), nonNullIndices, numNonNull, values);
  }
}

void appendStrings(
    const uint64_t* nulls,
    folly::Range<const vector_size_t*> rows,
    const StringView* views,
    VectorStream* stream,
    Scratch& scratch) {
  if (!nulls) {
    stream->appendLengths(nullptr, rows, rows.size(), [&](auto row) {
      return views[row].size();
    });
    for (auto i = 0; i < rows.size(); ++i) {
      auto& view = views[rows[i]];
      stream->values().appendStringPiece(
          folly::StringPiece(view.data(), view.size()));
    }
    return;
  }
  auto innerRows = rows.data();
  int32_t numInnerRows = rows.size();
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  auto mutableInner = innerRowsHolder.get(rows.size());
  numInnerRows = simd::indicesOfSetBits(nulls, 0, rows.size(), mutableInner);
  innerRows = mutableInner;
  stream->appendLengths(
      nulls, rows, numInnerRows, [&](auto row) { return views[row].size(); });
  for (auto i = 0; i < numInnerRows; ++i) {
    auto& view = views[rows[innerRows[i]]];
    stream->values().appendStringPiece(
        folly::StringPiece(view.data(), view.size()));
  }
}

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
    if (std::is_same_v<T, StringView>) {
      appendStrings(
          nullptr,
          rows,
          reinterpret_cast<const StringView*>(rawValues),
          stream,
          scratch);
      return;
    }

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
    gatherBits(vector->rawNulls(), rows, nulls);
    if (std::is_same_v<T, StringView>) {
      appendStrings(
          nulls,
          rows,
          reinterpret_cast<const StringView*>(rawValues),
          stream,
          scratch);
      return;
    }
    appendNonNull(stream, nulls, rows, rawValues, scratch);
  }
}

uint64_t bitsToBytesMap[256];

uint64_t bitsToBytes(uint8_t byte) {
  return bitsToBytesMap[byte];
}

template <>
void serializeFlatVector<TypeKind::BOOLEAN>(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  auto flatVector = reinterpret_cast<const FlatVector<bool>*>(vector);
  auto rawValues = flatVector->rawValues<uint64_t*>();
  uint64_t smallBits[16];
  uint64_t* valueBits = smallBits;
  ScratchPtr<uint64_t> bitsHolder(scratch);
  int32_t numValueBits;
  if (!flatVector->mayHaveNulls()) {
    stream->appendNonNull(rows.size());
    if (rows.size() > sizeof(smallBits) * 8) {
      valueBits = bitsHolder.get(bits::nwords(rows.size()));
    }
    gatherBits(reinterpret_cast<const uint64_t*>(rawValues), rows, valueBits);
    numValueBits = rows.size();
  } else {
    uint64_t* nulls = rows.size() <= sizeof(smallBits) * 8
        ? smallBits
        : bitsHolder.get(bits::nwords(rows.size()));
    gatherBits(vector->rawNulls(), rows, nulls);
    ScratchPtr<vector_size_t> nonNullsHolder(scratch);
    auto nonNulls = nonNullsHolder.get(rows.size());
    numValueBits = simd::indicesOfSetBits(nulls, 0, rows.size(), nonNulls);
    stream->appendNulls(nulls, 0, rows.size(), numValueBits);
    valueBits = nulls;
    simd::translate(
        rows.data(),
        folly::Range<const vector_size_t*>(nonNulls, numValueBits),
        nonNulls);
    gatherBits(
        reinterpret_cast<const uint64_t*>(rawValues),
        folly::Range<const vector_size_t*>(nonNulls, numValueBits),
        valueBits);
  }
  AppendWindow<uint8_t> window(stream->values(), scratch);
  uint8_t* output = window.get(numValueBits);
  const auto numBytes = bits::nbytes(numValueBits);
  for (auto i = 0; i < numBytes; ++i) {
    uint64_t word = bitsToBytes(reinterpret_cast<uint8_t*>(valueBits)[i]);
    if (i < numBytes - 1) {
      reinterpret_cast<uint64_t*>(output)[i] = word;
    } else {
      memcpy(output + i * 8, &word, numValueBits - i * 8);
    }
  }
}

void serializeWrapped(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  ScratchPtr<vector_size_t> innerRowsHolder(scratch);
  const int32_t numRows = rows.size();
  int32_t numInner = 0;
  auto innerRows = innerRowsHolder.get(numRows);
  const BaseVector* wrapped;
  if (vector->encoding() == VectorEncoding::Simple::DICTIONARY &&
      !vector->rawNulls()) {
    // Dictionary with no nulls.
    auto* indices = vector->wrapInfo()->as<vector_size_t>();
    wrapped = vector->valueVector().get();
    simd::translate(indices, rows, innerRows);
    numInner = numRows;
  } else {
    wrapped = vector->wrappedVector();
    for (int32_t i = 0; i < rows.size(); ++i) {
      if (vector->isNullAt(rows[i])) {
        if (numInner > 0) {
          serializeColumn(
              wrapped,
              folly::Range<const vector_size_t*>(innerRows, numInner),
              stream,
              scratch);
          numInner = 0;
        }
        stream->appendNull();
        continue;
      }
      innerRows[numInner++] = vector->wrappedIndex(rows[i]);
    }
  }
  if (numInner > 0) {
    serializeColumn(
        wrapped,
        folly::Range<const vector_size_t*>(innerRows, numInner),
        stream,
        scratch);
  }
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

void serializeTimestampWithTimeZone(
    const RowVector* rowVector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  auto timestamps = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
  auto timezones = rowVector->childAt(1)->as<SimpleVector<int16_t>>();
  for (auto i : rows) {
    if (rowVector->isNullAt(i)) {
      stream->appendNull();
    } else {
      stream->appendNonNull();
      stream->appendOne(packTimestampWithTimeZone(
          timestamps->valueAt(i), timezones->valueAt(i)));
    }
  }
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
    gatherBits(rawNulls, rows, nulls);
    auto* mutableInnerRows = innerRowsHolder.get(rows.size());
    numInnerRows =
        simd::indicesOfSetBits(nulls, 0, rows.size(), mutableInnerRows);
    stream->appendLengths(nulls, rows, numInnerRows, [](int32_t) { return 1; });
    simd::translate(
        rows.data(),
        folly::Range<const vector_size_t*>(mutableInnerRows, numInnerRows),
        mutableInnerRows);
    innerRows = mutableInnerRows;
  } else {
    stream->appendLengths(
        nullptr, rows, rows.size(), [](int32_t) { return 1; });
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
  if (numRanges == 0) {
    return;
  }
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
  auto mapVector = reinterpret_cast<const MapVector*>(vector);

  ScratchPtr<IndexRange> rangesHolder(scratch);
  int32_t numRanges = rowsToRanges(
      rows,
      mapVector->rawNulls(),
      mapVector->rawOffsets(),
      mapVector->rawSizes(),
      nullptr,
      rangesHolder,
      nullptr,
      stream,
      scratch);
  if (numRanges == 0) {
    return;
  }
  serializeColumn(
      mapVector->mapKeys().get(),
      folly::Range<const IndexRange*>(rangesHolder.get(), numRanges),
      stream->childAt(0));
  serializeColumn(
      mapVector->mapValues().get(),
      folly::Range<const IndexRange*>(rangesHolder.get(), numRanges),
      stream->childAt(1));
}

template <TypeKind kind>
void serializeConstantVector(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    VectorStream* stream,
    Scratch& scratch) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  auto constVector = dynamic_cast<const ConstantVector<T>*>(vector);
  if (constVector->valueVector()) {
    serializeWrapped(constVector, rows, stream, scratch);
    return;
  }
  const auto numRows = rows.size();
  if (vector->isNullAt(0)) {
    for (int32_t i = 0; i < numRows; ++i) {
      stream->appendNull();
    }
    return;
  }

  T value = constVector->valueAtFast(0);
  for (int32_t i = 0; i < numRows; ++i) {
    stream->appendNonNull();
    stream->appendOne(value);
  }
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
