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

#include "velox/dwio/dwrf/reader/SelectiveColumnReaderInternal.h"
#include "velox/dwio/dwrf/reader/SelectiveByteRleColumnReader.h"

namespace facebook::velox::dwrf {

using common::FilterKind;
using dwio::common::TypeWithId;
using dwio::common::typeutils::CompatChecker;
using namespace facebook::velox::aggregate;
using V64 = simd::Vectors<int64_t>;
using V32 = simd::Vectors<int32_t>;
using V16 = simd::Vectors<int16_t>;
using V8 = simd::Vectors<int8_t>;

SelectiveColumnReader::SelectiveColumnReader(
    std::shared_ptr<const TypeWithId> requestedType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    // TODO: why is data type instead of requested type passed in?
    const TypePtr& type,
    FlatMapContext flatMapContext)
    : ColumnReader(std::move(requestedType), stripe, std::move(flatMapContext)),
      scanSpec_(scanSpec),
      type_{type},
      rowsPerRowGroup_{stripe.rowsPerRowGroup()} {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  // We always initialize indexStream_ because indices are needed as
  // soon as there is a single filter that can trigger row group skips
  // anywhere in the reader tree. This is not known at construct time
  // because the first filter can come from a hash join or other run
  // time pushdown.
  indexStream_ = stripe.getStream(
      encodingKey.forKind(proto::Stream_Kind_ROW_INDEX), false);
}

std::vector<uint32_t> SelectiveColumnReader::filterRowGroups(
    uint64_t rowGroupSize,
    const StatsContext& context) const {
  if ((!index_ && !indexStream_) || !scanSpec_->filter()) {
    return ColumnReader::filterRowGroups(rowGroupSize, context);
  }

  ensureRowGroupIndex();
  auto filter = scanSpec_->filter();

  std::vector<uint32_t> stridesToSkip;
  for (auto i = 0; i < index_->entry_size(); i++) {
    const auto& entry = index_->entry(i);
    auto columnStats =
        buildColumnStatisticsFromProto(entry.statistics(), context);
    if (!testFilter(filter, columnStats.get(), rowGroupSize, type_)) {
      stridesToSkip.push_back(i); // Skipping stride based on column stats.
    }
  }
  return stridesToSkip;
}

void SelectiveColumnReader::seekTo(vector_size_t offset, bool readsNullsOnly) {
  if (offset == readOffset_) {
    return;
  }
  if (readOffset_ < offset) {
    if (readsNullsOnly) {
      ColumnReader::skip(offset - readOffset_);
    } else {
      skip(offset - readOffset_);
    }
    readOffset_ = offset;
  } else {
    VELOX_FAIL("Seeking backward on a ColumnReader");
  }
}

void SelectiveColumnReader::prepareNulls(RowSet rows, bool hasNulls) {
  if (!hasNulls) {
    anyNulls_ = false;
    return;
  }
  auto numRows = rows.size();
  if (useBulkPath()) {
    bool isDense = rows.back() == rows.size() - 1;
    if (!scanSpec_->filter()) {
      anyNulls_ = nullsInReadRange_ != nullptr;
      returnReaderNulls_ = anyNulls_ && isDense;
      // No need for null flags if fast path
      if (returnReaderNulls_) {
        return;
      }
    }
  }
  if (resultNulls_ && resultNulls_->unique() &&
      resultNulls_->capacity() >= bits::nbytes(numRows) + simd::kPadding) {
    // Clear whole capacity because future uses could hit
    // uncleared data between capacity() and 'numBytes'.
    simd::memset(rawResultNulls_, bits::kNotNullByte, resultNulls_->capacity());
    anyNulls_ = false;
    return;
  }

  anyNulls_ = false;
  resultNulls_ = AlignedBuffer::allocate<bool>(
      numRows + (simd::kPadding * 8), &memoryPool_);
  rawResultNulls_ = resultNulls_->asMutable<uint64_t>();
  simd::memset(rawResultNulls_, bits::kNotNullByte, resultNulls_->capacity());
}


bool SelectiveColumnReader::shouldMoveNulls(RowSet rows) {
  if (rows.size() == numValues_) {
    // Nulls will only be moved if there is a selection on values. A cast alone
    // does not move nulls.
    return false;
  }
  VELOX_CHECK(
      !returnReaderNulls_,
      "Do not return reader nulls if retrieving a subset of values");
  if (anyNulls_) {
    VELOX_CHECK(
        resultNulls_ && resultNulls_->as<uint64_t>() == rawResultNulls_);
    VELOX_CHECK_GT(resultNulls_->capacity() * 8, rows.size());
    return true;
  }
  return false;
}

void SelectiveColumnReader::getIntValues(
    RowSet rows,
    const Type* requestedType,
    VectorPtr* result) {
  switch (requestedType->kind()) {
    case TypeKind::SMALLINT: {
      switch (valueSize_) {
        case 8:
          getFlatValues<int64_t, int16_t>(rows, result);
          break;
        case 4:
          getFlatValues<int32_t, int16_t>(rows, result);
          break;
        case 2:
          getFlatValues<int16_t, int16_t>(rows, result);
          break;
        default:
          VELOX_FAIL("Unsupported value size");
      }
      break;
      case TypeKind::INTEGER:
        switch (valueSize_) {
          case 8:
            getFlatValues<int64_t, int32_t>(rows, result);
            break;
          case 4:
            getFlatValues<int32_t, int32_t>(rows, result);
            break;
          case 2:
            getFlatValues<int16_t, int32_t>(rows, result);
            break;
          default:
            VELOX_FAIL("Unsupported value size");
        }
        break;
      case TypeKind::BIGINT:
        switch (valueSize_) {
          case 8:
            getFlatValues<int64_t, int64_t>(rows, result);
            break;
          case 4:
            getFlatValues<int32_t, int64_t>(rows, result);
            break;
          case 2:
            getFlatValues<int16_t, int64_t>(rows, result);
            break;
          default:
            VELOX_FAIL("Unsupported value size");
        }
        break;
      default:
        VELOX_FAIL(
            "Not a valid type for integer reader: {}",
            requestedType->toString());
    }
  }
}

template <>
void SelectiveColumnReader::getFlatValues<int8_t, bool>(
    RowSet rows,
    VectorPtr* result,
    const TypePtr& type,
    bool isFinal) {
  using V8 = simd::Vectors<int8_t>;
  constexpr int32_t kWidth = V8::VSize;
  static_assert(kWidth == 32);
  VELOX_CHECK_EQ(valueSize_, sizeof(int8_t));
  compactScalarValues<int8_t, int8_t>(rows, isFinal);
  auto boolValues =
      AlignedBuffer::allocate<bool>(numValues_, &memoryPool_, false);
  auto rawBits = boolValues->asMutable<uint32_t>();
  auto rawBytes = values_->as<int8_t>();
  auto zero = V8::setAll(0);
  for (auto i = 0; i < numValues_; i += kWidth) {
    rawBits[i / kWidth] = ~V8::compareBitMask(
        V8::compareResult(V8::compareEq(zero, V8::load(rawBytes + i))));
  }
  BufferPtr nulls = anyNulls_
      ? (returnReaderNulls_ ? nullsInReadRange_ : resultNulls_)
      : nullptr;
  *result = std::make_shared<FlatVector<bool>>(
      &memoryPool_,
      type,
      nulls,
      numValues_,
      std::move(boolValues),
      std::move(stringBuffers_));
}

  
template <>
void SelectiveColumnReader::compactScalarValues<bool, bool>(
    RowSet rows,
    bool isFinal) {
  if (!values_ || rows.size() == numValues_) {
    if (values_) {
      values_->setSize(bits::nbytes(numValues_));
    }
    return;
  }
  auto rawBits = reinterpret_cast<uint64_t*>(rawValues_);
  vector_size_t rowIndex = 0;
  auto nextRow = rows[rowIndex];
  bool moveNulls = shouldMoveNulls(rows);
  for (size_t i = 0; i < numValues_; i++) {
    if (outputRows_[i] < nextRow) {
      continue;
    }

    VELOX_DCHECK(outputRows_[i] == nextRow);

    bits::setBit(rawBits, rowIndex, bits::isBitSet(rawBits, i));
    if (moveNulls && rowIndex != i) {
      bits::setBit(
          rawResultNulls_, rowIndex, bits::isBitSet(rawResultNulls_, i));
    }
    if (!isFinal) {
      outputRows_[rowIndex] = nextRow;
    }
    rowIndex++;
    if (rowIndex >= rows.size()) {
      break;
    }
    nextRow = rows[rowIndex];
  }
  numValues_ = rows.size();
  outputRows_.resize(numValues_);
  values_->setSize(bits::nbytes(numValues_));
}

common::AlwaysTrue Filters::alwaysTrue;

char* SelectiveColumnReader::copyStringValue(folly::StringPiece value) {
  uint64_t size = value.size();
  if (stringBuffers_.empty() || rawStringUsed_ + size > rawStringSize_) {
    if (!stringBuffers_.empty()) {
      stringBuffers_.back()->setSize(rawStringUsed_);
    }
    auto bytes = std::max(size, kStringBufferSize);
    BufferPtr buffer = AlignedBuffer::allocate<char>(bytes, &memoryPool_);
    stringBuffers_.push_back(buffer);
    rawStringBuffer_ = buffer->asMutable<char>();
    rawStringUsed_ = 0;
    // Adjust the size downward so that the last store can take place
    // at full width.
    rawStringSize_ = buffer->capacity() - simd::kPadding;
  }
  memcpy(rawStringBuffer_ + rawStringUsed_, value.data(), size);
  auto start = rawStringUsed_;
  rawStringUsed_ += size;
  return rawStringBuffer_ + start;
}

void SelectiveColumnReader::addStringValue(folly::StringPiece value) {
  auto copy = copyStringValue(value);
  reinterpret_cast<StringView*>(rawValues_)[numValues_++] =
      StringView(copy, value.size());
}

std::vector<uint64_t> toPositions(const proto::RowIndexEntry& entry) {
  return std::vector<uint64_t>(
      entry.positions().begin(), entry.positions().end());
}

class SelectiveIntegerDirectColumnReader : public SelectiveColumnReader {
 public:
  using ValueType = int64_t;

  SelectiveIntegerDirectColumnReader(
      std::shared_ptr<const TypeWithId> requestedType,
      const std::shared_ptr<const TypeWithId>& dataType,
      StripeStreams& stripe,
      uint32_t numBytes,
      common::ScanSpec* scanSpec)
      : SelectiveColumnReader(
            std::move(requestedType),
            stripe,
            scanSpec,
            dataType->type) {
    EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
    auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
    bool dataVInts = stripe.getUseVInts(data);
    auto decoder = IntDecoder</*isSigned*/ true>::createDirect(
        stripe.getStream(data, true), dataVInts, numBytes);
    auto rawDecoder = decoder.release();
    auto directDecoder = dynamic_cast<DirectDecoder<true>*>(rawDecoder);
    ints.reset(directDecoder);
  }

  bool hasBulkPath() const override {
    return true;
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    ints->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

 private:
  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      ExtractValues extractValues,
      RowSet rows);

  template <bool isDence>
  void processValueHook(RowSet rows, ValueHook* hook);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void
  readHelper(common::Filter* filter, RowSet rows, ExtractValues extractValues);

  void getValues(RowSet rows, VectorPtr* result) override {
    getIntValues(rows, nodeType_->type.get(), result);
  }

  std::unique_ptr<DirectDecoder</*isSigned*/ true>> ints;
};

uint64_t SelectiveIntegerDirectColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  ints->skip(numValues);
  return numValues;
}

template <typename ColumnVisitor>
void SelectiveIntegerDirectColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  if (nullsInReadRange_) {
    ints->readWithVisitor<true>(nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    ints->readWithVisitor<false>(nullptr, visitor);
  }
  readOffset_ += numRows;
}

template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveIntegerDirectColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (valueSize_) {
    case 2:
      readWithVisitor(
          rows,
          ColumnVisitor<int16_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
      break;

    case 4:
      readWithVisitor(
          rows,
          ColumnVisitor<int32_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));

      break;

    case 8:
      readWithVisitor(
          rows,
          ColumnVisitor<int64_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
      break;
    default:
      VELOX_FAIL("Unsupported valueSize_ {}", valueSize_);
  }
}

template <bool isDense, typename ExtractValues>
void SelectiveIntegerDirectColumnReader::processFilter(
    common::Filter* filter,
    ExtractValues extractValues,
    RowSet rows) {
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kIsNull:
      filterNulls<int64_t>(
          rows,
          true,
          !std::is_same<decltype(extractValues), DropValues>::value);
      break;
    case FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), DropValues>::value) {
        filterNulls<int64_t>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case FilterKind::kBigintRange:
      readHelper<common::BigintRange, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kBigintValuesUsingHashTable:
      readHelper<common::BigintValuesUsingHashTable, isDense>(
          filter, rows, extractValues);
      break;
    case FilterKind::kBigintValuesUsingBitmask:
      readHelper<common::BigintValuesUsingBitmask, isDense>(
          filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

template <bool isDense>
void SelectiveIntegerDirectColumnReader::processValueHook(
    RowSet rows,
    ValueHook* hook) {
  switch (hook->kind()) {
    case AggregationHook::kSumBigintToBigint:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<SumHook<int64_t, int64_t>>(hook));
      break;
    case AggregationHook::kBigintMax:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<MinMaxHook<int64_t, false>>(hook));
      break;
    case AggregationHook::kBigintMin:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<MinMaxHook<int64_t, true>>(hook));
      break;
    default:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue, rows, ExtractToGenericHook(hook));
  }
}

void SelectiveIntegerDirectColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  VELOX_WIDTH_DISPATCH(
      sizeOfIntKind(type_->kind()), prepareRead, offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;
  common::Filter* filter =
      scanSpec_->filter() ? scanSpec_->filter() : &Filters::alwaysTrue;
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        processValueHook<false>(rows, scanSpec_->valueHook());
      }
      return;
    }
    if (isDense) {
      processFilter<true>(filter, ExtractToReader(this), rows);
    } else {
      processFilter<false>(filter, ExtractToReader(this), rows);
    }
  } else {
    if (isDense) {
      processFilter<true>(filter, DropValues(), rows);
    } else {
      processFilter<false>(filter, DropValues(), rows);
    }
  }
}

class SelectiveIntegerDictionaryColumnReader : public SelectiveColumnReader {
 public:
  using ValueType = int64_t;

  SelectiveIntegerDictionaryColumnReader(
      std::shared_ptr<const TypeWithId> requestedType,
      const std::shared_ptr<const TypeWithId>& dataType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      uint32_t numBytes);

  void resetFilterCaches() override {
    if (!filterCache_.empty()) {
      simd::memset(
          filterCache_.data(), FilterResult::kUnknown, dictionarySize_);
    }
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    if (inDictionaryReader_) {
      inDictionaryReader_->seekToRowGroup(positionsProvider);
    }

    dataReader_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override {
    getIntValues(rows, nodeType_->type.get(), result);
  }

 private:
  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      ExtractValues extractValues,
      RowSet rows);

  template <bool isDence>
  void processValueHook(RowSet rows, ValueHook* hook);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void
  readHelper(common::Filter* filter, RowSet rows, ExtractValues extractValues);

  void ensureInitialized();

  BufferPtr dictionary_;
  BufferPtr inDictionary_;
  std::unique_ptr<ByteRleDecoder> inDictionaryReader_;
  std::unique_ptr<IntDecoder</* isSigned = */ false>> dataReader_;
  uint64_t dictionarySize_;
  std::unique_ptr<IntDecoder</* isSigned = */ true>> dictReader_;
  std::function<BufferPtr()> dictInit_;
  raw_vector<uint8_t> filterCache_;
  RleVersion rleVersion_;
  bool initialized_{false};
};

SelectiveIntegerDictionaryColumnReader::SelectiveIntegerDictionaryColumnReader(
    std::shared_ptr<const TypeWithId> requestedType,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    uint32_t numBytes)
    : SelectiveColumnReader(
          std::move(requestedType),
          stripe,
          scanSpec,
          dataType->type) {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  auto encoding = stripe.getEncoding(encodingKey);
  dictionarySize_ = encoding.dictionarysize();
  rleVersion_ = convertRleVersion(encoding.kind());
  auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
  bool dataVInts = stripe.getUseVInts(data);
  dataReader_ = IntDecoder</* isSigned = */ false>::createRle(
      stripe.getStream(data, true),
      rleVersion_,
      memoryPool_,
      dataVInts,
      numBytes);

  // make a lazy dictionary initializer
  dictInit_ = stripe.getIntDictionaryInitializerForNode(
      encodingKey, numBytes, numBytes);

  auto inDictStream = stripe.getStream(
      encodingKey.forKind(proto::Stream_Kind_IN_DICTIONARY), false);
  if (inDictStream) {
    inDictionaryReader_ =
        createBooleanRleDecoder(std::move(inDictStream), encodingKey);
  }
}

uint64_t SelectiveIntegerDictionaryColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  dataReader_->skip(numValues);
  if (inDictionaryReader_) {
    inDictionaryReader_->skip(numValues);
  }
  return numValues;
}

template <typename ColumnVisitor>
void SelectiveIntegerDictionaryColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  VELOX_CHECK_EQ(rleVersion_, RleVersion_1);
  auto reader = reinterpret_cast<RleDecoderV1<false>*>(dataReader_.get());
  if (nullsInReadRange_) {
    reader->readWithVisitor<true>(nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    reader->readWithVisitor<false>(nullptr, visitor);
  }
  readOffset_ += numRows;
}

template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveIntegerDictionaryColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (valueSize_) {
    case 2:
      readWithVisitor(
          rows,
          DictionaryColumnVisitor<int16_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter),
              this,
              rows,
              extractValues,
              dictionary_->as<int16_t>(),
              inDictionary_ ? inDictionary_->as<uint64_t>() : nullptr,
              filterCache_.data()));
      break;
    case 4:
      readWithVisitor(
          rows,
          DictionaryColumnVisitor<int32_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter),
              this,
              rows,
              extractValues,
              dictionary_->as<int32_t>(),
              inDictionary_ ? inDictionary_->as<uint64_t>() : nullptr,
              filterCache_.data()));
      break;

    case 8:
      readWithVisitor(
          rows,
          DictionaryColumnVisitor<int64_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter),
              this,
              rows,
              extractValues,
              dictionary_->as<int64_t>(),
              inDictionary_ ? inDictionary_->as<uint64_t>() : nullptr,
              filterCache_.data()));
      break;

    default:
      VELOX_FAIL("Unsupported valueSize_ {}", valueSize_);
  }
}

template <bool isDense, typename ExtractValues>
void SelectiveIntegerDictionaryColumnReader::processFilter(
    common::Filter* filter,
    ExtractValues extractValues,
    RowSet rows) {
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kIsNull:
      filterNulls<int64_t>(
          rows,
          true,
          !std::is_same<decltype(extractValues), DropValues>::value);
      break;
    case FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), DropValues>::value) {
        filterNulls<int64_t>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case FilterKind::kBigintRange:
      readHelper<common::BigintRange, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kBigintValuesUsingHashTable:
      readHelper<common::BigintValuesUsingHashTable, isDense>(
          filter, rows, extractValues);
      break;
    case FilterKind::kBigintValuesUsingBitmask:
      readHelper<common::BigintValuesUsingBitmask, isDense>(
          filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

template <bool isDense>
void SelectiveIntegerDictionaryColumnReader::processValueHook(
    RowSet rows,
    ValueHook* hook) {
  switch (hook->kind()) {
    case AggregationHook::kSumBigintToBigint:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<SumHook<int64_t, int64_t>>(hook));
      break;
    default:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue, rows, ExtractToGenericHook(hook));
  }
}

void SelectiveIntegerDictionaryColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  VELOX_WIDTH_DISPATCH(
      sizeOfIntKind(type_->kind()), prepareRead, offset, rows, incomingNulls);
  auto end = rows.back() + 1;
  const auto* rawNulls =
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;

  // read the stream of booleans indicating whether a given data entry
  // is an offset or a literal value.
  if (inDictionaryReader_) {
    bool isBulk = useBulkPath();
    int32_t numFlags = (isBulk && nullsInReadRange_)
        ? bits::countNonNulls(nullsInReadRange_->as<uint64_t>(), 0, end)
        : end;
    ensureCapacity<uint64_t>(
        inDictionary_, bits::nwords(numFlags), &memoryPool_);
    inDictionaryReader_->next(
        inDictionary_->asMutable<char>(),
        numFlags,
        isBulk ? nullptr : rawNulls);
  }

  // lazy load dictionary only when it's needed
  ensureInitialized();

  bool isDense = rows.back() == rows.size() - 1;
  common::Filter* filter = scanSpec_->filter();
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        processValueHook<false>(rows, scanSpec_->valueHook());
      }
      return;
    }
    if (isDense) {
      processFilter<true>(filter, ExtractToReader(this), rows);
    } else {
      processFilter<false>(filter, ExtractToReader(this), rows);
    }
  } else {
    if (isDense) {
      processFilter<true>(filter, DropValues(), rows);
    } else {
      processFilter<false>(filter, DropValues(), rows);
    }
  }
}

void SelectiveIntegerDictionaryColumnReader::ensureInitialized() {
  if (LIKELY(initialized_)) {
    return;
  }

  Timer timer;
  dictionary_ = dictInit_();
  // Make sure there is a cache even for an empty dictionary because
  // of asan failure when preparing a gather with all lanes masked
  // out.
  filterCache_.resize(std::max<int32_t>(1, dictionarySize_));
  simd::memset(filterCache_.data(), FilterResult::kUnknown, dictionarySize_);
  initialized_ = true;
  initTimeClocks_ = timer.elapsedClocks();
}

template <typename TData, typename TRequested>
class SelectiveFloatingPointColumnReader : public SelectiveColumnReader {
 public:
  using ValueType = TRequested;
  SelectiveFloatingPointColumnReader(
      std::shared_ptr<const TypeWithId> nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  // Offers fast path only if data and result widths match.
  bool hasBulkPath() const override {
    return std::is_same<TData, TRequested>::value;
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    decoder_.seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;
  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override {
    getFlatValues<TRequested, TRequested>(rows, result);
  }

 private:
  template <typename TVisitor>
  void readWithVisitor(RowSet rows, TVisitor visitor);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void readHelper(common::Filter* filter, RowSet rows, ExtractValues values);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      RowSet rows,
      ExtractValues extractValues);

  template <bool isDense>
  void processValueHook(RowSet rows, ValueHook* hook);

  FloatingPointDecoder<TData, TRequested> decoder_;
};

template <typename TData, typename TRequested>
SelectiveFloatingPointColumnReader<TData, TRequested>::
    SelectiveFloatingPointColumnReader(
        std::shared_ptr<const TypeWithId> requestedType,
        StripeStreams& stripe,
        common::ScanSpec* scanSpec,
        FlatMapContext flatMapContext)
    : SelectiveColumnReader(
          std::move(requestedType),
          stripe,
          scanSpec,
          CppToType<TData>::create(),
          std::move(flatMapContext)),
      decoder_(stripe.getStream(
          EncodingKey{nodeType_->id, flatMapContext_.sequence}.forKind(
              proto::Stream_Kind_DATA),
          true)) {}

template <typename TData, typename TRequested>
uint64_t SelectiveFloatingPointColumnReader<TData, TRequested>::skip(
    uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  decoder_.skip(numValues);
  return numValues;
}

template <typename TData, typename TRequested>
template <typename TVisitor>
void SelectiveFloatingPointColumnReader<TData, TRequested>::readWithVisitor(
    RowSet rows,
    TVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  if (nullsInReadRange_) {
    decoder_.template readWithVisitor<true, TVisitor>(
        SelectiveColumnReader::nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    decoder_.template readWithVisitor<false, TVisitor>(nullptr, visitor);
  }
  readOffset_ += numRows;
}

template <typename TData, typename TRequested>
template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveFloatingPointColumnReader<TData, TRequested>::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  readWithVisitor(
      rows,
      ColumnVisitor<TRequested, TFilter, ExtractValues, isDense>(
          *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
}

template <typename TData, typename TRequested>
template <bool isDense, typename ExtractValues>
void SelectiveFloatingPointColumnReader<TData, TRequested>::processFilter(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kIsNull:
      filterNulls<TRequested>(
          rows,
          true,
          !std::is_same<decltype(extractValues), DropValues>::value);
      break;
    case FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), DropValues>::value) {
        filterNulls<TRequested>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case FilterKind::kDoubleRange:
    case FilterKind::kFloatRange:
      readHelper<common::FloatingPointRange<TData>, isDense>(
          filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

template <typename TData, typename TRequested>
template <bool isDense>
void SelectiveFloatingPointColumnReader<TData, TRequested>::processValueHook(
    RowSet rows,
    ValueHook* hook) {
  switch (hook->kind()) {
    case AggregationHook::kSumFloatToDouble:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<SumHook<float, double>>(hook));
      break;
    case AggregationHook::kSumDoubleToDouble:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<SumHook<double, double>>(hook));
      break;
    case AggregationHook::kFloatMax:
    case AggregationHook::kDoubleMax:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<MinMaxHook<TRequested, false>>(hook));
      break;
    case AggregationHook::kFloatMin:
    case AggregationHook::kDoubleMin:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue,
          rows,
          ExtractToHook<MinMaxHook<TRequested, true>>(hook));
      break;
    default:
      readHelper<common::AlwaysTrue, isDense>(
          &Filters::alwaysTrue, rows, ExtractToGenericHook(hook));
  }
}

template <typename TData, typename TRequested>
void SelectiveFloatingPointColumnReader<TData, TRequested>::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  prepareRead<TRequested>(offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        processValueHook<false>(rows, scanSpec_->valueHook());
      }
      return;
    }
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, ExtractToReader(this));
    } else {
      processFilter<false>(scanSpec_->filter(), rows, ExtractToReader(this));
    }
  } else {
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, DropValues());
    } else {
      processFilter<false>(scanSpec_->filter(), rows, DropValues());
    }
  }
}

class SelectiveStringDirectColumnReader : public SelectiveColumnReader {
 public:
  using ValueType = StringView;
  SelectiveStringDirectColumnReader(
      const std::shared_ptr<const TypeWithId>& nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    blobStream_->seekToRowGroup(positionsProvider);
    lengthDecoder_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());

    bytesToSkip_ = 0;
    bufferStart_ = bufferEnd_;
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override {
    rawStringBuffer_ = nullptr;
    rawStringSize_ = 0;
    rawStringUsed_ = 0;
    getFlatValues<StringView, StringView>(rows, result, type_);
  }

 private:
  template <bool hasNulls>
  void skipInDecode(int32_t numValues, int32_t current, const uint64_t* nulls);

  folly::StringPiece readValue(int32_t length);

  template <bool hasNulls, typename Visitor>
  void decode(const uint64_t* nulls, Visitor visitor);

  template <typename TVisitor>
  void readWithVisitor(RowSet rows, TVisitor visitor);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void readHelper(common::Filter* filter, RowSet rows, ExtractValues values);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      RowSet rows,
      ExtractValues extractValues);

  void extractCrossBuffers(
      const int32_t* lengths,
      const int32_t* starts,
      int32_t rowIndex,
      int32_t numValues);

  inline void makeSparseStarts(
      int32_t startRow,
      const int32_t* rows,
      int32_t numRows,
      int32_t* starts);

  inline void extractNSparse(const int32_t* rows, int32_t row, int numRows);

  void extractSparse(const int32_t* rows, int32_t numRows);

  template <bool scatter, bool skip>
  bool try8Consecutive(int32_t start, const int32_t* rows, int32_t row);

  std::unique_ptr<IntDecoder</*isSigned*/ false>> lengthDecoder_;
  std::unique_ptr<SeekableInputStream> blobStream_;
  const char* bufferStart_ = nullptr;
  const char* bufferEnd_ = nullptr;
  BufferPtr lengths_;
  int32_t lengthIndex_ = 0;
  const uint32_t* rawLengths_ = nullptr;
  int64_t bytesToSkip_ = 0;
  // Storage for a string straddling a buffer boundary. Needed for calling
  // the filter.
  std::string tempString_;
};

SelectiveStringDirectColumnReader::SelectiveStringDirectColumnReader(
    const std::shared_ptr<const TypeWithId>& nodeType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveColumnReader(
          nodeType,
          stripe,
          scanSpec,
          nodeType->type,
          std::move(flatMapContext)) {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  RleVersion rleVersion =
      convertRleVersion(stripe.getEncoding(encodingKey).kind());
  auto lenId = encodingKey.forKind(proto::Stream_Kind_LENGTH);
  bool lenVInts = stripe.getUseVInts(lenId);
  lengthDecoder_ = IntDecoder</*isSigned*/ false>::createRle(
      stripe.getStream(lenId, true),
      rleVersion,
      memoryPool_,
      lenVInts,
      INT_BYTE_SIZE);
  blobStream_ =
      stripe.getStream(encodingKey.forKind(proto::Stream_Kind_DATA), true);
}

uint64_t SelectiveStringDirectColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  ensureCapacity<int64_t>(lengths_, numValues, &memoryPool_);
  lengthDecoder_->nextLengths(lengths_->asMutable<int32_t>(), numValues);
  rawLengths_ = lengths_->as<uint32_t>();
  for (auto i = 0; i < numValues; ++i) {
    bytesToSkip_ += rawLengths_[i];
  }
  skipBytes(bytesToSkip_, blobStream_.get(), bufferStart_, bufferEnd_);
  bytesToSkip_ = 0;
  return numValues;
}

void SelectiveStringDirectColumnReader::extractCrossBuffers(
    const int32_t* lengths,
    const int32_t* starts,
    int32_t rowIndex,
    int32_t numValues) {
  int32_t current = 0;
  bool scatter = !outerNonNullRows_.empty();
  for (auto i = 0; i < numValues; ++i) {
    auto gap = starts[i] - current;
    bytesToSkip_ += gap;
    auto size = lengths[i];
    auto value = readValue(size);
    current += size + gap;
    if (!scatter) {
      addValue(value);
    } else {
      auto index = outerNonNullRows_[rowIndex + i];
      if (size <= StringView::kInlineSize) {
        reinterpret_cast<StringView*>(rawValues_)[index] =
            StringView(value.data(), size);
      } else {
        auto copy = copyStringValue(value);
        reinterpret_cast<StringView*>(rawValues_)[index] =
            StringView(copy, size);
      }
    }
  }
  skipBytes(bytesToSkip_, blobStream_.get(), bufferStart_, bufferEnd_);
  bytesToSkip_ = 0;
  if (scatter) {
    numValues_ = outerNonNullRows_[rowIndex + numValues - 1] + 1;
  }
}

inline int32_t
rangeSum(const uint32_t* rows, int32_t start, int32_t begin, int32_t end) {
  for (auto i = begin; i < end; ++i) {
    start += rows[i];
  }
  return start;
}

inline void SelectiveStringDirectColumnReader::makeSparseStarts(
    int32_t startRow,
    const int32_t* rows,
    int32_t numRows,
    int32_t* starts) {
  auto previousRow = lengthIndex_;
  int32_t i = 0;
  int32_t startOffset = 0;
  for (; i < numRows; ++i) {
    int targetRow = rows[startRow + i];
    startOffset = rangeSum(rawLengths_, startOffset, previousRow, targetRow);
    starts[i] = startOffset;
    previousRow = targetRow + 1;
    startOffset += rawLengths_[targetRow];
  }
}

void SelectiveStringDirectColumnReader::extractNSparse(
    const int32_t* rows,
    int32_t row,
    int32_t numValues) {
  int32_t starts[8];
  if (numValues == 8 &&
      (outerNonNullRows_.empty() ? try8Consecutive<false, true>(0, rows, row)
                                 : try8Consecutive<true, true>(0, rows, row))) {
    return;
  }
  int32_t lengths[8];
  for (auto i = 0; i < numValues; ++i) {
    lengths[i] = rawLengths_[rows[row + i]];
  }
  makeSparseStarts(row, rows, numValues, starts);
  extractCrossBuffers(lengths, starts, row, numValues);
  lengthIndex_ = rows[row + numValues - 1] + 1;
}

template <bool scatter, bool sparse>
inline bool SelectiveStringDirectColumnReader::try8Consecutive(
    int32_t start,
    const int32_t* rows,
    int32_t row) {
  const char* data = bufferStart_ + start + bytesToSkip_;
  if (!data || bufferEnd_ - data < start + 8 * 12) {
    return false;
  }
  int32_t* result = reinterpret_cast<int32_t*>(rawValues_);
  int32_t resultIndex = numValues_ * 4 - 4;
  auto rawUsed = rawStringUsed_;
  auto previousRow = sparse ? lengthIndex_ : 0;
  auto endRow = row + 8;
  for (auto i = row; i < endRow; ++i) {
    if (scatter) {
      resultIndex = outerNonNullRows_[i] * 4;
    } else {
      resultIndex += 4;
    }
    if (sparse) {
      auto targetRow = rows[i];
      data += rangeSum(rawLengths_, 0, previousRow, rows[i]);
      previousRow = targetRow + 1;
    }
    auto length = rawLengths_[rows[i]];

    if (data + bits::roundUp(length, 16) > bufferEnd_) {
      // Slow path if the string does not fit whole or if there is no
      // space for a 16 byte load.
      return false;
    }
    result[resultIndex] = length;
    auto first16 = *reinterpret_cast<const __m128_u*>(data);
    *reinterpret_cast<__m128_u*>(result + resultIndex + 1) = first16;
    if (length <= 12) {
      data += length;
      *reinterpret_cast<int64_t*>(
          reinterpret_cast<char*>(result + resultIndex + 1) + length) = 0;
      continue;
    }
    if (!rawStringBuffer_ || rawUsed + length > rawStringSize_) {
      // Slow path if no space in raw strings
      return false;
    }
    *reinterpret_cast<char**>(result + resultIndex + 2) =
        rawStringBuffer_ + rawUsed;
    *reinterpret_cast<__m128_u*>(rawStringBuffer_ + rawUsed) = first16;
    if (length > 16) {
      size_t copySize = bits::roundUp(length - 16, 16);
      VELOX_CHECK_LE(data + copySize, bufferEnd_);
      simd::memcpy(rawStringBuffer_ + rawUsed + 16, data + 16, copySize);
    }
    rawUsed += length;
    data += length;
  }
  // Update the data members only after successful completion.
  bufferStart_ = data;
  bytesToSkip_ = 0;
  rawStringUsed_ = rawUsed;
  numValues_ = scatter ? outerNonNullRows_[row + 7] + 1 : numValues_ + 8;
  lengthIndex_ = sparse ? rows[row + 7] + 1 : lengthIndex_ + 8;
  return true;
}

void SelectiveStringDirectColumnReader::extractSparse(
    const int32_t* rows,
    int32_t numRows) {
  rowLoop(
      rows,
      0,
      numRows,
      8,
      [&](int32_t row) {
        int32_t start = rangeSum(rawLengths_, 0, lengthIndex_, rows[row]);
        lengthIndex_ = rows[row];
        auto lengths =
            reinterpret_cast<const int32_t*>(rawLengths_) + lengthIndex_;

        if (outerNonNullRows_.empty()
                ? try8Consecutive<false, false>(start, rows, row)
                : try8Consecutive<true, false>(start, rows, row)) {
          return;
        }
        int32_t starts[8];
        for (auto i = 0; i < 8; ++i) {
          starts[i] = start;
          start += lengths[i];
        }
        lengthIndex_ += 8;
        extractCrossBuffers(lengths, starts, row, 8);
      },
      [&](int32_t row) { extractNSparse(rows, row, 8); },
      [&](int32_t row, int32_t numRows) {
        extractNSparse(rows, row, numRows);
      });
}

template <bool hasNulls>
void SelectiveStringDirectColumnReader::skipInDecode(
    int32_t numValues,
    int32_t current,
    const uint64_t* nulls) {
  if (hasNulls) {
    numValues = bits::countNonNulls(nulls, current, current + numValues);
  }
  for (size_t i = lengthIndex_; i < lengthIndex_ + numValues; ++i) {
    bytesToSkip_ += rawLengths_[i];
  }
  lengthIndex_ += numValues;
}

folly::StringPiece SelectiveStringDirectColumnReader::readValue(
    int32_t length) {
  skipBytes(bytesToSkip_, blobStream_.get(), bufferStart_, bufferEnd_);
  bytesToSkip_ = 0;
  if (bufferStart_ + length <= bufferEnd_) {
    bytesToSkip_ = length;
    return folly::StringPiece(bufferStart_, length);
  }
  tempString_.resize(length);
  readBytes(
      length, blobStream_.get(), tempString_.data(), bufferStart_, bufferEnd_);
  return folly::StringPiece(tempString_);
}

template <bool hasNulls, typename Visitor>
void SelectiveStringDirectColumnReader::decode(
    const uint64_t* nulls,
    Visitor visitor) {
  int32_t current = visitor.start();
  bool atEnd = false;
  bool allowNulls = hasNulls && visitor.allowNulls();
  for (;;) {
    int32_t toSkip;
    if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
      toSkip = visitor.processNull(atEnd);
    } else {
      if (hasNulls && !allowNulls) {
        toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
        if (!Visitor::dense) {
          skipInDecode<false>(toSkip, current, nullptr);
        }
        if (atEnd) {
          return;
        }
      }

      // Check if length passes the filter first. Don't read the value if length
      // doesn't pass.
      auto length = rawLengths_[lengthIndex_++];
      auto toSkipOptional = visitor.processLength(length, atEnd);
      if (toSkipOptional.has_value()) {
        bytesToSkip_ += length;
        toSkip = toSkipOptional.value();
      } else {
        toSkip = visitor.process(readValue(length), atEnd);
      }
    }
    ++current;
    if (toSkip) {
      skipInDecode<hasNulls>(toSkip, current, nulls);
      current += toSkip;
    }
    if (atEnd) {
      return;
    }
  }
}

template <typename TVisitor>
void SelectiveStringDirectColumnReader::readWithVisitor(
    RowSet rows,
    TVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  int32_t current = visitor.start();
  constexpr bool isExtract =
      std::is_same<typename TVisitor::FilterType, common::AlwaysTrue>::value &&
      std::is_same<
          typename TVisitor::Extract,
          ExtractToReader<SelectiveStringDirectColumnReader>>::value;
  auto nulls = nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;

  if (process::hasAvx2() && isExtract) {
    if (nullsInReadRange_) {
      if (TVisitor::dense) {
        returnReaderNulls_ = true;
        nonNullRowsFromDense(nulls, rows.size(), outerNonNullRows_);
        extractSparse(rows.data(), outerNonNullRows_.size());
      } else {
        int32_t tailSkip = -1;
        anyNulls_ = nonNullRowsFromSparse<false, true>(
            nulls,
            rows,
            innerNonNullRows_,
            outerNonNullRows_,
            rawResultNulls_,
            tailSkip);
        extractSparse(innerNonNullRows_.data(), innerNonNullRows_.size());
        skipInDecode<false>(tailSkip, 0, nullptr);
      }
    } else {
      extractSparse(rows.data(), rows.size());
    }
    numValues_ = rows.size();
    readOffset_ += numRows;
    return;
  }

  if (nulls) {
    skipInDecode<true>(current, 0, nulls);
  } else {
    skipInDecode<false>(current, 0, nulls);
  }
  if (nulls) {
    decode<true, TVisitor>(nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    decode<false, TVisitor>(nullptr, visitor);
  }
  readOffset_ += numRows;
}

template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveStringDirectColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  readWithVisitor(
      rows,
      ColumnVisitor<folly::StringPiece, TFilter, ExtractValues, isDense>(
          *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
}

template <bool isDense, typename ExtractValues>
void SelectiveStringDirectColumnReader::processFilter(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case common::FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kIsNull:
      filterNulls<StringView>(
          rows,
          true,
          !std::is_same<decltype(extractValues), DropValues>::value);
      break;
    case common::FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), DropValues>::value) {
        filterNulls<StringView>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case common::FilterKind::kBytesRange:
      readHelper<common::BytesRange, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kBytesValues:
      readHelper<common::BytesValues, isDense>(filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

void SelectiveStringDirectColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  prepareRead<folly::StringPiece>(offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;

  auto end = rows.back() + 1;
  auto numNulls =
      nullsInReadRange_ ? BaseVector::countNulls(nullsInReadRange_, 0, end) : 0;
  ensureCapacity<int32_t>(lengths_, end - numNulls, &memoryPool_);
  lengthDecoder_->nextLengths(lengths_->asMutable<int32_t>(), end - numNulls);
  rawLengths_ = lengths_->as<uint32_t>();
  lengthIndex_ = 0;
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        readHelper<common::AlwaysTrue, true>(
            &Filters::alwaysTrue,
            rows,
            ExtractToGenericHook(scanSpec_->valueHook()));
      } else {
        readHelper<common::AlwaysTrue, false>(
            &Filters::alwaysTrue,
            rows,
            ExtractToGenericHook(scanSpec_->valueHook()));
      }
      return;
    }
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, ExtractToReader(this));
    } else {
      processFilter<false>(scanSpec_->filter(), rows, ExtractToReader(this));
    }
  } else {
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, DropValues());
    } else {
      processFilter<false>(scanSpec_->filter(), rows, DropValues());
    }
  }
}

class SelectiveStringDictionaryColumnReader : public SelectiveColumnReader {
 public:
  using ValueType = int32_t;

  SelectiveStringDictionaryColumnReader(
      const std::shared_ptr<const TypeWithId>& nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  void resetFilterCaches() override {
    // 'filterCache_' could be empty before first read.
    if (!filterCache_.empty()) {
      simd::memset(
          filterCache_.data(), FilterResult::kUnknown, dictionaryCount_);
    }
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (flatMapContext_.inMapDecoder) {
      flatMapContext_.inMapDecoder->seekToRowGroup(positionsProvider);
    }

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    if (strideDictStream_) {
      strideDictStream_->seekToRowGroup(positionsProvider);
      strideDictLengthDecoder_->seekToRowGroup(positionsProvider);
      // skip row group dictionary size
      positionsProvider.next();
    }

    dictIndex_->seekToRowGroup(positionsProvider);

    if (inDictionaryReader_) {
      inDictionaryReader_->seekToRowGroup(positionsProvider);
    }

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  void loadStrideDictionary();
  void makeDictionaryBaseVector();

  template <typename TVisitor>
  void readWithVisitor(RowSet rows, TVisitor visitor);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void readHelper(common::Filter* filter, RowSet rows, ExtractValues values);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      RowSet rows,
      ExtractValues extractValues);

  BufferPtr loadDictionary(
      uint64_t count,
      SeekableInputStream& data,
      IntDecoder</*isSigned*/ false>& lengthDecoder,
      BufferPtr& offsets);

  void ensureInitialized();

  BufferPtr dictionaryBlob_;
  BufferPtr dictionaryOffset_;
  BufferPtr inDict_;
  BufferPtr strideDict_;
  BufferPtr strideDictOffset_;
  std::unique_ptr<IntDecoder</*isSigned*/ false>> dictIndex_;
  std::unique_ptr<ByteRleDecoder> inDictionaryReader_;
  std::unique_ptr<SeekableInputStream> strideDictStream_;
  std::unique_ptr<IntDecoder</*isSigned*/ false>> strideDictLengthDecoder_;

  FlatVectorPtr<StringView> dictionaryValues_;

  uint64_t dictionaryCount_;
  uint64_t strideDictCount_{0};
  int64_t lastStrideIndex_;
  size_t positionOffset_;
  size_t strideDictSizeOffset_;

  const StrideIndexProvider& provider_;

  // lazy load the dictionary
  std::unique_ptr<IntDecoder</*isSigned*/ false>> lengthDecoder_;
  std::unique_ptr<SeekableInputStream> blobStream_;
  raw_vector<uint8_t> filterCache_;
  bool initialized_{false};
};

SelectiveStringDictionaryColumnReader::SelectiveStringDictionaryColumnReader(
    const std::shared_ptr<const TypeWithId>& nodeType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveColumnReader(
          nodeType,
          stripe,
          scanSpec,
          nodeType->type,
          std::move(flatMapContext)),
      lastStrideIndex_(-1),
      provider_(stripe.getStrideIndexProvider()) {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  RleVersion rleVersion =
      convertRleVersion(stripe.getEncoding(encodingKey).kind());
  dictionaryCount_ = stripe.getEncoding(encodingKey).dictionarysize();

  const auto dataId = encodingKey.forKind(proto::Stream_Kind_DATA);
  bool dictVInts = stripe.getUseVInts(dataId);
  dictIndex_ = IntDecoder</*isSigned*/ false>::createRle(
      stripe.getStream(dataId, true),
      rleVersion,
      memoryPool_,
      dictVInts,
      INT_BYTE_SIZE);

  const auto lenId = encodingKey.forKind(proto::Stream_Kind_LENGTH);
  bool lenVInts = stripe.getUseVInts(lenId);
  lengthDecoder_ = IntDecoder</*isSigned*/ false>::createRle(
      stripe.getStream(lenId, false),
      rleVersion,
      memoryPool_,
      lenVInts,
      INT_BYTE_SIZE);

  blobStream_ = stripe.getStream(
      encodingKey.forKind(proto::Stream_Kind_DICTIONARY_DATA), false);

  // handle in dictionary stream
  std::unique_ptr<SeekableInputStream> inDictStream = stripe.getStream(
      encodingKey.forKind(proto::Stream_Kind_IN_DICTIONARY), false);
  if (inDictStream) {
    DWIO_ENSURE_NOT_NULL(indexStream_, "String index is missing");

    inDictionaryReader_ =
        createBooleanRleDecoder(std::move(inDictStream), encodingKey);

    // stride dictionary only exists if in dictionary exists
    strideDictStream_ = stripe.getStream(
        encodingKey.forKind(proto::Stream_Kind_STRIDE_DICTIONARY), true);
    DWIO_ENSURE_NOT_NULL(strideDictStream_, "Stride dictionary is missing");

    const auto strideDictLenId =
        encodingKey.forKind(proto::Stream_Kind_STRIDE_DICTIONARY_LENGTH);
    bool strideLenVInt = stripe.getUseVInts(strideDictLenId);
    strideDictLengthDecoder_ = IntDecoder</*isSigned*/ false>::createRle(
        stripe.getStream(strideDictLenId, true),
        rleVersion,
        memoryPool_,
        strideLenVInt,
        INT_BYTE_SIZE);
  }
}

uint64_t SelectiveStringDictionaryColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  dictIndex_->skip(numValues);
  if (inDictionaryReader_) {
    inDictionaryReader_->skip(numValues);
  }
  return numValues;
}

BufferPtr SelectiveStringDictionaryColumnReader::loadDictionary(
    uint64_t count,
    SeekableInputStream& data,
    IntDecoder</*isSigned*/ false>& lengthDecoder,
    BufferPtr& offsets) {
  // read lengths from length reader
  auto* offsetsPtr = offsets->asMutable<int64_t>();
  offsetsPtr[0] = 0;
  lengthDecoder.next(offsetsPtr + 1, count, nullptr);

  // set up array that keeps offset of start positions of individual entries
  // in the dictionary
  for (uint64_t i = 1; i < count + 1; ++i) {
    offsetsPtr[i] += offsetsPtr[i - 1];
  }

  // read bytes from underlying string
  int64_t blobSize = offsetsPtr[count];
  BufferPtr dictionary = AlignedBuffer::allocate<char>(blobSize, &memoryPool_);
  data.readFully(dictionary->asMutable<char>(), blobSize);
  return dictionary;
}

void SelectiveStringDictionaryColumnReader::loadStrideDictionary() {
  auto nextStride = provider_.getStrideIndex();
  if (nextStride == lastStrideIndex_) {
    return;
  }

  // get stride dictionary size and load it if needed
  auto& positions = index_->entry(nextStride).positions();
  strideDictCount_ = positions.Get(strideDictSizeOffset_);
  if (strideDictCount_ > 0) {
    // seek stride dictionary related streams
    std::vector<uint64_t> pos(
        positions.begin() + positionOffset_, positions.end());
    PositionProvider pp(pos);
    strideDictStream_->seekToRowGroup(pp);
    strideDictLengthDecoder_->seekToRowGroup(pp);

    ensureCapacity<int64_t>(
        strideDictOffset_, strideDictCount_ + 1, &memoryPool_);
    strideDict_ = loadDictionary(
        strideDictCount_,
        *strideDictStream_,
        *strideDictLengthDecoder_,
        strideDictOffset_);
  } else {
    strideDict_.reset();
  }

  lastStrideIndex_ = nextStride;

  dictionaryValues_.reset();
  filterCache_.resize(dictionaryCount_ + strideDictCount_);
  simd::memset(
      filterCache_.data(),
      FilterResult::kUnknown,
      dictionaryCount_ + strideDictCount_);
}

void SelectiveStringDictionaryColumnReader::makeDictionaryBaseVector() {
  const auto* dictionaryBlob_Ptr = dictionaryBlob_->as<char>();
  const auto* dictionaryOffset_sPtr = dictionaryOffset_->as<int64_t>();
  if (strideDictCount_) {
    // TODO Reuse memory
    BufferPtr values = AlignedBuffer::allocate<StringView>(
        dictionaryCount_ + strideDictCount_, &memoryPool_);
    auto* valuesPtr = values->asMutable<StringView>();
    for (size_t i = 0; i < dictionaryCount_; i++) {
      valuesPtr[i] = StringView(
          dictionaryBlob_Ptr + dictionaryOffset_sPtr[i],
          dictionaryOffset_sPtr[i + 1] - dictionaryOffset_sPtr[i]);
    }

    const auto* strideDictPtr = strideDict_->as<char>();
    const auto* strideDictOffset_Ptr = strideDictOffset_->as<int64_t>();
    for (size_t i = 0; i < strideDictCount_; i++) {
      valuesPtr[dictionaryCount_ + i] = StringView(
          strideDictPtr + strideDictOffset_Ptr[i],
          strideDictOffset_Ptr[i + 1] - strideDictOffset_Ptr[i]);
    }

    dictionaryValues_ = std::make_shared<FlatVector<StringView>>(
        &memoryPool_,
        type_,
        BufferPtr(nullptr), // TODO nulls
        dictionaryCount_ + strideDictCount_ /*length*/,
        values,
        std::vector<BufferPtr>{dictionaryBlob_, strideDict_});
  } else {
    // TODO Reuse memory
    BufferPtr values =
        AlignedBuffer::allocate<StringView>(dictionaryCount_, &memoryPool_);
    auto* valuesPtr = values->asMutable<StringView>();
    for (size_t i = 0; i < dictionaryCount_; i++) {
      valuesPtr[i] = StringView(
          dictionaryBlob_Ptr + dictionaryOffset_sPtr[i],
          dictionaryOffset_sPtr[i + 1] - dictionaryOffset_sPtr[i]);
    }

    dictionaryValues_ = std::make_shared<FlatVector<StringView>>(
        &memoryPool_,
        type_,
        BufferPtr(nullptr), // TODO nulls
        dictionaryCount_ /*length*/,
        values,
        std::vector<BufferPtr>{dictionaryBlob_});
  }
}

template <typename TVisitor>
void SelectiveStringDictionaryColumnReader::readWithVisitor(
    RowSet rows,
    TVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  auto decoder = dynamic_cast<RleDecoderV1<false>*>(dictIndex_.get());
  VELOX_CHECK(decoder, "Only RLEv1 is supported");
  if (nullsInReadRange_) {
    decoder->readWithVisitor<true, TVisitor>(
        nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    decoder->readWithVisitor<false, TVisitor>(nullptr, visitor);
  }
  readOffset_ += numRows;
}

template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveStringDictionaryColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues values) {
  readWithVisitor(
      rows,
      StringDictionaryColumnVisitor<TFilter, ExtractValues, isDense>(
          *reinterpret_cast<TFilter*>(filter),
          this,
          rows,
          values,
          (strideDict_ && inDict_) ? inDict_->as<uint64_t>() : nullptr,
          filterCache_.empty() ? nullptr : filterCache_.data(),
          dictionaryBlob_->as<char>(),
          dictionaryOffset_->as<uint64_t>(),
          dictionaryCount_,
          strideDict_ ? strideDict_->as<char>() : nullptr,
          strideDictOffset_ ? strideDictOffset_->as<uint64_t>() : nullptr));
}

template <bool isDense, typename ExtractValues>
void SelectiveStringDictionaryColumnReader::processFilter(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case common::FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kIsNull:
      filterNulls<int32_t>(
          rows,
          true,
          !std::is_same<decltype(extractValues), DropValues>::value);
      break;
    case common::FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), DropValues>::value) {
        filterNulls<int32_t>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case common::FilterKind::kBytesRange:
      readHelper<common::BytesRange, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kBytesValues:
      readHelper<common::BytesValues, isDense>(filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

void SelectiveStringDictionaryColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  static std::array<char, 1> EMPTY_DICT;

  prepareRead<int32_t>(offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;
  const auto* nullsPtr =
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  // lazy loading dictionary data when first hit
  ensureInitialized();

  if (inDictionaryReader_) {
    auto end = rows.back() + 1;
    bool isBulk = useBulkPath();
    int32_t numFlags = (isBulk && nullsInReadRange_)
        ? bits::countNonNulls(nullsInReadRange_->as<uint64_t>(), 0, end)
        : end;
    ensureCapacity<uint64_t>(inDict_, bits::nwords(numFlags), &memoryPool_);
    inDictionaryReader_->next(
        inDict_->asMutable<char>(), numFlags, isBulk ? nullptr : nullsPtr);
    loadStrideDictionary();
    if (strideDict_) {
      DWIO_ENSURE_NOT_NULL(strideDictOffset_);

      // It's possible strideDictBlob is nullptr when stride dictionary only
      // contains empty string. In that case, we need to make sure
      // strideDictBlob points to some valid address, and the last entry of
      // strideDictOffset_ have value 0.
      auto strideDictBlob = strideDict_->as<char>();
      if (!strideDictBlob) {
        strideDictBlob = EMPTY_DICT.data();
        DWIO_ENSURE_EQ(strideDictOffset_->as<int64_t>()[strideDictCount_], 0);
      }
    }
  }
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        readHelper<common::AlwaysTrue, true>(
            &Filters::alwaysTrue,
            rows,
            ExtractStringDictionaryToGenericHook(
                scanSpec_->valueHook(),
                rows,
                (strideDict_ && inDict_) ? inDict_->as<uint64_t>() : nullptr,
                dictionaryBlob_->as<char>(),
                dictionaryOffset_->as<uint64_t>(),
                dictionaryCount_,
                strideDict_ ? strideDict_->as<char>() : nullptr,
                strideDictOffset_ ? strideDictOffset_->as<uint64_t>()
                                  : nullptr));
      } else {
        readHelper<common::AlwaysTrue, false>(
            &Filters::alwaysTrue,
            rows,
            ExtractStringDictionaryToGenericHook(
                scanSpec_->valueHook(),
                rows,
                (strideDict_ && inDict_) ? inDict_->as<uint64_t>() : nullptr,
                dictionaryBlob_->as<char>(),
                dictionaryOffset_->as<uint64_t>(),
                dictionaryCount_,
                strideDict_ ? strideDict_->as<char>() : nullptr,
                strideDictOffset_ ? strideDictOffset_->as<uint64_t>()
                                  : nullptr));
      }
      return;
    }
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, ExtractToReader(this));
    } else {
      processFilter<false>(scanSpec_->filter(), rows, ExtractToReader(this));
    }
  } else {
    if (isDense) {
      processFilter<true>(scanSpec_->filter(), rows, DropValues());
    } else {
      processFilter<false>(scanSpec_->filter(), rows, DropValues());
    }
  }
}

void SelectiveStringDictionaryColumnReader::getValues(
    RowSet rows,
    VectorPtr* result) {
  if (!dictionaryValues_) {
    makeDictionaryBaseVector();
  }
  compactScalarValues<int32_t, int32_t>(rows, false);

  *result = std::make_shared<DictionaryVector<StringView>>(
      &memoryPool_,
      !anyNulls_               ? nullptr
          : returnReaderNulls_ ? nullsInReadRange_
                               : resultNulls_,
      numValues_,
      dictionaryValues_,
      TypeKind::INTEGER,
      values_);

  if (scanSpec_->makeFlat()) {
    BaseVector::ensureWritable(
        SelectivityVector::empty(), (*result)->type(), &memoryPool_, result);
  }
}

void SelectiveStringDictionaryColumnReader::ensureInitialized() {
  if (LIKELY(initialized_)) {
    return;
  }

  Timer timer;

  ensureCapacity<int64_t>(
      dictionaryOffset_, dictionaryCount_ + 1, &memoryPool_);
  dictionaryBlob_ = loadDictionary(
      dictionaryCount_, *blobStream_, *lengthDecoder_, dictionaryOffset_);
  dictionaryValues_.reset();
  filterCache_.resize(dictionaryCount_);
  simd::memset(filterCache_.data(), FilterResult::kUnknown, dictionaryCount_);

  // handle in dictionary stream
  if (inDictionaryReader_) {
    ensureRowGroupIndex();
    // load stride dictionary offsets
    auto indexStartOffset = flatMapContext_.inMapDecoder
        ? flatMapContext_.inMapDecoder->loadIndices(*index_, 0)
        : 0;
    positionOffset_ = notNullDecoder_
        ? notNullDecoder_->loadIndices(*index_, indexStartOffset)
        : indexStartOffset;
    size_t offset = strideDictStream_->loadIndices(*index_, positionOffset_);
    strideDictSizeOffset_ =
        strideDictLengthDecoder_->loadIndices(*index_, offset);
  }
  initialized_ = true;
  initTimeClocks_ = timer.elapsedClocks();
}

class SelectiveTimestampColumnReader : public SelectiveColumnReader {
 public:
  // The readers produce int64_t, the vector is Timestamps.
  using ValueType = int64_t;

  SelectiveTimestampColumnReader(
      const std::shared_ptr<const TypeWithId>& nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flaatMapContext);

  void seekToRowGroup(uint32_t index) override;
  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  template <bool dense>
  void readHelper(RowSet rows);

  std::unique_ptr<IntDecoder</*isSigned*/ true>> seconds_;
  std::unique_ptr<IntDecoder</*isSigned*/ false>> nano_;

  // Values from copied from 'seconds_'. Nanos are in 'values_'.
  BufferPtr secondsValues_;
};

SelectiveTimestampColumnReader::SelectiveTimestampColumnReader(
    const std::shared_ptr<const TypeWithId>& nodeType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveColumnReader(
          nodeType,
          stripe,
          scanSpec,
          nodeType->type,
          std::move(flatMapContext)) {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  RleVersion vers = convertRleVersion(stripe.getEncoding(encodingKey).kind());
  auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
  bool vints = stripe.getUseVInts(data);
  seconds_ = IntDecoder</*isSigned*/ true>::createRle(
      stripe.getStream(data, true), vers, memoryPool_, vints, LONG_BYTE_SIZE);
  auto nanoData = encodingKey.forKind(proto::Stream_Kind_NANO_DATA);
  bool nanoVInts = stripe.getUseVInts(nanoData);
  nano_ = IntDecoder</*isSigned*/ false>::createRle(
      stripe.getStream(nanoData, true),
      vers,
      memoryPool_,
      nanoVInts,
      LONG_BYTE_SIZE);
}

uint64_t SelectiveTimestampColumnReader::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);
  seconds_->skip(numValues);
  nano_->skip(numValues);
  return numValues;
}

void SelectiveTimestampColumnReader::seekToRowGroup(uint32_t index) {
  ensureRowGroupIndex();

  auto positions = toPositions(index_->entry(index));
  PositionProvider positionsProvider(positions);
  if (notNullDecoder_) {
    notNullDecoder_->seekToRowGroup(positionsProvider);
  }

  seconds_->seekToRowGroup(positionsProvider);
  nano_->seekToRowGroup(positionsProvider);
  // Check that all the provided positions have been consumed.
  VELOX_CHECK(!positionsProvider.hasNext());
}

template <bool dense>
void SelectiveTimestampColumnReader::readHelper(RowSet rows) {
  vector_size_t numRows = rows.back() + 1;
  ExtractToReader extractValues(this);
  common::AlwaysTrue filter;
  auto secondsV1 = dynamic_cast<RleDecoderV1<true>*>(seconds_.get());
  VELOX_CHECK(secondsV1, "Only RLEv1 is supported");
  if (nullsInReadRange_) {
    secondsV1->readWithVisitor<true>(
        nullsInReadRange_->as<uint64_t>(),
        DirectRleColumnVisitor<
            int64_t,
            common::AlwaysTrue,
            decltype(extractValues),
            dense>(filter, this, rows, extractValues));
  } else {
    secondsV1->readWithVisitor<false>(
        nullptr,
        DirectRleColumnVisitor<
            int64_t,
            common::AlwaysTrue,
            decltype(extractValues),
            dense>(filter, this, rows, extractValues));
  }

  // Save the seconds into their own buffer before reading nanos into
  // 'values_'
  ensureCapacity<uint64_t>(secondsValues_, numValues_, &memoryPool_);
  secondsValues_->setSize(numValues_ * sizeof(int64_t));
  memcpy(
      secondsValues_->asMutable<char>(),
      rawValues_,
      numValues_ * sizeof(int64_t));

  // We read the nanos into 'values_' starting at index 0.
  numValues_ = 0;
  auto nanosV1 = dynamic_cast<RleDecoderV1<false>*>(nano_.get());
  VELOX_CHECK(nanosV1, "Only RLEv1 is supported");
  if (nullsInReadRange_) {
    nanosV1->readWithVisitor<true>(
        nullsInReadRange_->as<uint64_t>(),
        DirectRleColumnVisitor<
            int64_t,
            common::AlwaysTrue,
            decltype(extractValues),
            dense>(filter, this, rows, extractValues));
  } else {
    nanosV1->readWithVisitor<false>(
        nullptr,
        DirectRleColumnVisitor<
            int64_t,
            common::AlwaysTrue,
            decltype(extractValues),
            dense>(filter, this, rows, extractValues));
  }
  readOffset_ += numRows;
}

void SelectiveTimestampColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  prepareRead<int64_t>(offset, rows, incomingNulls);
  VELOX_CHECK(!scanSpec_->filter());
  bool isDense = rows.back() == rows.size() - 1;
  if (isDense) {
    readHelper<true>(rows);
  } else {
    readHelper<false>(rows);
  }
}

void SelectiveTimestampColumnReader::getValues(RowSet rows, VectorPtr* result) {
  // We merge the seconds and nanos into 'values_'
  auto tsValues = AlignedBuffer::allocate<Timestamp>(numValues_, &memoryPool_);
  auto rawTs = tsValues->asMutable<Timestamp>();
  auto secondsData = secondsValues_->as<int64_t>();
  auto nanosData = values_->as<uint64_t>();
  auto rawNulls = nullsInReadRange_
      ? (returnReaderNulls_ ? nullsInReadRange_->as<uint64_t>()
                            : rawResultNulls_)
      : nullptr;
  for (auto i = 0; i < numValues_; i++) {
    if (!rawNulls || !bits::isBitNull(rawNulls, i)) {
      auto nanos = nanosData[i];
      uint64_t zeros = nanos & 0x7;
      nanos >>= 3;
      if (zeros != 0) {
        for (uint64_t j = 0; j <= zeros; ++j) {
          nanos *= 10;
        }
      }
      auto seconds = secondsData[i] + EPOCH_OFFSET;
      if (seconds < 0 && nanos != 0) {
        seconds -= 1;
      }
      rawTs[i] = Timestamp(seconds, nanos);
    }
  }
  values_ = tsValues;
  rawValues_ = values_->asMutable<char>();
  getFlatValues<Timestamp, Timestamp>(rows, result, type_, true);
}

class SelectiveStructColumnReader : public SelectiveColumnReader {
 public:
  SelectiveStructColumnReader(
      const std::shared_ptr<const TypeWithId>& requestedType,
      const std::shared_ptr<const TypeWithId>& dataType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  void resetFilterCaches() override {
    for (auto& child : children_) {
      child->resetFilterCaches();
    }
  }

  void seekToRowGroup(uint32_t index) override {
    if (isTopLevel_ && !notNullDecoder_) {
      readOffset_ = index * rowsPerRowGroup_;
      return;
    }
    if (notNullDecoder_) {
      ensureRowGroupIndex();
      auto positions = toPositions(index_->entry(index));
      PositionProvider positionsProvider(positions);
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }
    // Set the read offset recursively. Do this before seeking the
    // children because list/map children will reset the offsets for
    // their children.
    setReadOffsetRecursive(index * rowsPerRowGroup_);
    for (auto& child : children_) {
      child->seekToRowGroup(index);
    }
  }

  uint64_t skip(uint64_t numValues) override;

  void next(
      uint64_t numValues,
      VectorPtr& result,
      const uint64_t* incomingNulls) override;

  std::vector<uint32_t> filterRowGroups(
      uint64_t rowGroupSize,
      const StatsContext& context) const override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

  uint64_t numReads() const {
    return numReads_;
  }

  vector_size_t lazyVectorReadOffset() const {
    return lazyVectorReadOffset_;
  }

  /// Advance field reader to the row group closest to specified offset by
  /// calling seekToRowGroup.
  void advanceFieldReader(SelectiveColumnReader* reader, vector_size_t offset) {
    if (!reader->isTopLevel()) {
      return;
    }
    auto rowGroup = reader->readOffset() / rowsPerRowGroup_;
    auto nextRowGroup = offset / rowsPerRowGroup_;
    if (nextRowGroup > rowGroup) {
      reader->seekToRowGroup(nextRowGroup);
      reader->setReadOffset(nextRowGroup * rowsPerRowGroup_);
    }
  }

  // Returns the nulls bitmap from reading this. Used in LazyVector loaders.
  const uint64_t* nulls() const {
    return nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  }

  void setReadOffsetRecursive(vector_size_t readOffset) override {
    readOffset_ = readOffset;
    for (auto& child : children_) {
      child->setReadOffsetRecursive(readOffset);
    }
  }

  void setIsTopLevel() override {
    isTopLevel_ = true;
    if (!notNullDecoder_) {
      for (auto& child : children_) {
        child->setIsTopLevel();
      }
    }
  }

 private:
  const std::shared_ptr<const dwio::common::TypeWithId> requestedType_;
  std::vector<std::unique_ptr<SelectiveColumnReader>> children_;
  // Sequence number of output batch. Checked against ColumnLoaders
  // created by 'this' to verify they are still valid at load.
  uint64_t numReads_ = 0;
  vector_size_t lazyVectorReadOffset_;

  // Dense set of rows to read in next().
  raw_vector<vector_size_t> rows_;
};

SelectiveStructColumnReader::SelectiveStructColumnReader(
    const std::shared_ptr<const TypeWithId>& requestedType,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveColumnReader(
          dataType,
          stripe,
          scanSpec,
          dataType->type,
          std::move(flatMapContext)),
      requestedType_{requestedType} {
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  DWIO_ENSURE_EQ(encodingKey.node, dataType->id, "working on the same node");
  auto encoding = static_cast<int64_t>(stripe.getEncoding(encodingKey).kind());
  DWIO_ENSURE_EQ(
      encoding,
      proto::ColumnEncoding_Kind_DIRECT,
      "Unknown encoding for StructColumnReader");

  const auto& cs = stripe.getColumnSelector();
  auto& childSpecs = scanSpec->children();
  for (auto i = 0; i < childSpecs.size(); ++i) {
    auto childSpec = childSpecs[i].get();
    if (childSpec->isConstant()) {
      continue;
    }
    auto childDataType = nodeType_->childByName(childSpec->fieldName());
    auto childRequestedType =
        requestedType_->childByName(childSpec->fieldName());
    VELOX_CHECK(cs.shouldReadNode(childDataType->id));
    children_.push_back(SelectiveColumnReader::build(
        childRequestedType,
        childDataType,
        stripe,
        childSpec,
        FlatMapContext{encodingKey.sequence, nullptr}));
    childSpec->setSubscript(children_.size() - 1);
  }
}

std::vector<uint32_t> SelectiveStructColumnReader::filterRowGroups(
    uint64_t rowGroupSize,
    const StatsContext& context) const {
  auto stridesToSkip =
      SelectiveColumnReader::filterRowGroups(rowGroupSize, context);
  for (const auto& child : children_) {
    auto childStridesToSkip = child->filterRowGroups(rowGroupSize, context);
    if (stridesToSkip.empty()) {
      stridesToSkip = std::move(childStridesToSkip);
    } else {
      std::vector<uint32_t> merged;
      merged.reserve(childStridesToSkip.size() + stridesToSkip.size());
      std::merge(
          childStridesToSkip.begin(),
          childStridesToSkip.end(),
          stridesToSkip.begin(),
          stridesToSkip.end(),
          std::back_inserter(merged));
      stridesToSkip = std::move(merged);
    }
  }
  return stridesToSkip;
}

uint64_t SelectiveStructColumnReader::skip(uint64_t numValues) {
  auto numNonNulls = ColumnReader::skip(numValues);
  // 'readOffset_' of struct child readers is aligned with
  // 'readOffset_' of the struct. The child readers may have fewer
  // values since there is no value in children where the struct is
  // null. But because struct nulls are injected as nulls in child
  // readers, it is practical to keep the row numbers in terms of the
  // enclosing struct.
  for (auto& child : children_) {
    if (child) {
      child->skip(numNonNulls);
      child->setReadOffset(child->readOffset() + numValues);
    }
  }
  return numValues;
}

void SelectiveStructColumnReader::next(
    uint64_t numValues,
    VectorPtr& result,
    const uint64_t* incomingNulls) {
  VELOX_CHECK(!incomingNulls, "next may only be called for the root reader.");
  if (children_.empty()) {
    // no readers
    // This can be either count(*) query or a query that select only
    // constant columns (partition keys or columns missing from an old file
    // due to schema evolution)
    result->resize(numValues);

    auto resultRowVector = std::dynamic_pointer_cast<RowVector>(result);
    auto& childSpecs = scanSpec_->children();
    for (auto& childSpec : childSpecs) {
      VELOX_CHECK(childSpec->isConstant());
      auto channel = childSpec->channel();
      resultRowVector->childAt(channel) =
          BaseVector::wrapInConstant(numValues, 0, childSpec->constantValue());
    }
    return;
  }
  auto oldSize = rows_.size();
  rows_.resize(numValues);
  if (numValues > oldSize) {
    std::iota(&rows_[oldSize], &rows_[rows_.size()], oldSize);
  }
  read(readOffset_, rows_, nullptr);
  getValues(outputRows(), &result);
}

class ColumnLoader : public velox::VectorLoader {
 public:
  ColumnLoader(
      SelectiveStructColumnReader* structReader,
      SelectiveColumnReader* fieldReader,
      uint64_t version)
      : structReader_(structReader),
        fieldReader_(fieldReader),
        version_(version) {}

 protected:
  void loadInternal(RowSet rows, ValueHook* hook, VectorPtr* result) override;

 private:
  SelectiveStructColumnReader* structReader_;
  SelectiveColumnReader* fieldReader_;
  // This is checked against the version of 'structReader' on load. If
  // these differ, 'structReader' has been advanced since the creation
  // of 'this' and 'this' is no longer loadable.
  const uint64_t version_;
};

// Wraps '*result' in a dictionary to make the contiguous values
// appear at the indices i 'rows'. Used when loading a LazyVector for
// a sparse set of rows in conditional exprs.
static void scatter(RowSet rows, VectorPtr* result) {
  auto end = rows.back() + 1;
  // Initialize the indices to 0 to make the dictionary safely
  // readable also for uninitialized positions.
  auto indices =
      AlignedBuffer::allocate<vector_size_t>(end, (*result)->pool(), 0);
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (int32_t i = 0; i < rows.size(); ++i) {
    rawIndices[rows[i]] = i;
  }
  *result =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), indices, end, *result);
}

void ColumnLoader::loadInternal(
    RowSet rows,
    ValueHook* hook,
    VectorPtr* result) {
  VELOX_CHECK_EQ(
      version_,
      structReader_->numReads(),
      "Loading LazyVector after the enclosing reader has moved");
  auto offset = structReader_->lazyVectorReadOffset();
  auto incomingNulls = structReader_->nulls();
  auto outputRows = structReader_->outputRows();
  raw_vector<vector_size_t> selectedRows;
  RowSet effectiveRows;
  if (rows.size() == outputRows.size()) {
    // All the rows planned at creation are accessed.
    effectiveRows = outputRows;
  } else {
    // rows is a set of indices into outputRows. There has been a
    // selection between creation and loading.
    selectedRows.resize(rows.size());
    assert(!selectedRows.empty());
    for (auto i = 0; i < rows.size(); ++i) {
      selectedRows[i] = outputRows[rows[i]];
    }
    effectiveRows = RowSet(selectedRows);
  }

  structReader_->advanceFieldReader(fieldReader_, offset);
  fieldReader_->scanSpec()->setValueHook(hook);
  fieldReader_->read(offset, effectiveRows, incomingNulls);
  if (!hook) {
    fieldReader_->getValues(effectiveRows, result);
    if (rows.size() != outputRows.size()) {
      // We read sparsely. The values that were read should appear
      // at the indices in the result vector that were given by
      // 'rows'.
      scatter(rows, result);
    }
  }
}

void SelectiveStructColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  numReads_ = scanSpec_->newRead();
  prepareRead<char>(offset, rows, incomingNulls);
  RowSet activeRows = rows;
  auto& childSpecs = scanSpec_->children();
  const uint64_t* structNulls =
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  bool hasFilter = false;
  assert(!children_.empty());
  for (size_t i = 0; i < childSpecs.size(); ++i) {
    auto& childSpec = childSpecs[i];
    if (childSpec->isConstant()) {
      continue;
    }
    auto fieldIndex = childSpec->subscript();
    auto reader = children_.at(fieldIndex).get();
    if (reader->isTopLevel() && childSpec->projectOut() &&
        !childSpec->filter() && !childSpec->extractValues()) {
      // Will make a LazyVector.
      continue;
    }
    advanceFieldReader(reader, offset);
    if (childSpec->filter()) {
      hasFilter = true;
      {
        SelectivityTimer timer(childSpec->selectivity(), activeRows.size());

        reader->resetInitTimeClocks();
        reader->read(offset, activeRows, structNulls);

        // Exclude initialization time.
        timer.subtract(reader->initTimeClocks());

        activeRows = reader->outputRows();
        childSpec->selectivity().addOutput(activeRows.size());
      }
      if (activeRows.empty()) {
        break;
      }
    } else {
      reader->read(offset, activeRows, structNulls);
    }
  }
  if (hasFilter) {
    setOutputRows(activeRows);
  }
  lazyVectorReadOffset_ = offset;
  readOffset_ = offset + rows.back() + 1;
}

void SelectiveStructColumnReader::getValues(RowSet rows, VectorPtr* result) {
  assert(!children_.empty());
  VELOX_CHECK(
      *result != nullptr,
      "SelectiveStructColumnReader expects a non-null result");
  RowVector* resultRow = dynamic_cast<RowVector*>(result->get());
  VELOX_CHECK(resultRow, "Struct reader expects a result of type ROW.");
  resultRow->resize(rows.size());
  if (!rows.size()) {
    return;
  }
  if (nullsInReadRange_) {
    auto readerNulls = nullsInReadRange_->as<uint64_t>();
    auto nulls = resultRow->mutableNulls(rows.size())->asMutable<uint64_t>();
    for (size_t i = 0; i < rows.size(); ++i) {
      bits::setBit(nulls, i, bits::isBitSet(readerNulls, rows[i]));
    }
  } else {
    resultRow->clearNulls(0, rows.size());
  }
  bool lazyPrepared = false;
  auto& childSpecs = scanSpec_->children();
  for (auto i = 0; i < childSpecs.size(); ++i) {
    auto& childSpec = childSpecs[i];
    if (!childSpec->projectOut()) {
      continue;
    }
    auto index = childSpec->subscript();
    auto channel = childSpec->channel();
    if (childSpec->isConstant()) {
      resultRow->childAt(channel) = BaseVector::wrapInConstant(
          rows.size(), 0, childSpec->constantValue());
    } else {
      if (!childSpec->extractValues() && !childSpec->filter() &&
          children_[index]->isTopLevel()) {
        // LazyVector result.
        if (!lazyPrepared) {
          if (rows.size() != outputRows_.size()) {
            setOutputRows(rows);
          }
          lazyPrepared = true;
        }
        resultRow->childAt(channel) = std::make_shared<LazyVector>(
            &memoryPool_,
            resultRow->type()->childAt(channel),
            rows.size(),
            std::make_unique<ColumnLoader>(
                this, children_[index].get(), numReads_));
      } else {
        children_[index]->getValues(rows, &resultRow->childAt(channel));
      }
    }
  }
}

// Abstract superclass for list and map readers. Encapsulates common
// logic for dealing with mapping between enclosing and nested rows.
class SelectiveRepeatedColumnReader : public SelectiveColumnReader {
 public:
  bool useBulkPath() const override {
    return false;
  }

 protected:
  // Buffer size for reading length stream
  static constexpr size_t kBufferSize = 1024;

  SelectiveRepeatedColumnReader(
      std::shared_ptr<const TypeWithId> nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      const TypePtr& type,
      FlatMapContext flatMapContext = FlatMapContext::nonFlatMapContext())
      : SelectiveColumnReader(
            std::move(nodeType),
            stripe,
            scanSpec,
            type,
            flatMapContext) {
    EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
    auto rleVersion = convertRleVersion(stripe.getEncoding(encodingKey).kind());
    auto lenId = encodingKey.forKind(proto::Stream_Kind_LENGTH);
    bool lenVints = stripe.getUseVInts(lenId);
    length_ = IntDecoder</*isSigned*/ false>::createRle(
        stripe.getStream(lenId, true),
        rleVersion,
        memoryPool_,
        lenVints,
        INT_BYTE_SIZE);
  }

  void makeNestedRowSet(RowSet rows) {
    allLengths_.resize(rows.back() + 1);
    assert(!allLengths_.empty()); // for lint only.
    auto nulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    // Reads the lengths, leaves an uninitialized gap for a null
    // map/list. Reading these checks the null nask.
    length_->next(allLengths_.data(), rows.back() + 1, nulls);
    ensureCapacity<vector_size_t>(offsets_, rows.size(), &memoryPool_);
    ensureCapacity<vector_size_t>(sizes_, rows.size(), &memoryPool_);
    auto rawOffsets = offsets_->asMutable<vector_size_t>();
    auto rawSizes = sizes_->asMutable<vector_size_t>();
    vector_size_t nestedLength = 0;
    for (auto row : rows) {
      if (!nulls || !bits::isBitNull(nulls, row)) {
        nestedLength += allLengths_[row];
      }
    }
    nestedRows_.resize(nestedLength);
    vector_size_t currentRow = 0;
    vector_size_t nestedRow = 0;
    vector_size_t nestedOffset = 0;
    for (auto rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
      auto row = rows[rowIndex];
      // Add up the lengths of non-null rows skipped since the last
      // non-null.
      for (auto i = currentRow; i < row; ++i) {
        if (!nulls || !bits::isBitNull(nulls, i)) {
          nestedOffset += allLengths_[i];
        }
      }
      currentRow = row + 1;
      // Check if parent is null after adding up the lengths leading
      // up to this. If null, add a null to the result and keep
      // looping. If the null is last, the lengths will all have been
      // added up.
      if (nulls && bits::isBitNull(nulls, row)) {
        rawOffsets[rowIndex] = 0;
        rawSizes[rowIndex] = 0;
        bits::setNull(rawResultNulls_, rowIndex);
        anyNulls_ = true;
        continue;
      }

      auto lengthAtRow = allLengths_[row];
      std::iota(
          &nestedRows_[nestedRow],
          &nestedRows_[nestedRow + lengthAtRow],
          nestedOffset);
      rawOffsets[rowIndex] = nestedRow;
      rawSizes[rowIndex] = lengthAtRow;
      nestedRow += lengthAtRow;
      nestedOffset += lengthAtRow;
    }
    childTargetReadOffset_ += nestedOffset;
  }

  void compactOffsets(RowSet rows) {
    auto rawOffsets = offsets_->asMutable<vector_size_t>();
    auto rawSizes = sizes_->asMutable<vector_size_t>();
    VELOX_CHECK(
        outputRows_.empty(), "Repeated reader does not support filters");
    RowSet rowsToCompact;
    if (valueRows_.empty()) {
      valueRows_.resize(rows.size());
      rowsToCompact = inputRows_;
    } else {
      rowsToCompact = valueRows_;
    }
    if (rows.size() == rowsToCompact.size()) {
      return;
    }

    int32_t current = 0;
    bool moveNulls = shouldMoveNulls(rows);
    for (int i = 0; i < rows.size(); ++i) {
      auto row = rows[i];
      while (rowsToCompact[current] < row) {
        ++current;
      }
      VELOX_CHECK(rowsToCompact[current] == row);
      valueRows_[i] = row;
      rawOffsets[i] = rawOffsets[current];
      rawSizes[i] = rawSizes[current];
      if (moveNulls && i != current) {
        bits::setBit(
            rawResultNulls_, i, bits::isBitSet(rawResultNulls_, current));
      }
    }
    numValues_ = rows.size();
    valueRows_.resize(numValues_);
    offsets_->setSize(numValues_ * sizeof(vector_size_t));
    sizes_->setSize(numValues_ * sizeof(vector_size_t));
  }

  // Creates a struct if '*result' is empty and 'type' is a row.
  void prepareStructResult(const TypePtr& type, VectorPtr* result) {
    if (!*result && type->kind() == TypeKind::ROW) {
      *result = BaseVector::create(type, 0, &memoryPool_);
    }
  }

  std::vector<int64_t> allLengths_;
  raw_vector<vector_size_t> nestedRows_;
  BufferPtr offsets_;
  BufferPtr sizes_;
  // The position in the child readers that corresponds to the
  // position in the length stream. The child readers can be behind if
  // the last parents were null, so that the child stream was only
  // read up to the last position corresponding to
  // the last non-null parent.
  vector_size_t childTargetReadOffset_ = 0;
  std::unique_ptr<IntDecoder</*isSigned*/ false>> length_;
};

class SelectiveListColumnReader : public SelectiveRepeatedColumnReader {
 public:
  SelectiveListColumnReader(
      const std::shared_ptr<const TypeWithId>& requestedType,
      const std::shared_ptr<const TypeWithId>& dataType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  void resetFilterCaches() override {
    child_->resetFilterCaches();
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    length_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());

    child_->seekToRowGroup(index);
    child_->setReadOffsetRecursive(0);
    childTargetReadOffset_ = 0;
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  std::unique_ptr<SelectiveColumnReader> child_;
  const std::shared_ptr<const dwio::common::TypeWithId> requestedType_;
};

SelectiveListColumnReader::SelectiveListColumnReader(
    const std::shared_ptr<const TypeWithId>& requestedType,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveRepeatedColumnReader(
          dataType,
          stripe,
          scanSpec,
          dataType->type,
          std::move(flatMapContext)),
      requestedType_{requestedType} {
  DWIO_ENSURE_EQ(nodeType_->id, dataType->id, "working on the same node");
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  // count the number of selected sub-columns
  const auto& cs = stripe.getColumnSelector();
  auto& childType = requestedType_->childAt(0);
  VELOX_CHECK(
      cs.shouldReadNode(childType->id),
      "SelectiveListColumnReader must select the values stream");
  if (scanSpec_->children().empty()) {
    scanSpec->getOrCreateChild(common::Subfield("elements"));
  }
  scanSpec_->children()[0]->setProjectOut(true);
  scanSpec_->children()[0]->setExtractValues(true);

  child_ = SelectiveColumnReader::build(
      childType,
      nodeType_->childAt(0),
      stripe,
      scanSpec_->children()[0].get(),
      FlatMapContext{encodingKey.sequence, nullptr});
}

uint64_t SelectiveListColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  if (child_) {
    std::array<int64_t, kBufferSize> buffer;
    uint64_t childElements = 0;
    uint64_t lengthsRead = 0;
    while (lengthsRead < numValues) {
      uint64_t chunk =
          std::min(numValues - lengthsRead, static_cast<uint64_t>(kBufferSize));
      length_->next(buffer.data(), chunk, nullptr);
      for (size_t i = 0; i < chunk; ++i) {
        childElements += static_cast<size_t>(buffer[i]);
      }
      lengthsRead += chunk;
    }
    child_->skip(childElements);
    childTargetReadOffset_ += childElements;
    child_->setReadOffset(child_->readOffset() + childElements);
  } else {
    length_->skip(numValues);
  }
  return numValues;
}

void SelectiveListColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  // Catch up if the child is behind the length stream.
  child_->seekTo(childTargetReadOffset_, false);
  prepareRead<char>(offset, rows, incomingNulls);
  makeNestedRowSet(rows);
  if (child_ && !nestedRows_.empty()) {
    child_->read(child_->readOffset(), nestedRows_, nullptr);
  }
  numValues_ = rows.size();
  readOffset_ = offset + rows.back() + 1;
}

void SelectiveListColumnReader::getValues(RowSet rows, VectorPtr* result) {
  compactOffsets(rows);
  VectorPtr elements;
  if (child_ && !nestedRows_.empty()) {
    prepareStructResult(type_->childAt(0), &elements);
    child_->getValues(nestedRows_, &elements);
  }
  *result = std::make_shared<ArrayVector>(
      &memoryPool_,
      requestedType_->type,
      anyNulls_ ? resultNulls_ : nullptr,
      rows.size(),
      offsets_,
      sizes_,
      elements);
}

class SelectiveMapColumnReader : public SelectiveRepeatedColumnReader {
 public:
  SelectiveMapColumnReader(
      const std::shared_ptr<const TypeWithId>& requestedType,
      const std::shared_ptr<const TypeWithId>& dataType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flatMapContext);

  void resetFilterCaches() override {
    keyReader_->resetFilterCaches();
    elementReader_->resetFilterCaches();
  }

  void seekToRowGroup(uint32_t index) override {
    ensureRowGroupIndex();

    auto positions = toPositions(index_->entry(index));
    PositionProvider positionsProvider(positions);

    if (notNullDecoder_) {
      notNullDecoder_->seekToRowGroup(positionsProvider);
    }

    length_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());

    keyReader_->seekToRowGroup(index);
    keyReader_->setReadOffsetRecursive(0);
    elementReader_->seekToRowGroup(index);
    elementReader_->setReadOffsetRecursive(0);
    childTargetReadOffset_ = 0;
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  std::unique_ptr<SelectiveColumnReader> keyReader_;
  std::unique_ptr<SelectiveColumnReader> elementReader_;
  const std::shared_ptr<const dwio::common::TypeWithId> requestedType_;
};

SelectiveMapColumnReader::SelectiveMapColumnReader(
    const std::shared_ptr<const TypeWithId>& requestedType,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext)
    : SelectiveRepeatedColumnReader(
          dataType,
          stripe,
          scanSpec,
          dataType->type,
          std::move(flatMapContext)),
      requestedType_{requestedType} {
  DWIO_ENSURE_EQ(nodeType_->id, dataType->id, "working on the same node");
  EncodingKey encodingKey{nodeType_->id, flatMapContext_.sequence};
  if (scanSpec_->children().empty()) {
    scanSpec->getOrCreateChild(common::Subfield("keys"));
    scanSpec->getOrCreateChild(common::Subfield("elements"));
  }
  scanSpec->children()[0]->setProjectOut(true);
  scanSpec->children()[0]->setExtractValues(true);
  scanSpec->children()[1]->setProjectOut(true);
  scanSpec_->children()[1]->setExtractValues(true);

  const auto& cs = stripe.getColumnSelector();
  auto& keyType = requestedType_->childAt(0);
  VELOX_CHECK(
      cs.shouldReadNode(keyType->id),
      "Map key must be selected in SelectiveMapColumnReader");
  keyReader_ = SelectiveColumnReader::build(
      keyType,
      nodeType_->childAt(0),
      stripe,
      scanSpec_->children()[0].get(),
      FlatMapContext{encodingKey.sequence, nullptr});

  auto& valueType = requestedType_->childAt(1);
  VELOX_CHECK(
      cs.shouldReadNode(valueType->id),
      "Map Values must be selected in SelectiveMapColumnReader");
  elementReader_ = SelectiveColumnReader::build(
      valueType,
      nodeType_->childAt(1),
      stripe,
      scanSpec_->children()[1].get(),
      FlatMapContext{encodingKey.sequence, nullptr});

  VLOG(1) << "[Map] Initialized map column reader for node " << nodeType_->id;
}

uint64_t SelectiveMapColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  if (keyReader_ || elementReader_) {
    std::array<int64_t, kBufferSize> buffer;
    uint64_t childElements = 0;
    uint64_t lengthsRead = 0;
    while (lengthsRead < numValues) {
      uint64_t chunk =
          std::min(numValues - lengthsRead, static_cast<uint64_t>(kBufferSize));
      length_->next(buffer.data(), chunk, nullptr);
      for (size_t i = 0; i < chunk; ++i) {
        childElements += buffer[i];
      }
      lengthsRead += chunk;
    }
    if (keyReader_) {
      keyReader_->skip(childElements);
      keyReader_->setReadOffset(keyReader_->readOffset() + childElements);
    }
    if (elementReader_) {
      elementReader_->skip(childElements);
      elementReader_->setReadOffset(
          elementReader_->readOffset() + childElements);
    }
    childTargetReadOffset_ += childElements;

  } else {
    length_->skip(numValues);
  }
  return numValues;
}

void SelectiveMapColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  // Catch up if child readers are behind the length stream.
  if (keyReader_) {
    keyReader_->seekTo(childTargetReadOffset_, false);
  }
  if (elementReader_) {
    elementReader_->seekTo(childTargetReadOffset_, false);
  }

  prepareRead<char>(offset, rows, incomingNulls);
  makeNestedRowSet(rows);
  if (keyReader_ && elementReader_ && !nestedRows_.empty()) {
    keyReader_->read(keyReader_->readOffset(), nestedRows_, nullptr);
    elementReader_->read(elementReader_->readOffset(), nestedRows_, nullptr);
  }
  numValues_ = rows.size();
  readOffset_ = offset + rows.back() + 1;
}

void SelectiveMapColumnReader::getValues(RowSet rows, VectorPtr* result) {
  compactOffsets(rows);
  VectorPtr keys;
  VectorPtr values;
  VELOX_CHECK(
      keyReader_ && elementReader_,
      "keyReader_ and elementReaer_ must exist in "
      "SelectiveMapColumnReader::getValues");
  if (!nestedRows_.empty()) {
    keyReader_->getValues(nestedRows_, &keys);
    prepareStructResult(type_->childAt(1), &values);
    elementReader_->getValues(nestedRows_, &values);
  }
  *result = std::make_shared<MapVector>(
      &memoryPool_,
      requestedType_->type,
      anyNulls_ ? resultNulls_ : nullptr,
      rows.size(),
      offsets_,
      sizes_,
      keys,
      values);
}



std::unique_ptr<SelectiveColumnReader> buildIntegerReader(
    const std::shared_ptr<const TypeWithId>& requestedType,
    FlatMapContext flatMapContext,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    uint32_t numBytes,
    common::ScanSpec* scanSpec) {
  EncodingKey ek{requestedType->id, flatMapContext.sequence};
  switch (static_cast<int64_t>(stripe.getEncoding(ek).kind())) {
    case proto::ColumnEncoding_Kind_DICTIONARY:
      return std::make_unique<SelectiveIntegerDictionaryColumnReader>(
          requestedType, dataType, stripe, scanSpec, numBytes);
    case proto::ColumnEncoding_Kind_DIRECT:
      return std::make_unique<SelectiveIntegerDirectColumnReader>(
          requestedType, dataType, stripe, numBytes, scanSpec);
    default:
      DWIO_RAISE("buildReader unhandled integer encoding");
  }
}

std::unique_ptr<SelectiveColumnReader> SelectiveColumnReader::build(
    const std::shared_ptr<const TypeWithId>& requestedType,
    const std::shared_ptr<const TypeWithId>& dataType,
    StripeStreams& stripe,
    common::ScanSpec* scanSpec,
    FlatMapContext flatMapContext) {
  CompatChecker::check(*dataType->type, *requestedType->type);
  EncodingKey ek{dataType->id, flatMapContext.sequence};

  switch (dataType->type->kind()) {
    case TypeKind::INTEGER:
      return buildIntegerReader(
          requestedType,
          std::move(flatMapContext),
          dataType,
          stripe,
          INT_BYTE_SIZE,
          scanSpec);
    case TypeKind::BIGINT:
      return buildIntegerReader(
          requestedType,
          std::move(flatMapContext),
          dataType,
          stripe,
          LONG_BYTE_SIZE,
          scanSpec);
    case TypeKind::SMALLINT:
      return buildIntegerReader(
          requestedType,
          std::move(flatMapContext),
          dataType,
          stripe,
          SHORT_BYTE_SIZE,
          scanSpec);
    case TypeKind::ARRAY:
      return std::make_unique<SelectiveListColumnReader>(
          requestedType, dataType, stripe, scanSpec, flatMapContext);
    case TypeKind::MAP:
      if (stripe.getEncoding(ek).kind() ==
          proto::ColumnEncoding_Kind_MAP_FLAT) {
        VELOX_UNSUPPORTED("SelectiveColumnReader does not support flat maps");
      }
      return std::make_unique<SelectiveMapColumnReader>(
          requestedType, dataType, stripe, scanSpec, std::move(flatMapContext));
    case TypeKind::REAL:
      if (requestedType->type->kind() == TypeKind::REAL) {
        return std::make_unique<
            SelectiveFloatingPointColumnReader<float, float>>(
            requestedType, stripe, scanSpec, std::move(flatMapContext));
      } else {
        return std::make_unique<
            SelectiveFloatingPointColumnReader<float, double>>(
            requestedType, stripe, scanSpec, std::move(flatMapContext));
      }
    case TypeKind::DOUBLE:
      return std::make_unique<
          SelectiveFloatingPointColumnReader<double, double>>(
          requestedType, stripe, scanSpec, std::move(flatMapContext));
    case TypeKind::ROW:
      return std::make_unique<SelectiveStructColumnReader>(
          requestedType, dataType, stripe, scanSpec, std::move(flatMapContext));
    case TypeKind::BOOLEAN:
      return std::make_unique<SelectiveByteRleColumnReader>(
          requestedType,
          dataType,
          stripe,
          scanSpec,
          true,
          std::move(flatMapContext));
    case TypeKind::TINYINT:
      return std::make_unique<SelectiveByteRleColumnReader>(
          requestedType,
          dataType,
          stripe,
          scanSpec,
          false,
          std::move(flatMapContext));
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      switch (static_cast<int64_t>(stripe.getEncoding(ek).kind())) {
        case proto::ColumnEncoding_Kind_DIRECT:
          return std::make_unique<SelectiveStringDirectColumnReader>(
              requestedType, stripe, scanSpec, std::move(flatMapContext));
        case proto::ColumnEncoding_Kind_DICTIONARY:
          return std::make_unique<SelectiveStringDictionaryColumnReader>(
              requestedType, stripe, scanSpec, std::move(flatMapContext));
        default:
          DWIO_RAISE("buildReader string unknown encoding");
      }
    case TypeKind::TIMESTAMP:
      return std::make_unique<SelectiveTimestampColumnReader>(
          requestedType, stripe, scanSpec, std::move(flatMapContext));
    default:
      DWIO_RAISE(
          "buildReader unhandled type: " +
          mapTypeKindToName(dataType->type->kind()));
  }
}

} // namespace facebook::velox::dwrf
