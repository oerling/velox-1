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

#include "velox/exec/OperatorUtils.h"
#include "velox/exec/VectorHasher.h"
#include "velox/expression/EvalCtx.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

namespace {

template <TypeKind kind>
void scalarGatherCopy(
    BaseVector* target,
    vector_size_t targetIndex,
    vector_size_t count,
    const std::vector<const RowVector*>& sources,
    const std::vector<vector_size_t>& sourceIndices,
    column_index_t sourceColumnChannel) {
  VELOX_DCHECK(target->isFlatEncoding());

  using T = typename TypeTraits<kind>::NativeType;
  auto* flatVector = target->template asUnchecked<FlatVector<T>>();
  uint64_t* rawNulls = nullptr;
  if (std::is_same_v<T, StringView>) {
    for (int i = 0; i < count; ++i) {
      VELOX_DCHECK(!sources[i]->mayHaveNulls());
      if (sources[i]
              ->childAt(sourceColumnChannel)
              ->isNullAt(sourceIndices[i])) {
        if (FOLLY_UNLIKELY(rawNulls == nullptr)) {
          rawNulls = target->mutableRawNulls();
        }
        bits::setNull(rawNulls, targetIndex + i, true);
        continue;
      }
      auto* source = sources[i]->childAt(sourceColumnChannel).get();
      flatVector->setNoCopy(
          targetIndex + i,
          source->template asUnchecked<FlatVector<T>>()->valueAt(
              sourceIndices[i]));
      flatVector->acquireSharedStringBuffers(source);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      VELOX_DCHECK(!sources[i]->mayHaveNulls());
      if (sources[i]
              ->childAt(sourceColumnChannel)
              ->isNullAt(sourceIndices[i])) {
        if (FOLLY_UNLIKELY(rawNulls == nullptr)) {
          rawNulls = target->mutableRawNulls();
        }
        bits::setNull(rawNulls, targetIndex + i, true);
        continue;
      }
      flatVector->set(
          targetIndex + i,
          sources[i]
              ->childAt(sourceColumnChannel)
              ->template asUnchecked<FlatVector<T>>()
              ->valueAt(sourceIndices[i]));
    }
  }
}

void complexGatherCopy(
    BaseVector* target,
    vector_size_t targetIndex,
    vector_size_t count,
    const std::vector<const RowVector*>& sources,
    const std::vector<vector_size_t>& sourceIndices,
    column_index_t sourceChannel) {
  for (int i = 0; i < count; ++i) {
    target->copy(
        sources[i]->childAt(sourceChannel).get(),
        targetIndex + i,
        sourceIndices[i],
        1);
  }
}

void gatherCopy(
    BaseVector* target,
    vector_size_t targetIndex,
    vector_size_t count,
    const std::vector<const RowVector*>& sources,
    const std::vector<vector_size_t>& sourceIndices,
    column_index_t sourceChannel) {
  if (target->isScalar()) {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        scalarGatherCopy,
        target->type()->kind(),
        target,
        targetIndex,
        count,
        sources,
        sourceIndices,
        sourceChannel);
  } else {
    complexGatherCopy(
        target, targetIndex, count, sources, sourceIndices, sourceChannel);
  }
}
} // namespace

void deselectRowsWithNulls(
    const std::vector<std::unique_ptr<VectorHasher>>& hashers,
    SelectivityVector& rows) {
  bool anyChange = false;
  for (int32_t i = 0; i < hashers.size(); ++i) {
    auto& decoded = hashers[i]->decodedVector();
    if (decoded.mayHaveNulls()) {
      anyChange = true;
      const auto* nulls = hashers[i]->decodedVector().nulls();
      bits::andBits(rows.asMutableRange().bits(), nulls, 0, rows.end());
    }
  }

  if (anyChange) {
    rows.updateBounds();
  }
}

uint64_t* FilterEvalCtx::getRawSelectedBits(
    vector_size_t size,
    memory::MemoryPool* pool) {
  uint64_t* rawBits;
  BaseVector::ensureBuffer<bool, uint64_t>(size, pool, &selectedBits, &rawBits);
  return rawBits;
}

vector_size_t* FilterEvalCtx::getRawSelectedIndices(
    vector_size_t size,
    memory::MemoryPool* pool) {
  vector_size_t* rawSelected;
  BaseVector::ensureBuffer<vector_size_t>(
      size, pool, &selectedIndices, &rawSelected);
  return rawSelected;
}

namespace {
vector_size_t processConstantFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows) {
  auto constant = filterResult->as<ConstantVector<bool>>();
  if (constant->isNullAt(0) || constant->valueAt(0) == false) {
    return 0;
  }
  return rows.size();
}

vector_size_t processFlatFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  auto size = rows.size();

  auto selectedBits = filterEvalCtx.getRawSelectedBits(size, pool);
  auto nonNullBits =
      filterResult->as<FlatVector<bool>>()->rawValues<uint64_t>();
  if (filterResult->mayHaveNulls()) {
    bits::andBits(selectedBits, nonNullBits, filterResult->rawNulls(), 0, size);
  } else {
    memcpy(selectedBits, nonNullBits, bits::nbytes(size));
  }

  vector_size_t passed = 0;
  auto* rawSelected = filterEvalCtx.getRawSelectedIndices(size, pool);
  bits::forEachSetBit(
      selectedBits, 0, size, [&rawSelected, &passed](vector_size_t row) {
        rawSelected[passed++] = row;
      });
  return passed;
}

vector_size_t processEncodedFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  auto size = rows.size();

  DecodedVector& decoded = filterEvalCtx.decodedResult;
  decoded.decode(*filterResult.get(), rows);
  auto values = decoded.data<uint64_t>();
  auto nulls = decoded.nulls();
  auto indices = decoded.indices();

  vector_size_t passed = 0;
  auto* rawSelected = filterEvalCtx.getRawSelectedIndices(size, pool);
  auto* rawSelectedBits = filterEvalCtx.getRawSelectedBits(size, pool);
  memset(rawSelectedBits, 0, bits::nbytes(size));
  for (int32_t i = 0; i < size; ++i) {
    auto index = indices[i];
    if ((!nulls || !bits::isBitNull(nulls, i)) &&
        bits::isBitSet(values, index)) {
      rawSelected[passed++] = i;
      bits::setBit(rawSelectedBits, i);
    }
  }
  return passed;
}
} // namespace

vector_size_t processFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  switch (filterResult->encoding()) {
    case VectorEncoding::Simple::CONSTANT:
      return processConstantFilterResults(filterResult, rows);
    case VectorEncoding::Simple::FLAT:
      return processFlatFilterResults(filterResult, rows, filterEvalCtx, pool);
    default:
      return processEncodedFilterResults(
          filterResult, rows, filterEvalCtx, pool);
  }
}

struct WrapState {
  std::vector<Buffer*> previousIndices;
  std::vector<BufferPtr> newIndices;
};

void transposeWithNulls(
    const vector_size_t* base,
    const uint64_t* nulls,
    vector_size_t size,
    const vector_size_t* indices,
    const uint64_t* extraNulls,
    vector_size_t* result,
    uint64_t* resultNulls) {
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  for (auto i = 0; i < size; i += kBatch) {
    auto indexBatch = xsimd::load_unaligned(indices + i);
    uint8_t extraNullsByte = i + kBatch > size ? bits::lowMask(size - i) : 0xff;

    if (extraNulls) {
      extraNullsByte &= reinterpret_cast<const uint8_t*>(extraNulls)[i / 8];
    }
    if (extraNullsByte != 0xff) {
      auto mask = simd::fromBitMask<int32_t>(extraNullsByte);
      indexBatch = indexBatch &
          xsimd::load_unaligned(reinterpret_cast<const vector_size_t*>(&mask));
    }
    uint8_t flags = simd::gather8Bits(nulls, indexBatch, 8);
    flags &= extraNullsByte;
    reinterpret_cast<uint8_t*>(resultNulls)[i / 8] = flags;
    simd::gather<int32_t>(base, indexBatch).store_unaligned(result + i);
  }
}

void transpose(
    const vector_size_t* base,
    vector_size_t size,
    const vector_size_t* indices,
    vector_size_t* result) {
  constexpr int32_t kBatch = xsimd::batch<int32_t>::size;
  int32_t i = 0;
  for (; i + kBatch <= size; i += kBatch) {
    auto indexBatch = xsimd::load_unaligned(indices + i);
    simd::gather(base, indexBatch).store_unaligned(result + i);
  }
  if (i < size) {
    auto indexBatch = xsimd::load_unaligned(indices + i);
    auto mask = simd::leadingMask<int32_t>(size - i);
    simd::maskGather(
        xsimd::batch<int32_t>::broadcast(0), mask, base, indexBatch)
        .store_unaligned(result + i);
  }
}

VectorPtr wrapOne(
    vector_size_t size,
    BufferPtr mapping,
    const VectorPtr& vector,
    BufferPtr extraNulls,
    WrapState& state) {
  if (!mapping) {
    return vector;
  }
  if (vector->encoding() != VectorEncoding::Simple::DICTIONARY) {
    return BaseVector::wrapInDictionary(extraNulls, mapping, size, vector);
  }
  auto indices = vector->wrapInfo();
  auto base = vector->valueVector();
  for (auto i = 0; i < state.previousIndices.size(); ++i) {
    if (indices.get() == state.previousIndices[i]) {
      return BaseVector::wrapInDictionary(
          nullptr, state.newIndices[i], size, vector);
    }
    if (const uint64_t* rawNulls = vector->rawNulls()) {
      // Dictionary adds nulls.
      BufferPtr newIndices =
          AlignedBuffer::allocate<vector_size_t>(size, vector->pool());
      BufferPtr newNulls = AlignedBuffer::allocate<bool>(size, vector->pool());
      const uint64_t* rawExtraNulls =
          extraNulls ? extraNulls->as<uint64_t>() : nullptr;
      transposeWithNulls(
          indices->as<vector_size_t>(),
          rawNulls,
          size,
          mapping->as<vector_size_t>(),
          rawExtraNulls,
          newIndices->asMutable<vector_size_t>(),
          newNulls->asMutable<uint64_t>());

      return BaseVector::wrapInDictionary(newNulls, newIndices, size, base);
    }
  }
  auto newIndices =
      AlignedBuffer::allocate<vector_size_t>(size, vector->pool());
  transpose(
      indices->as<vector_size_t>(),
      size,
      mapping->as<vector_size_t>(),
      newIndices->asMutable<vector_size_t>());
  state.previousIndices.push_back(indices.get());
  state.newIndices.push_back(newIndices);
  return BaseVector::wrapInDictionary(extraNulls, newIndices, size, vector);
}

VectorPtr wrapChild(
    vector_size_t size,
    BufferPtr mapping,
    const VectorPtr& child,
    BufferPtr nulls) {
  if (!mapping) {
    return child;
  }

  return BaseVector::wrapInDictionary(nulls, mapping, size, child);
}

RowVectorPtr
wrap(vector_size_t size, BufferPtr mapping, const RowVectorPtr& vector) {
  if (!mapping) {
    return vector;
  }

  return wrap(
      size,
      std::move(mapping),
      asRowType(vector->type()),
      vector->children(),
      vector->pool());
}

RowVectorPtr wrap(
    vector_size_t size,
    BufferPtr mapping,
    const RowTypePtr& rowType,
    const std::vector<VectorPtr>& childVectors,
    memory::MemoryPool* pool) {
  if (mapping == nullptr) {
    return RowVector::createEmpty(rowType, pool);
  }
  std::vector<VectorPtr> wrappedChildren;
  wrappedChildren.reserve(childVectors.size());
  for (auto& child : childVectors) {
    wrappedChildren.emplace_back(wrapChild(size, mapping, child));
  }
  return std::make_shared<RowVector>(
      pool, rowType, nullptr, size, wrappedChildren);
}

void loadColumns(const RowVectorPtr& input, core::ExecCtx& execCtx) {
  LocalDecodedVector decodedHolder(execCtx);
  LocalSelectivityVector baseRowsHolder(&execCtx);
  LocalSelectivityVector rowsHolder(&execCtx);
  SelectivityVector* rows = nullptr;
  for (auto& child : input->children()) {
    if (isLazyNotLoaded(*child)) {
      if (!rows) {
        rows = rowsHolder.get(input->size());
        rows->setAll();
      }
      LazyVector::ensureLoadedRows(
          child,
          *rows,
          *decodedHolder.get(),
          *baseRowsHolder.get(input->size()));
    }
  }
}

void gatherCopy(
    RowVector* target,
    vector_size_t targetIndex,
    vector_size_t count,
    const std::vector<const RowVector*>& sources,
    const std::vector<vector_size_t>& sourceIndices,
    const std::vector<IdentityProjection>& columnMap) {
  VELOX_DCHECK_GE(count, 0);
  if (FOLLY_UNLIKELY(count <= 0)) {
    return;
  }
  VELOX_CHECK_LE(count, sources.size());
  VELOX_CHECK_LE(count, sourceIndices.size());
  VELOX_DCHECK_EQ(sources.size(), sourceIndices.size());
  if (!columnMap.empty()) {
    for (const auto& columnProjection : columnMap) {
      gatherCopy(
          target->childAt(columnProjection.outputChannel).get(),
          targetIndex,
          count,
          sources,
          sourceIndices,
          columnProjection.inputChannel);
    }
  } else {
    for (auto i = 0; i < target->type()->size(); ++i) {
      gatherCopy(
          target->childAt(i).get(),
          targetIndex,
          count,
          sources,
          sourceIndices,
          i);
    }
  }
}

std::string makeOperatorSpillPath(
    const std::string& spillDir,
    int pipelineId,
    int driverId,
    int32_t operatorId) {
  VELOX_CHECK(!spillDir.empty());
  return fmt::format("{}/{}_{}_{}", spillDir, pipelineId, driverId, operatorId);
}

void addOperatorRuntimeStats(
    const std::string& name,
    const RuntimeCounter& value,
    std::unordered_map<std::string, RuntimeMetric>& stats) {
  if (UNLIKELY(stats.count(name) == 0)) {
    stats.insert(std::pair(name, RuntimeMetric(value.unit)));
  } else {
    VELOX_CHECK_EQ(stats.at(name).unit, value.unit);
  }
  stats.at(name).addValue(value.value);
}

} // namespace facebook::velox::exec
