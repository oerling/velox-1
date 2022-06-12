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

//
// Created by Ying Su on 2/14/22.
//

#include "/velox/dwio/parquet/reader/NativeParquetColumnReader.h"
#include <dwio/dwrf/reader/SelectiveColumnReaderInternal.h>

#include "/velox/dwio/parquet/reader/StructColumnReader.h"
#include "/velox/dwio/parquet/reader/IntegerColumnReader.h"


#include <type/Type.h>
#include "ParquetThriftTypes.h"
#include "ReaderUtil.h"
#include "Statistics.h"


namespace facebook::velox::parquet {

  // static 
std::unique_ptr<ParquetColumnReader> ParquetColumnReader::build(
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    ParquetParams& params,
    common::ScanSpec* scanSpec) {
  auto colName = scanSpec->fieldName();

  switch (dataType->type->kind()) {
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
      return std::make_unique<IntegerColumnReader>(dataType, params, scanSpec);
    case TypeKind::ROW:
      return std::make_unique<StructColumnReader>(dataType, params, scanSpec);

    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::BOOLEAN:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      VELOX_UNSUPPORTED("Type is not supported: ", dataType->type->kind());
    default:
      DWIO_RAISE(
          "buildReader unhandled type: " +
          mapTypeKindToName(dataType->type->kind()));
  }
}

void ParquetData::enqueueRowGroup(
    uint32_t idex,
    BufferedInput& input) {
  auto chunk = columnChunks_[index];
  streams[index] = input.enqueue({start, size}, type_->index);
  DWIO_ENSURE(
      columnChunk_->__isset.meta_data,
      "ColumnMetaData does not exist for schema Id ",
      nodeType_->id);
  auto columnMetaData = &columnChunk_->meta_data;
  DWIO_ENSURE(
      columnChunk_->__isset.meta_data,
      "ColumnMetaData does not exist for schema Id ",
      nodeType_->id);
  columnMetaData_ = &columnChunk_->meta_data;
  //  valuesInColumnChunk_ = columnMetaData_->num_values;

  chunkReadOffset_ = columnMetaData_->data_page_offset;
  if (columnMetaData_->__isset.dictionary_page_offset &&
      columnMetaData_->dictionary_page_offset >= 4) {
    // this assumes the data pages follow the dict pages directly.
    chunkReadOffset_ = columnMetaData_->dictionary_page_offset;
  }
  DWIO_ENSURE(
      chunkReadOffset_ >= 0, "Invalid chunkReadOffset_ ", chunkReadOffset_);

  uint64_t readSize = std::min(
      columnMetaData_->total_compressed_size,
      columnMetaData_->total_uncompressed_size);

  auto strea streams.bufferedInput.enqueue({chunkReadOffset_, readSize});
  streams.streams[fileColumnIndex] = std::move(

}

void ParquetColumnReader::openRowGroup(
    const RowGroup& rowGroup,
    StreamSet& streams) {
  currentRowGroup_ = &rowGroup;
  rowsInRowGroup_ = currentRowGroup_->num_rows;

  uint32_t fileColumnId = nodeType_->column;
  DWIO_ENSURE(fileColumnId < rowGroup.columns.size());
  columnChunk_ = &rowGroup.columns[fileColumnId];
  DWIO_ENSURE(columnChunk_ != nullptr);
if (columnMetaData_->__isset.statistics) {
  columnChunkStats_ = &columnMetaData_->statistics;

  }




}
}

void ParquetLeafColumnReader::openRowGroup(
    const RowGroup& rowGroup,
    StreamSet& streams) {
  uint32_t fileColumnId = nodeType_->column;
  auto stream = std::move(streams.streams[fileColumnId]);
  decoder_->reset(std::move(stream));
}

  bool ParquetData::filterMatches(const RowGroup& rowGroup, common::Filter& filter) {
    auto colIdx = nodeType_->column;
    auto type = type_->type;
    if (rowGroup.columns[colIdx].__isset.meta_data &&
        rowGroup.columns[colIdx].meta_data.__isset.statistics) {
      auto columnStats = buildColumnStatisticsFromThrift(
							 rowGroup.columns[colIdx].meta_data.statistics,
							 *type,
							 rowGroup.num_rows);
      if (!testFilter(
		      &filter,
		      columnStats.get(),
		      rowGroup.num_rows,
		      type)) {
        return false;
      }
    }
    return true;
  }

void ParquetData::prepareRead(RowSet& rows) {
  numRowsToRead_ = rows.back() + 1;
  if (numRowsToRead_ > 0) {
    if (maxRepeat_ > 0) {
      dwrf::detail::ensureCapacity<uint8_t>(
          repeatOutBuffer_, numRowsToRead_, &memoryPool_);
      repeatOutBuffer_->setSize(0);
    }

    if (maxDefine_ > 0 && !canNotHaveNull()) {
      dwrf::detail::ensureCapacity<uint8_t>(
          defineOutBuffer_, numRowsToRead_, &memoryPool_);
      defineOutBuffer_->setSize(0);

  }

}
}


std::unique_ptr<dwrf::SeekableInputStream>
ParquetLeafColumnReader::getPageStream(
    int64_t compressedSize,
    int64_t unCompressedSize) {
  auto stream = input_.read(
      chunkReadOffset_, compressedSize, dwio::common::LogType::BLOCK);

  if (columnChunk_->meta_data.codec != CompressionCodec::UNCOMPRESSED) {
    dwrf::CompressionKind kind;
    switch (columnMetaData_->codec) {
      case CompressionCodec::UNCOMPRESSED:
        kind = dwrf::CompressionKind::CompressionKind_NONE;
        break;
      case CompressionCodec::GZIP: {
        kind = dwrf::CompressionKind::CompressionKind_ZLIB;
        break;
      }
      case CompressionCodec::SNAPPY: {
        kind = dwrf::CompressionKind::CompressionKind_SNAPPY;
        break;
      }
      case CompressionCodec::ZSTD: {
        kind = dwrf::CompressionKind::CompressionKind_ZSTD;
        break;
      }
      default:
        DWIO_RAISE(
            "Unsupported Parquet compression type ", columnMetaData_->codec);
    }

    stream = createDecompressor(
        kind,
        std::move(stream),
        unCompressedSize,
        memoryPool_,
        "Data Page",
        nullptr);
  }

  return stream;
}

bool ParquetLeafColumnReader::canNotHaveNull() {
  if (maxDefine_ == 0 ||
      (columnChunkStats_ != nullptr && columnChunkStats_->__isset.null_count &&
       columnChunkStats_->null_count == 0) ||
      // TODO: confirm columnMetaData_->num_values doesn't contain nulls
      columnMetaData_->num_values == currentRowGroup_->num_rows) {
    return true;
  }
  return false;
}


template <typename T>
void ParquetVisitorIntegerColumnReader::prepareRead(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  ParquetLeafColumnReader::prepareRead(rows);

  seekTo(offset, scanSpec_->readsNullsOnly());
  vector_size_t numRows = rows.back() + 1;
  innerNonNullRows_.clear();
  outerNonNullRows_.clear();
  valueSize_ = sizeof(T);
  ensureValuesCapacity<T>(rows.size());
}

void ParquetVisitorIntegerColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  VELOX_WIDTH_DISPATCH(
      sizeOfIntKind(type_->kind()), prepareRead, offset, rows, incomingNulls);

  uint64_t rowsRead = 0; // offset in rows
  auto rowsIter = rows.begin();

  while (numRowsToRead_ > 0) {
    if (remainingRowsInPage_ == 0) {
      readNextPage();
    }

    auto batchSize = std::min(numRowsToRead_, remainingRowsInPage_);

    if (maxRepeat_ > 0) {
      repeatDecoder_->next(repeatOutBuffer_, rows, batchSize);
      // TODO: calculate nested structure
    }

    if (maxDefine_ > 0) {
      if (canNotHaveNull()) {
        defineDecoder_->skip(batchSize);
        anyNulls_ = false;
        allNull_ = false;
      } else {
        defineDecoder_->next(defineOutBuffer_, rows, batchSize);
        uint32_t nullCount = decodeNulls(
            rowsRead,
            batchSize,
            maxDefine_,
            defineOutBuffer_,
            nullsInReadRange_);
        SelectiveColumnReader::prepareNulls(rows, nullCount > 0);
      }
    }

    if (false /*has dictionary*/) {
      // TO BE IMPLEMENTED
    } else {
      // binary search
      auto endIter =
          std::upper_bound(rowsIter, rows.end(), rowsRead + batchSize);
      RowSet rowsBatch(rowsIter, endIter);
      rowsIter = endIter;

      bool isDense = rowsBatch.back() == rowsBatch.size() - 1;
      // TODO: Why assign to alwaysTrue since it'll be checked again in
      // fixedWidthScan?
      common::Filter* filter =
          scanSpec_->filter() ? scanSpec_->filter() : &dwrf::alwaysTrue();
      if (scanSpec_->keepValues()) {
        if (scanSpec_->valueHook()) {
          // Not implemented
          return;
        }
        if (isDense) {
          processFilter<true>(filter, dwrf::ExtractToReader(this), rowsBatch);
        } else {
          processFilter<false>(filter, dwrf::ExtractToReader(this), rowsBatch);
        }
      } else {
        if (isDense) {
          processFilter<true>(filter, dwrf::DropValues(), rowsBatch);
        } else {
          processFilter<false>(filter, dwrf::DropValues(), rowsBatch);
        }
      }
    }

    remainingRowsInPage_ -= batchSize;
    numRowsToRead_ -= batchSize;
    rowsRead += batchSize;
  }
}

template <bool isDense, typename ExtractValues>
void ParquetVisitorIntegerColumnReader::processFilter(
    common::Filter* filter,
    ExtractValues extractValues,
    RowSet rows) {
  switch (filter ? filter->kind() : common::FilterKind::kAlwaysTrue) {
    case common::FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kIsNull:
      filterNulls<int8_t>(
          rows,
          true,
          !std::is_same<decltype(extractValues), dwrf::DropValues>::value);
      break;
    case common::FilterKind::kIsNotNull:
      if (std::is_same<decltype(extractValues), dwrf::DropValues>::value) {
        filterNulls<int8_t>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case common::FilterKind::kBigintRange:
      readHelper<common::BigintRange, isDense>(filter, rows, extractValues);
      break;
    case common::FilterKind::kBigintValuesUsingBitmask:
      readHelper<common::BigintValuesUsingBitmask, isDense>(
          filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

template <typename TFilter, bool isDense, typename ExtractValues>
void ParquetVisitorIntegerColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  switch (valueSize_) {
    case 2:
      readWithVisitor(
          rows,
          dwrf::ColumnVisitor<int16_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
      break;

    case 4:
      readWithVisitor(
          rows,
          dwrf::ColumnVisitor<int32_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
      break;

    case 8:
      readWithVisitor(
          rows,
          dwrf::ColumnVisitor<int64_t, TFilter, ExtractValues, isDense>(
              *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
      break;
    default:
      VELOX_FAIL("Unsupported valueSize_ {}", valueSize_);
  }
}

template <typename ColumnVisitor>
void ParquetVisitorIntegerColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  if (nullsInReadRange_) {
    valuesDecoder_->readWithVisitor<true>(
        nullsInReadRange_->as<uint64_t>(), visitor);
  } else {
    valuesDecoder_->readWithVisitor<false>(nullptr, visitor);
  }
  readOffset_ += numRows;
}


bool ParquetStructColumnReader::filterMatches(const RowGroup& rowGroup) {
  bool matched = true;

  auto& childSpecs = scanSpec_->children();
  assert(!children_.empty());
  for (size_t i = 0; i < childSpecs.size(); ++i) {
    auto& childSpec = childSpecs[i];
    if (childSpec->isConstant()) {
      // TODO: match constant
      continue;
    }
    auto fieldIndex = childSpec->subscript();
    auto reader = children_.at(fieldIndex).get();
    //    auto colName = childSpec->fieldName();

    if (!reader->filterMatches(rowGroup)) {
      matched = false;
      break;
    }
  }
  return matched;
}

//  for (auto& childSpec : options_.getScanSpec()->children()) {
//    if (childSpec->filter() != nullptr) {
//      auto schema = readerBase_->getSchema();
//      auto colName = childSpec->fieldName();
//      uint32_t colIdx = schema->getChildIdx(colName);
//      auto type = schema->findChild(colName);
//      if (rowGroup.columns[colIdx].__isset.meta_data &&
//      rowGroup.columns[colIdx].meta_data.__isset.statistics) {
//        auto columnStats = buildColumnStatisticsFromThrift(
//            rowGroup.columns[colIdx].meta_data.statistics,
//            *type,
//            rowGroup.num_rows);
//        if (!testFilter(
//            childSpec->filter(),
//            columnStats.get(),
//            rowGroup.num_rows,
//            type)) {
//          matched = false;
//          break;
//        }
//      }
//    }
//    if (matched) {
//      rowGroupIds_.push_back(i);
//    }
//  }
//}

void ParquetStructColumnReader::initializeRowGroup(const RowGroup& rowGroup) {
  for (auto& child : children_) {
    child->initializeRowGroup(rowGroup);
  }
}

uint64_t ParquetStructColumnReader::skip(uint64_t numRows) {
  return 0;
}

void ParquetStructColumnReader::next(
    uint64_t numRows,
    VectorPtr& result,
    const uint64_t* nulls) {
  VELOX_CHECK(!nulls, "next may only be called for the root reader.");
  if (children_.empty()) {
    // no readers
    // This can be either count(*) query or a query that select only
    // constant columns (partition keys or columns missing from an old file
    // due to schema evolution)
    result->resize(numRows);

    auto resultRowVector = std::dynamic_pointer_cast<RowVector>(result);
    auto& childSpecs = scanSpec_->children();
    for (auto& childSpec : childSpecs) {
      VELOX_CHECK(childSpec->isConstant());
      auto channel = childSpec->channel();
      resultRowVector->childAt(channel) =
          BaseVector::wrapInConstant(numRows, 0, childSpec->constantValue());
    }
  } else {
    auto oldSize = rows_.size();
    rows_.resize(numRows);
    if (numRows > oldSize) {
      std::iota(&rows_[oldSize], &rows_[rows_.size()], oldSize);
    }
    read(readOffset_, rows_, nullptr);
    getValues(outputRows(), &result);
  }
}

void facebook::velox::parquet::ParquetStructColumnReader::prepareRead(
    uint64_t numRows) {
  selectivityVec_.reset(numRows);
  numRowsToRead_ = 0;
  numReads_ = scanSpec_->newRead();
} // namespace facebook::velox::parquet

void ParquetStructColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  prepareRead(rows.size());

  RowSet activeRows = rows;
  auto& childSpecs = scanSpec_->children();

  bool hasFilter = false;
  assert(!children_.empty());
  for (size_t i = 0; i < childSpecs.size(); ++i) {
    auto& childSpec = childSpecs[i];
    if (childSpec->isConstant()) {
      continue;
    }
    auto fieldIndex = childSpec->subscript();
    auto reader = children_.at(fieldIndex).get();
    //    if (reader->isTopLevel() && childSpec->projectOut() &&
    //    !childSpec->filter() && !childSpec->extractValues()) {
    //      // Will make a LazyVector.
    //      continue;
    //    }

    if (childSpec->filter()) {
      hasFilter = true;
      {
        SelectivityTimer timer(childSpec->selectivity(), activeRows.size());

        reader->resetInitTimeClocks();
        reader->read(offset, activeRows, nullptr);

        // Exclude initialization time.
        timer.subtract(reader->initTimeClocks());

        activeRows = reader->outputRows();
        childSpec->selectivity().addOutput(activeRows.size());
      }
      if (activeRows.empty()) {
        break;
      }
    } else {
      reader->read(offset, activeRows, nullptr);
    }
  }
  if (hasFilter) {
    setOutputRows(activeRows);
  }
  //  lazyVectorReadOffset_ = offset;
  readOffset_ = offset + rows.back() + 1;
}

void ParquetStructColumnReader::getValues(RowSet rows, VectorPtr* result) {
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
      //      if (!childSpec->extractValues() && !childSpec->filter() &&
      //      children_[index]->isTopLevel()) {
      //        // LazyVector result.
      //        if (!lazyPrepared) {
      //          if (rows.size() != outputRows_.size()) {
      //            setOutputRows(rows);
      //          }
      //          lazyPrepared = true;
      //        }
      //        resultRow->childAt(channel) = std::make_shared<LazyVector>(
      //            &memoryPool_,
      //            resultRow->type()->childAt(channel),
      //            rows.size(),
      //            std::make_unique<ColumnLoader>(
      //                this, children_[index].get(), numReads_));
      //      } else {
      children_[index]->getValues(rows, &resultRow->childAt(channel));
      //      }
    }
  }
}

} // namespace facebook::velox::parquet
