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

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include <thrift/protocol/TCompactProtocol.h>
#include "velox/dwio/common/MetricsLog.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/parquet/reader/StructColumnReader.h"
#include "velox/dwio/parquet/thrift/ThriftTransport.h"

namespace facebook::velox::parquet {

ReaderBase::ReaderBase(
    std::unique_ptr<dwio::common::InputStream> stream,
    const dwio::common::ReaderOptions& options)
    : pool_(options.getMemoryPool()),
      options_(options),
      stream_{std::move(stream)},
      bufferedInputFactory_(
          options.getBufferedInputFactory()
              ? options.getBufferedInputFactory()
              : dwio::common::BufferedInputFactory::baseFactoryShared()) {
  input_ = bufferedInputFactory_->create(*stream_, pool_, options.getFileNum());
  fileLength_ = stream_->getLength();
  DWIO_ENSURE(fileLength_ > 0, "Parquet file is empty");
  DWIO_ENSURE(fileLength_ >= 12, "Parquet file is too small");

  loadFileMetaData();
  initializeSchema();
}

void ReaderBase::loadFileMetaData() {
  bool preloadFile_ = fileLength_ <= FILE_PRELOAD_THRESHOLD;
  uint64_t readSize =
      preloadFile_ ? fileLength_ : std::min(fileLength_, DIRECTORY_SIZE_GUESS);

  auto stream = input_->read(
      fileLength_ - readSize, readSize, dwio::common::LogType::FOOTER);

  std::vector<char> copy(readSize);
  const char* bufferStart = nullptr;
  const char* bufferEnd = nullptr;
  dwio::common::readBytes(
      readSize, stream.get(), copy.data(), bufferStart, bufferEnd);
  DWIO_ENSURE(
      strncmp(copy.data() + readSize - 4, "PAR1", 4) == 0,
      "No magic bytes found at end of the Parquet file");

  uint32_t footerLength =
      *(reinterpret_cast<const uint32_t*>(copy.data() + readSize - 8));
  VELOX_CHECK_LT(footerLength + 12, fileLength_);
  int32_t footerOffsetInBuffer = readSize - 8 - footerLength;
  if (footerLength > readSize - 8) {
    footerOffsetInBuffer = 0;
    auto missingLength = footerLength - readSize - 8;
    stream = input_->read(
        fileLength_ - footerLength - 8,
        missingLength,
        dwio::common::LogType::FOOTER);
    copy.resize(footerLength);
    std::memmove(copy.data() + missingLength, copy.data(), readSize - 8);
    bufferStart = nullptr;
    bufferEnd = nullptr;
    dwio::common::readBytes(
        missingLength, stream.get(), copy.data(), bufferStart, bufferEnd);
  }

  auto thriftTransport = std::make_shared<thrift::ThriftBufferedTransport>(
      copy.data() + footerOffsetInBuffer, footerLength);
  auto thriftProtocol =
      std::make_unique<apache::thrift::protocol::TCompactProtocolT<
          thrift::ThriftBufferedTransport>>(thriftTransport);
  fileMetaData_ = std::make_unique<thrift::FileMetaData>();
  fileMetaData_->read(thriftProtocol.get());
}

void ReaderBase::initializeSchema() {
  if (fileMetaData_->__isset.encryption_algorithm) {
    VELOX_UNSUPPORTED("Encrypted Parquet files are not supported");
  }

  DWIO_ENSURE(
      fileMetaData_->schema.size() > 1,
      "Invalid Parquet schema: Need at least one non-root column in the file");
  DWIO_ENSURE(
      fileMetaData_->schema[0].repetition_type ==
          thrift::FieldRepetitionType::REQUIRED,
      "Invalid Parquet schema: root element must be REQUIRED");
  DWIO_ENSURE(
      fileMetaData_->schema[0].num_children > 0,
      "Invalid Parquet schema: root element must have at least 1 child");

  std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>> children;
  children.reserve(fileMetaData_->schema[0].num_children);

  uint32_t maxDefine = 0;
  uint32_t maxRepeat = 0;
  uint32_t schemaIdx = 0;
  uint32_t columnIdx = 0;
  uint32_t maxSchemaElementIdx = fileMetaData_->schema.size() - 1;
  schemaWithId_ = getParquetColumnInfo(
      maxSchemaElementIdx, maxRepeat, maxDefine, schemaIdx, columnIdx);
  schema_ = createRowType(schemaWithId_->getChildren());
}

std::shared_ptr<const ParquetTypeWithId> ReaderBase::getParquetColumnInfo(
    uint32_t maxSchemaElementIdx,
    uint32_t maxRepeat,
    uint32_t maxDefine,
    uint32_t& schemaIdx,
    uint32_t& columnIdx) const {
  DWIO_ENSURE(fileMetaData_ != nullptr);
  DWIO_ENSURE(schemaIdx < fileMetaData_->schema.size());

  auto& schema = fileMetaData_->schema;
  uint32_t curSchemaIdx = schemaIdx;
  auto& schemaElement = schema[curSchemaIdx];

  if (schemaElement.__isset.repetition_type) {
    if (schemaElement.repetition_type !=
        thrift::FieldRepetitionType::REQUIRED) {
      maxDefine++;
    }
    if (schemaElement.repetition_type ==
        thrift::FieldRepetitionType::REPEATED) {
      maxRepeat++;
    }
  }

  if (!schemaElement.__isset.type) { // inner node
    DWIO_ENSURE(
        schemaElement.__isset.num_children && schemaElement.num_children > 0,
        "Node has no children but should");

    std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>> children;

    for (int32_t i = 0; i < schemaElement.num_children; i++) {
      auto child = getParquetColumnInfo(
          maxSchemaElementIdx, maxRepeat, maxDefine, ++schemaIdx, columnIdx);
      children.push_back(child);
    }
    DWIO_ENSURE(!children.empty());

    if (schemaElement.__isset.converted_type) {
      switch (schemaElement.converted_type) {
        case thrift::ConvertedType::LIST:
        case thrift::ConvertedType::MAP: {
          auto element = children[0]->getChildren();
          DWIO_ENSURE(children.size() == 1);
          return std::make_shared<const ParquetTypeWithId>(
              children[0]->type,
              std::move(element),
              curSchemaIdx, // TODO: there are holes in the ids
              maxSchemaElementIdx,
              -1, // columnIdx,
              schemaElement.name,
              std::nullopt,
              maxRepeat,
              maxDefine);
        }
        case thrift::ConvertedType::MAP_KEY_VALUE: // child of MAP
          DWIO_ENSURE(
              schemaElement.repetition_type ==
              thrift::FieldRepetitionType::REPEATED);
          DWIO_ENSURE(children.size() == 2);
          return std::make_shared<const ParquetTypeWithId>(
              TypeFactory<TypeKind::MAP>::create(
                  children[0]->type, children[1]->type),
              std::move(children),
              curSchemaIdx, // TODO: there are holes in the ids
              maxSchemaElementIdx,
              -1, // columnIdx,
              schemaElement.name,
              std::nullopt,
              maxRepeat,
              maxDefine);
        default:
          VELOX_UNSUPPORTED(
              "Unsupported SchemaElement type: {}",
              schemaElement.converted_type);
      }
    } else {
      if (schemaElement.repetition_type ==
          thrift::FieldRepetitionType::REPEATED) {
        // child of LIST: "bag"
        DWIO_ENSURE(children.size() == 1);
        return std::make_shared<ParquetTypeWithId>(
            TypeFactory<TypeKind::ARRAY>::create(children[0]->type),
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            -1, // columnIdx,
            schemaElement.name,
            std::nullopt,
            maxRepeat,
            maxDefine);
      } else {
        // Row type
        return std::make_shared<const ParquetTypeWithId>(
            createRowType(children),
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            -1, // columnIdx,
            schemaElement.name,
            std::nullopt,
            maxRepeat,
            maxDefine);
      }
    }
  } else { // leaf node
    const auto veloxType = convertType(schemaElement);
    int32_t precision =
        schemaElement.__isset.precision ? schemaElement.precision : 0;
    int32_t scale = schemaElement.__isset.scale ? schemaElement.scale : 0;
    int32_t type_length =
        schemaElement.__isset.type_length ? schemaElement.type_length : 0;
    std::vector<std::shared_ptr<const dwio::common::TypeWithId>> children;
    std::shared_ptr<const ParquetTypeWithId> leafTypePtr =
        std::make_shared<const ParquetTypeWithId>(
            veloxType,
            std::move(children),
            curSchemaIdx,
            maxSchemaElementIdx,
            columnIdx++,
            schemaElement.name,
            schemaElement.type,
            maxRepeat,
            maxDefine,
            precision,
            scale);

    if (schemaElement.repetition_type ==
        thrift::FieldRepetitionType::REPEATED) {
      // Array
      children.reserve(1);
      children.push_back(leafTypePtr);
      return std::make_shared<const ParquetTypeWithId>(
          TypeFactory<TypeKind::ARRAY>::create(veloxType),
          std::move(children),
          curSchemaIdx,
          maxSchemaElementIdx,
          columnIdx++,
          schemaElement.name,
          std::nullopt,
          maxRepeat,
          maxDefine);
    }

    return leafTypePtr;
  }
}

TypePtr ReaderBase::convertType(
    const thrift::SchemaElement& schemaElement) const {
  DWIO_ENSURE(schemaElement.__isset.type && schemaElement.num_children == 0);
  DWIO_ENSURE(
      schemaElement.type != thrift::Type::FIXED_LEN_BYTE_ARRAY ||
          schemaElement.__isset.type_length,
      "FIXED_LEN_BYTE_ARRAY requires length to be set");

  if (schemaElement.__isset.converted_type) {
    switch (schemaElement.converted_type) {
      case thrift::ConvertedType::INT_8:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT8 converted type can only be set for value of thrift::Type::INT32");
        return TINYINT();

      case thrift::ConvertedType::INT_16:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT16 converted type can only be set for value of thrift::Type::INT32");
        return SMALLINT();

      case thrift::ConvertedType::INT_32:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT32 converted type can only be set for value of thrift::Type::INT32");
        return INTEGER();

      case thrift::ConvertedType::INT_64:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "INT64 converted type can only be set for value of thrift::Type::INT32");
        return BIGINT();

      case thrift::ConvertedType::UINT_8:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_8 converted type can only be set for value of thrift::Type::INT32");
        return TINYINT();

      case thrift::ConvertedType::UINT_16:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_16 converted type can only be set for value of thrift::Type::INT32");
        return SMALLINT();

      case thrift::ConvertedType::UINT_32:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "UINT_32 converted type can only be set for value of thrift::Type::INT32");
        return INTEGER();

      case thrift::ConvertedType::UINT_64:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT64,
            "UINT_64 converted type can only be set for value of thrift::Type::INT64");
        return TINYINT();

      case thrift::ConvertedType::DATE:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT32,
            "DATE converted type can only be set for value of thrift::Type::INT32");
        return DATE();

      case thrift::ConvertedType::TIMESTAMP_MICROS:
      case thrift::ConvertedType::TIMESTAMP_MILLIS:
        DWIO_ENSURE(
            schemaElement.type == thrift::Type::INT64,
            "TIMESTAMP_MICROS or TIMESTAMP_MILLIS converted type can only be set for value of thrift::Type::INT64");
        return TIMESTAMP();

      case thrift::ConvertedType::DECIMAL:
        DWIO_ENSURE(
            !schemaElement.__isset.precision || !schemaElement.__isset.scale,
            "DECIMAL requires a length and scale specifier!");
        VELOX_UNSUPPORTED("Decimal type is not supported yet");

      case thrift::ConvertedType::UTF8:
        switch (schemaElement.type) {
          case thrift::Type::BYTE_ARRAY:
          case thrift::Type::FIXED_LEN_BYTE_ARRAY:
            return VARCHAR();
          default:
            DWIO_RAISE(
                "UTF8 converted type can only be set for thrift::Type::(FIXED_LEN_)BYTE_ARRAY");
        }
      case thrift::ConvertedType::MAP:
      case thrift::ConvertedType::MAP_KEY_VALUE:
      case thrift::ConvertedType::LIST:
      case thrift::ConvertedType::ENUM:
      case thrift::ConvertedType::TIME_MILLIS:
      case thrift::ConvertedType::TIME_MICROS:
      case thrift::ConvertedType::JSON:
      case thrift::ConvertedType::BSON:
      case thrift::ConvertedType::INTERVAL:
      default:
        DWIO_RAISE(
            "Unsupported Parquet SchemaElement converted type: ",
            schemaElement.converted_type);
    }
  } else {
    switch (schemaElement.type) {
      case thrift::Type::type::BOOLEAN:
        return BOOLEAN();
      case thrift::Type::type::INT32:
        return INTEGER();
      case thrift::Type::type::INT64:
        return BIGINT();
      case thrift::Type::type::INT96:
        return DOUBLE(); // TODO: Lose precision
      case thrift::Type::type::FLOAT:
        return REAL();
      case thrift::Type::type::DOUBLE:
        return DOUBLE();
      case thrift::Type::type::BYTE_ARRAY:
      case thrift::Type::type::FIXED_LEN_BYTE_ARRAY:
        if (binaryAsString) {
          return VARCHAR();
        } else {
          return VARBINARY();
        }

      default:
        DWIO_RAISE("Unknown Parquet SchemaElement type: ", schemaElement.type);
    }
  }
}

std::shared_ptr<const RowType> ReaderBase::createRowType(
    std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>>
        children) {
  std::vector<std::string> childNames;
  std::vector<TypePtr> childTypes;
  for (auto& child : children) {
    childNames.push_back(
        std::static_pointer_cast<const ParquetTypeWithId>(child)->name_);
    childTypes.push_back(child->type);
  }
  return TypeFactory<TypeKind::ROW>::create(
      std::move(childNames), std::move(childTypes));
}
void ReaderBase::scheduleRowGroups(
    const std::vector<uint32_t>& rowGroupIds,
    int32_t currentGroup,
    StructColumnReader& reader) {
  auto thisGroup = rowGroupIds[currentGroup];
  auto nextGroup =
      currentGroup + 1 < rowGroupIds.size() ? rowGroupIds[currentGroup + 1] : 0;
  auto input = inputs_[thisGroup].get();
  if (!input) {
    auto newInput =
        bufferedInputFactory_->create(*stream_, pool_, options_.getFileNum());
    reader.enqueueRowGroup(thisGroup, *newInput);
    newInput->load(dwio::common::LogType::STRIPE);
    inputs_[thisGroup] = std::move(newInput);
  }
  if (nextGroup) {
    auto newInput =
        bufferedInputFactory_->create(*stream_, pool_, options_.getFileNum());
    reader.enqueueRowGroup(nextGroup, *newInput);
    newInput->load(dwio::common::LogType::STRIPE);
    inputs_[nextGroup] = std::move(newInput);
  }
  if (currentGroup > 1) {
    inputs_.erase(rowGroupIds[currentGroup - 1]);
  }
}

int64_t ReaderBase::rowGroupUncompressedSize(
    int32_t rowGroupIndex,
    const dwio::common::TypeWithId& type) const {
  if (type.column >= 0) {
    return fileMetaData_->row_groups[rowGroupIndex]
        .columns[type.column]
        .meta_data.total_uncompressed_size;
  }
  int64_t sum = 0;
  for (auto child : type.getChildren()) {
    sum += rowGroupUncompressedSize(rowGroupIndex, *child);
  }
  return sum;
}

ParquetRowReader::ParquetRowReader(
    const std::shared_ptr<ReaderBase>& readerBase,
    const dwio::common::RowReaderOptions& options)
    : pool_(readerBase->getMemoryPool()),
      readerBase_(readerBase),
      options_(options),
      rowGroups_(readerBase_->fileMetaData().row_groups),
      currentRowGroupIdsIdx_(0),
      currentRowGroupPtr_(&rowGroups_[currentRowGroupIdsIdx_]),
      rowsInCurrentRowGroup_(currentRowGroupPtr_->num_rows),
      currentRowInGroup_(rowsInCurrentRowGroup_) {
  // Validate the requested type is compatible with what's in the file
  std::function<std::string()> createExceptionContext = [&]() {
    std::string exceptionMessageContext = fmt::format(
        "The schema loaded in the reader does not match the schema in the file footer."
        "Input Stream Name: {},\n"
        "File Footer Schema (without partition columns): {},\n"
        "Input Table Schema (with partition columns): {}\n",
        readerBase_->stream().getName(),
        readerBase_->schema()->toString(),
        requestedType_->toString());
    return exceptionMessageContext;
  };

  if (rowGroups_.empty()) {
    return; // TODO
  }
  ParquetParams params(pool_, readerBase_->fileMetaData());

  columnReader_ = ParquetColumnReader::build(
      readerBase_->schemaWithId(), // Id is schema id
      params,
      *options_.getScanSpec());

  filterRowGroups();
}

//
void ParquetRowReader::filterRowGroups() {
  auto scanSpec = options_.getScanSpec();
  auto rowGroups = readerBase_->fileMetaData().row_groups;
  rowGroupIds_.reserve(rowGroups.size());
  auto excluded =
      columnReader_->filterRowGroups(0, dwio::common::StatsContext());
  skippedRowGroups_ = excluded.size();
  for (auto i = 0; i < rowGroups.size(); i++) {
    if (std::find(excluded.begin(), excluded.end(), i) == excluded.end())
      rowGroupIds_.push_back(i);
  }
}

uint64_t ParquetRowReader::next(uint64_t size, velox::VectorPtr& result) {
  DWIO_ENSURE_GT(size, 0);

  if (currentRowInGroup_ >= rowsInCurrentRowGroup_) {
    // attempt to advance to next row group
    if (!advanceToNextRowGroup()) {
      return 0;
    }
  }

  uint64_t rowsToRead = std::min(
      static_cast<uint64_t>(size), rowsInCurrentRowGroup_ - currentRowInGroup_);

  if (rowsToRead > 0) {
    columnReader_->next(rowsToRead, result, nullptr);
    currentRowInGroup_ += rowsToRead;
  }

  return rowsToRead;
}

bool ParquetRowReader::advanceToNextRowGroup() {
  if (currentRowGroupIdsIdx_ == rowGroupIds_.size()) {
    return false;
  }

  auto nextRowGroupIndex = rowGroupIds_[currentRowGroupIdsIdx_];
  readerBase_->scheduleRowGroups(
      rowGroupIds_,
      currentRowGroupIdsIdx_,
      *reinterpret_cast<StructColumnReader*>(columnReader_.get()));
  currentRowGroupPtr_ = &rowGroups_[rowGroupIds_[currentRowGroupIdsIdx_]];
  rowsInCurrentRowGroup_ = currentRowGroupPtr_->num_rows;
  currentRowInGroup_ = 0;
  currentRowGroupIdsIdx_++;
  columnReader_->seekToRowGroup(nextRowGroupIndex);
  return true;
}

void ParquetRowReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& stats) const {
  stats.skippedStrides += skippedRowGroups_;
}

void ParquetRowReader::resetFilterCaches() {
  columnReader_->resetFilterCaches();
}

std::optional<size_t> ParquetRowReader::estimatedRowSize() const {
  auto index =
      currentRowGroupIdsIdx_ < 1 ? 0 : rowGroupIds_[currentRowGroupIdsIdx_ - 1];
  return readerBase_->rowGroupUncompressedSize(
             index, *readerBase_->schemaWithId()) /
      rowsInCurrentRowGroup_;
}

ParquetReader::ParquetReader(
    std::unique_ptr<dwio::common::InputStream> stream,
    const dwio::common::ReaderOptions& options)
    : readerBase_(std::make_shared<ReaderBase>(std::move(stream), options)) {}

std::unique_ptr<dwio::common::RowReader> ParquetReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<ParquetRowReader>(readerBase_, options);
}
} // namespace facebook::velox::parquet
