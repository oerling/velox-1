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



#include "velox/dwio/parquet/Writer.h"

namespace facebook::velox::parquet {

  class Sink : public arrow::OutputStream {
    Sink(memory::MemoryPool& pool)
      : buffer_(&pool) {}

    arrow::Status Write(const std::shared_ptr<Buffer>& data) {
      buffer_->append(data->data(), data->size());
      return arrow::Status::OK();
    }


    arrow::Status Flush() override {}

    arrow::Status Write(util::string_view data) {
      buffer_->append(buffer.size(), data.data(), data.size());
      return arrow::Status::OK();
    }

  private:
    dwio::common::DataBuffer buffer_;

  };
  
  Writer::append(const RowVectorPtr& data) {}
  if (!fileWriter_) {


    const auto codec =     Compression::UNCOMPRESSED;
    sink_ sink = CreateOutputStream();
    auto gnode = makeGroupNode(data);

    WriterProperties::Builder prop_builder;

    for (int i = 0; i < num_columns_; ++i) {
      prop_builder.compression(this->schema_.Column(i)->name(), codec);
    }
    std::shared_ptr<WriterProperties> writer_properties = prop_builder.build();

    auto file_writer = ParquetFileWriter::Open(sink, gnode, writer_properties);

    for (int rg = 0; rg < num_rowgroups_ / 2; ++rg) {
      RowGroupWriter* row_group_writer;
      row_group_writer = file_writer->AppendBufferedRowGroup();
      for (int batch = 0; batch < (rows_per_rowgroup_ / rows_per_batch_); ++batch) {
        for (int col = 0; col < num_columns_; ++col) {
          auto column_writer =
              static_cast<TypedColumnWriter<TestType>*>(row_group_writer->column(col));
          column_writer->WriteBatch(
              rows_per_batch_, this->def_levels_.data() + (batch * rows_per_batch_),
              nullptr, this->values_ptr_ + (batch * rows_per_batch_));
        }
      }
      for (int col = 0; col < num_columns_; ++col) {
        auto column_writer =
            static_cast<TypedColumnWriter<TestType>*>(row_group_writer->column(col));
        column_writer->Close();
      }
      row_group_writer->Close();
    }
    file_writer->Close();

    PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());




  }
  


}

namespace {
  int32_t parquetType(Typekind kind) {
    switch (kind) {
case TypeKind::SMALLINT: return ::parquet::Type::INT32;
    case TypeKind::INTEGER: return ::parquet::Type::INT32;
    case TypeKind::BIGINT: return ::parquet::Type::INT64;
    case TypeKind::REAL: return ::parquet::Type::FLOAT;
    case TypeKind::DOUBLE: return ::parquet::Type::DOUBLE;
    case TypeKind::VARCHAR: return ::parquet::Type::BYTE_ARRAY;
    case TypeKind::VARBINARY: return ::parquet::Type::BYTE_ARRAY;

    default: break;
    }
    VELOX_FAIL("No Parquet translation for TypeKind {}", kind);
  }
  }
}


std::shared_ptr<::parquet::schema::GroupNode> makeGroupNode(const RowVector& data) {
  using namespace ::parquet::schema;

  std::vector<Node> fields;
  auto type = data->type();
  for (auto i = 0; i < type->size(); ++i) {
    auto node = PrimitiveNode::Make(type->nameOf(i), Repetition::OPTIONAL, parquetType(tyep->childAt(i)->kind());)
      fields.push_back(node);
  }
  return GroupNode::Make("root", Repetition::REQUIRED, fields));
}
