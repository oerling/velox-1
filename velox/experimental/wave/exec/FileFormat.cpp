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

#include "velox/experimental/wave/exec/FileFormat.h"

namespace facebook::velox::wave {

std::mutex Table::mutex_;
std::unordered_map<std::string, std::unique_ptr<Table>> Table::allTables_;

int32_t bitWidth(uint64_t max) {
  return 64 - __builtin_clzll(max);
}

template <typename T>
BufferPtr
encodeInts(const std::vector<T>& ints, uint64_t max, memory::MemoryPool* pool) {
  int32_t width = bitWidth(max);
  int32_t size = bits::roundUp(ints.size() * width, 128) / 128;
  auto buffer = AlignedBuffer::allocate<char>(size, pool);
  auto destination = buffer->asMutable<uint64_t>();
  auto source = reinterpret_cast<const uint64_t*>(ints.data());
  int32_t sourceWidth = sizeof(T) * 8;
  for (auto i = 0; i < ints.size(); ++i) {
    bits::copyBits(source, i * sourceWidth, destination, i * width, width);
  }
  return buffer;
}

int64_t Encoder::flatSize() {
  if (kind_ == TypeKind::VARCHAR) {
    return totalStringBytes_ + (count_ * bitWidth(maxLength_) / 8);
  }
  return count_ * bitWidth(max_) / 8;
}

int64_t Encoder::dictSize() {
  if (kind_ == TypeKind::VARCHAR) {
    return (count_ * bitWidth(strings_.size() - 1) / 8) + dictBytes_;
  }
  return (bitWidth(max_) * ints_.size() / 8) +
      (bitWidth(ints_.size() - 1) * count_ / 8);
}

struct StringWithId {
  StringView string;
  int32_t id;
};

template <typename T>
std::unique_ptr<Column>
directInts(std::vector<T>& ints, uint64_t max, memory::MemoryPool* pool) {
  auto column = std::make_unique<Column>();
  column->values = encodeInts(ints, max, pool);
  column->numValues = ints.size();
  return column;
}

std::unique_ptr<Column> Encoder::toColumn() {
  auto column = std::make_unique<Column>();
  column->kind = kind_;
  if (!abandonDict_ && dictSize() < flatSize()) {
    if (kind_ == TypeKind::VARCHAR) {
      column->encoding = kDict;
      column->values = encodeInts(indices_, strings_.size() - 1, pool_);
      column->bitWidth = bitWidth(strings_.size() - 1);
      column->alphabet = dictStrings_.toColumn();
      return column;
    } else {
      column->alphabet = directInts(dictInts_, max_, pool_);
    }
  }
}

template <typename T>
void Encoder::add(T data) {
  uint64_t item = 0;
  memcpy(&item, &data, std::min(sizeof(item), sizeof(data)));
  if (item > max_) {
    max_ = item;
  }
  direct_.push_back(item);
  if (abandonDict_) {
    return;
  }
  auto it = ints_.find(item);
  if (it != ints_.end()) {
    indices_.push_back(it->second);
    return;
  }
  auto id = ints_.size();
  ints_[item] = id;
  dictInts_.push_back(item);
  indices_.push_back(id);
}

StringView StringSet::add(StringView data) {
  int32_t stringSize = data.size();
  totalSize_ += stringSize;
  if (stringSize > maxLength_) {
    maxLength_ = stringSize;
  }
  if (buffers_.empty() ||
      buffers_.back()->size() + stringSize > buffers_.back()->capacity()) {
    buffers_.push_back(AlignedBuffer::allocate<char>(1 << 20, pool_));
    buffers_.back()->setSize(0);
  }
  lengths_.push_back(stringSize);
  auto& buffer = buffers_.back();
  auto size = buffer->size();
  memcpy(buffer->asMutable<char>() + size, data.data(), data.size());
  buffer->setSize(size + data.size());
  return StringView(buffer->as<char>() + size, data.size());
};

std::unique_ptr<Column> StringSet::toColumn() {
  auto buffer = AlignedBuffer::allocate<char>(totalSize_, pool_);
  int64_t fill = 0;
  for (auto& piece : buffers_) {
    memcpy(buffer->asMutable<char>(), piece->as<char>(), piece->size());
    fill += piece->size();
  }
  auto column = std::make_unique<Column>();
  column->kind = TypeKind::VARCHAR;
  column->encoding = kFlat;
  column->values = buffer;
  column->lengths = directInts(lengths_, maxLength_, pool_);
  column->bitWidth = bitWidth(maxLength_);
  return column;
}

template <>
void Encoder::add(StringView data) {
  auto size = data.size();
  totalStringBytes_ += size;
  if (size > maxLength_) {
    maxLength_ = size;
  }

  if (abandonDict_) {
    allStrings_.add(data);
    return;
  }
  auto it = strings_.find(data);
  if (it != strings_.end()) {
    indices_.push_back(it->second);
    return;
  }
  dictBytes_ += data.size();
  auto copy = dictStrings_.add(data);
  auto id = strings_.size();
  strings_[copy] = id;
  indices_.push_back(id);
}

template <TypeKind kind>
void Encoder::appendTyped(VectorPtr data) {
  using T = typename TypeTraits<kind>::NativeType;

  auto size = data->size();
  SelectivityVector allRows(size);
  DecodedVector decoded(*data, allRows, true);
  for (auto i = 0; i < size; ++i) {
    add<T>(decoded.valueAt<T>(i));
  }
}

void Encoder::append(VectorPtr data) {
  auto kind = data->type()->kind();
  if (kind == TypeKind::UNKNOWN) {
    kind_ = kind;
  }
  count_ += data->size();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(appendTyped, kind, data);
}

void Writer::append(RowVectorPtr data) {
  if (encoders_.empty()) {
    for (auto i = 0; i < data->type()->size(); ++i) {
      encoders_.push_back(std::make_unique<Encoder>(pool_.get()));
    }
  }
  VELOX_CHECK_EQ(encoders_.size(), data->type()->size());
  for (auto i = 0; i < encoders_.size(); ++i) {
    encoders_[i]->append(data->childAt(i));
  }
}

void Writer::finishStripe() {
  std::vector<std::unique_ptr<Column>> columns;
  for (auto& encoder : encoders_) {
    columns.push_back(encoder->toColumn());
  }
  stripes_.push_back(std::make_unique<Stripe>(std::move(columns)));
}

void Writer::finalize(std::string tableName) {
  finishStripe();
  auto table = Table::getTable(tableName, true);
  table->addStripes(std::move(stripes_), pool_);
}

} // namespace facebook::velox::wave
