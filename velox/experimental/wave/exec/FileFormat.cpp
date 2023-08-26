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

template <typename T>
void Encoder::add(T data) {
  uint64_t item = 0;
  memcpy(&item, &data, std::min(sizeof(item), sizeof(data)));
  if (abandonDict_) {
    direct_.push_back(item);
    return;
  }
  auto it = ints_.find(item);
  if (it != ints_.end()) {
    indices_.push_back(it->second);
    return;
  }
  auto id = ints_.size();
  ints_[item] = id;
  indices_.push_back(id);
}

StringView StringSet::add(StringView data) {
  if (buffers_.empty() ||
      buffers_.back()->size() + data.size() > buffers_.back()->capacity()) {
    buffers_.push_back(AlignedBuffer::allocate<char>(1 << 20, pool_));
    buffers_.back()->setSize(0);
  }
  lengths_.push_back(data.size());
  auto& buffer = buffers_.back();
  auto size = buffer->size();
  memcpy(buffer->asMutable<char>() + size, data.data(), data.size());
  buffer->setSize(size + data.size());
  return StringView(buffer->as<char>() + size, data.size());
};

template <>
void Encoder::add(StringView data) {
  if (abandonDict_) {
    allStrings_.add(data);
    return;
  }
  auto it = strings_.find(data);
  if (it != strings_.end()) {
    indices_.push_back(it->second);
    return;
  }
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
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
      appendTyped, data->type()->kind(), data);
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
    columns.push_back(encoder->column());
  }
  stripes_.push_back(std::make_unique<Stripe>(std::move(columns)));
}

void Writer::finalize(std::string tableName) {
  finishStripe();
  auto table = Table::getTable(tableName, true);
  table->addStripes(std::move(stripes_), pool_);
}

} // namespace facebook::velox::wave
