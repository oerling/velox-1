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

#pragma once

#include "velox/dwio/parquet/reader/StructColumnReader.h"

namespace facebook::velox::parquet {

class StructColumnReader : public dwrf::SelectiveStructColumnReader {
 public:
  StructColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      ParquetParams& params,
      common::ScanSpec* scanSpec)
      : SelectiveStructColumnReader(
            dataType,
            params,
            scanSpec,
            dataType->type) {
    auto& childSpecs = scanSpec->children();
    for (auto i = 0; i < childSpecs.size(); ++i) {
      if (childSpecs[i]->isConstant()) {
        continue;
      }
      auto childDataType = nodeType_->childByName(childSpecs[i]->fieldName());

      children_.push_back(ParquetColumnReader::build(
          childDataType, params, childSpecs[i].get()));
      childSpecs[i]->setSubscript(children_.size() - 1);
    }
  }

  std::vector<uint32_t> filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsWriterInfo& context) const override {
    if (!scanSpec_->filter()) {
      return {};
    }
    return {};
  }

  void seekToRowGroup(uint32_t index) override;

 private:
  bool filterMatches(const RowGroup& rowGroup);
};

void StructColumnReader::seekToRowGroup(uint32_t index) {
  for (auto& child : children_) {
    child->seekToRowGroup(index);
  }
}

bool StructColumnReader::filterMatches(const RowGroup& rowGroup) {
  return true;
#if 0
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
#endif
}

} // namespace facebook::velox::parquet
