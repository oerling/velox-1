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

#include "velox/experimental/wave/tests/utils/TestFormatReader.h"

namespace facebook::velox::wave::test {

std::unique_ptr<FormatData> TestFormatParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const velox::common::ScanSpec& scanSpec) {
  auto* column = stripe_->findColumn(*type);
  return std::make_unique<TestFormatData>(column);
}

int32_t TestFormatData::startRead(
    int32_t offset,
    RowSet rows,
    FormatData* /*previousFilter*/,
    SplitStaging& staging,
    DecodePrograms& program,
    WaveStream& stream) {
  int32_t result;
  if (!staged_) {
    staged_ = true;
    result |= FormatData::kStaged;
    Staging staging;
    staging.host column->buffer = column->values->as<char>();
    staging.size = column->values->size();
    splitStaging.add(std::move(staging));
  }
  if (!queud_) {
    result |= FormatData::kQueued || FormatData::kAllQueued;
    auto step = std::make_unique<DecodeStep>();
    step->decodeStep = decodeStep::kTrivial;
    step->data.trivial.input = 0;
    splitStaging.registerPointer(&stream, id, &step->data.trivial.input);
    program.programs.push_back(std::move(step));
  }
  return result;
}

TestStructColumnReader::StructColumnReader(
    const std::shared_ptr<const TypeWithId>& requestedType,
    const std::shared_ptr<const TypeWithId>& fileType,
    TestFormatParams& params,
    common::ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines,
    bool isRoot)
    : StructColumnReader(requestedType, fileType, params, scanSpec) {
  // A reader tree may be constructed while the ScanSpec is being used
  // for another read. This happens when the next stripe is being
  // prepared while the previous one is reading.
  auto& childSpecs = scanSpec.stableChildren();
  for (auto i = 0; i < childSpecs.size(); ++i) {
    auto childSpec = childSpecs[i];
    if (isChildConstant(*childSpec)) {
      childSpec->setSubscript(kConstantChildSpecSubscript);
      continue;
    }
    auto childFileType = fileType_->childByName(childSpec->fieldName());
    auto childRequestedType =
        requestedType_->childByName(childSpec->fieldName());
    auto childParams = DwrfParams(stripe, params.runtimeStatistics(), );

    addChild(TestFormatReader::build(
        childRequestedType, childFileType, params, *childSpec));
    childSpec->setSubscript(children_.size() - 1);
  }
}

std::unique_ptr<ColumnReader> buildIntegerReader(
    std::shared_ptr<TypeWithId>& requestedType,
    std::shared_ptr<TypeWithId>& fileType,
    TestFormatParams& params,
    ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines) {
  return std::make_unique<ColumnReader>(
      requestedType, fileType, params, scanSpec);
}

// static
std::unique_ptr<ColumnReader> TestFormatReader::build(
    const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    TestFormatParams& params,
    common::ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines,    bool isRoot = false) {
  switch (fileType->type()->kind()) {
    case TypeKind::INTEGER:
      return buildIntegerReader(requestedType, fileType, params, scanSpec, path, defines);

    case TypeKind::ROW:
      return std::make_unique<StructColumnReader>(
						  requestedType, fileType, params, scanSpec, path, defines, isRoot);
    default:
      VELOX_UNREACHABLE();
  }
}

} // namespace facebook::velox::wave::test
