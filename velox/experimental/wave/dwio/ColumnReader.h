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

#include "velox/dwio/common/TypeWithId.h"
#include "velox/experimental/wave/dwio/FormatData.h"

namespace facebook::velox::wave {

/// dwio::SelectiveColumnReader for Wave
class ColumnReader {
 public:
  ColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : memoryPool_(params.pool()),
        requestedType_(requestedType),
        fileType_(fileType),
        formatData_(params.toFormatData(fileType, scanSpec)),
        scanSpec_(&scanSpec) {}

 protected:
  std::unique_ptr<FormatData> formatData_;
};

} // namespace facebook::velox::wave
