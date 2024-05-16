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

#include "velox/experimental/wave/dwio/decode/DecodeStep.h"
#include "velox/experimental/wave/dwio/ColumnReader.h"

namespace facebook::velox::wave {

void GpuDecode::setFilter(ColumnReader* reader, Stream* stream) {
  auto* veloxFilter = reader->scanSpec().filter();
  if (!veloxFilter) {
    filterKind = WaveFilterKind::kAlwaysTrue;
    return;
  }
  switch (veloxFilter->kind()) {
    case common::FilterKind::kBigintRange: {
      filterKind = WaveFilterKind::kBigintRange;
      nullsAllowed = veloxFilter->testNull();
      filter._.int64Range[0] =
          reinterpret_cast<common::BigintRange*>(veloxFilter)->lower();
      filter._.int64Range[1] =
          reinterpret_cast<common::BigintRange*>(veloxFilter)->upper();
      break;
    }

    default:
      VELOX_UNSUPPORTED("Unsupported filter kind", static_cast<int32_t>(veloxFilter->kind()));
  }
}

} // namespace facebook::velox::wave
