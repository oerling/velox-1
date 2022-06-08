//
// Created by Ying Su on 4/2/22.
//

#pragma once

#include "dwio/dwrf/common/BufferedInput.h"

namespace facebook::velox::parquet {

static int32_t decodeNulls(
    int64_t offset,
    int32_t batchSize,
    uint32_t maxDefine,
    BufferPtr defineLevelsBuffer,
    BufferPtr nullsOutBuffer) {
  const uint8_t* defineLevels =
      defineLevelsBuffer->template as<const uint8_t>();
  auto nullsBuf = nullsOutBuffer->template asMutable<uint8_t>();

  // TODO: Work on a fast path
  int32_t nullCount = 0;
  for (auto i = 0; i < batchSize; i++) {
    uint8_t isNull = (defineLevels[i + offset] != maxDefine);
    bits::setBit(nullsBuf, offset + i, isNull);
    nullCount += isNull;
  }

  return nullCount;
}

} // namespace facebook::velox::parquet
