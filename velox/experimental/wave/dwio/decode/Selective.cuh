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

#include "velox/experimental/wave/common/Bits.cuh"
#include "velox/experimental/dwio/decode/DecodeStep.h"


namespace facebook::velox::wave {




template <typename T>
void DecodeRandomAccessLane (GpuDecode* op, int32_t base, int32_t numRows, int32_t& nonNullOffset, int32_t& filterOffset) {
  int32_t dataIdx;
  bool filterPass = true;
  bool filterDecided = false;
  switch(op->nullMode) {
  case NullMode::kDenseNonNull:
      dataIdx = threadIdx.x + base;
      break;
  case NullMode::kSparseNonNull:
      dataIdx = op->rows[threadIdx.x + base];
      break;
  case NullMode::kDenseNullable:
      dataIdx = nonNullIndex(op->nulls + (base / 8), numRows, nonNullOffset, temp);
      break;
  case NullMode::kSparseNullable:
      dataIdx = nonNullIndex256Sparse(op->nulls, op->rows, base, base + numRows, nonNullOffset, temp);
      break;
    }
  T value();
    switch (op->decodeType) {
    case kBit:
    case kBitDict:
    case kDirect:
    case kDirectDict:
      
    }
    if (op->haveException) {
      
    }
    if (!filterDecided) {
      switch (op->filterKind) {
      case kNone:
	op->result[base + threadIdx.x] = value;
	if (op->resultNulls) {
	  op->resultNulls = dataIdx == -1 ? kNull : kNotNull;
	}
	return;
      case kNotNull:
      filterPass = dataIdx != -1;
      break;
    case kIsNull:
      filterPass = dataIdx == -1;
    case kRange:
      if (dataIdx == -1) {
	filterPass = op->nullsAllowed;
      } else {
	filterPass = value >= op->filterLow<T>() && value <= op->filterHigh<T>();
      }
      break;
      }
    }
    // Write passing rows compacted.
    resultIndex = exclusiveSum<int16_t, 256>(filterPass, temp) + filterBase;
    if (threadIdx.x == blockDim.x - 1) {
      filterFill = resultIdx + filterPassed;
    }
    if (filterPass) {
      op->filterResult[resultIdx] = base + threadIdx.x;
      if (result) {
	result[resultIndex] = value;
	if (resultNulls) {
	  resultNulls[resultIndex] = dataIdx == -1 ? kNull : kNotNull;
	}
      }
    }
}

      using Scan16 = cub::WarpScan<uint16_t>;
    
})



  

voide decode(DecodeOp* op) {

}

}
