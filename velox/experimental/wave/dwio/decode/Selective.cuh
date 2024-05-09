

#include "velox/experimental/wave/common/Bits.cuh"


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
