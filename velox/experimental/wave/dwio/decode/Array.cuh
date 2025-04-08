





template <
	     int32_t kBlockSize,
    DecodeStep kEncoding>
__device__ void decodeLengths(GpuDecode* op) {
  int32_t nthLoop = 0;
  constexpr bool kAlwaysDict = !std::is_same_v<T, IndexT>;
  switch (op->nullMode) {
    case NullMode::kDenseNonNull: {

      if (threadIdx.x == 0) {
	op->temp[8] = 0;
      }
        auto base = op->baseRow;
        auto i = threadIdx.x;
        auto& d = op->data.dictionaryOnBitpack;
        auto end = op->maxRow - op->baseRow;
        auto bitWidth = d.bitWidth;
        auto address = reinterpret_cast<uint64_t>(d.indices);
        auto baseline = d.baseline;

          int32_t alignOffset = (address & 3) * 8;
          address &= ~3UL;
          auto words = reinterpret_cast<uint32_t*>(address);
          uint32_t mask = (1L << bitWidth) - 1;
          auto* result = reinterpret_cast<int32_t*>(op->result);
          auto* alphabet =
              reinterpret_cast<const int32_t*>(op->data.dictionaryOnBitpack.alphabet);
          for (; i < end; i += blockDim.x) {
	    int32_t data = 0;
	    if (i < end) {
	      int32_t bitIndex = (i + base) * bitWidth + alignOffset;
	      int32_t wordIndex = bitIndex >> 5;
	      int32_t bit = bitIndex & 31;
	      uint32_t word = words[wordIndex];
	      uint32_t index = __funnelshift_r(
					       word, (bitWidth + bit > 32 ? words[wordIndex + 1] : 0), bit);
	      index &= mask;
	      data = 
                alphabet ? alphabet[index + baseline] : index + baseline;
	      if (threadIdx.x == 0) {
		data += temp[8];
	      }
	    }
	    auto sum = inclusiveSum<int32_t>(data, &op->temp[9], op->temp);
	    if (i < end) {
	      result[i] = sum;
	    }
	  }

    }
      break;

    case NullMode::kDenseNullable: {
      int32_t maxRow = op->maxRow;
      int32_t dataIdx = 0;
      auto* state = reinterpret_cast<NonNullState*>(op->temp);
      if (threadIdx.x == 0) {
        state->nonNullsBelow = op->nthBlock == 0
            ? 0
            : op->nonNullBases
                  [op->nthBlock *
                       (op->gridNumRowsPerThread / (1024 / kBlockSize)) -
                   1];
        state->nonNullsBelowRow =
            op->gridNumRowsPerThread * op->nthBlock * kBlockSize;
      }
      __syncthreads();
      do {
        int32_t base = op->baseRow + nthLoop * kBlockSize;
        int32_t dataIdx;
        int32_t data = 0;
        if (base < maxRow) {
          dataIdx = nonNullIndex256(
              op->nulls, base, min(kBlockSize, maxRow - base), state);
          bool inRange = base + threadIdx.x < maxRow;
          if (inRange) {
            if (dataIdx == -1) {
	      data = 0;
            } else {
              data = randomAccessDecode<IndexT, kEncoding>(op, dataIdx);
            }
          }
        }
	      if (threadIdx.x == 0) {
		data += temp[8];
	      }
	      auto sum = inclusiveSum<int32_t>(data, &op->temp[8], op->temp);
	if (inRange) {
	  result[base + threadIdx.x] = sum;
	}
      } while (++nthLoop < op->numRowsPerThread);
      break;
    }
  }
  __syncthreads();
}

/// Finds row' in 'rows'. Returns the index of the first row <= 'row'.
inline __device__ int
lower(const int32_t* rows, int32_t size, int32_t row, GpuDecode* op) {
  int lo = 0, hi = size;
  while (lo < hi) {
    int i = (lo + hi) / 2;
    if (rows[i] == row) {
      return i;
    }
    if (rows[i] < row) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  printf("Expecting to find  row %d in findRow() size %d %p\n", row, size, op);
  assert(false);
}


int32_t arraySourceIdx(int32_t* resultEnds int32_t* rows, int32_t* sourceEnds) {
  auto endIdx = upper(resultEnds, numArrays, threadIdx.x);
  if (resultEnds[endIdx] >= threadIdx.x) {
    return sourceIx[0] + threadIdx.x;
  }
  return sourceIdx[endIdx-1] + (threadIdx.x - resultIdx[endIdx-1]);
}

void makeInnerRows(const int32_t* lengths, const int32_t* rows, int32_t numRows, int32_t* resultBases, int32_t* sourceRows) {
x
  
}


