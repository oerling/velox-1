

#include "velox/dwio/dwrf/reader/SelectiveByteRleColumnReader.h"

namespace facebook::velox::dwrf {



uint64_t SelectiveByteRleColumnReader::skip(uint64_t numValues) {
  numValues = ColumnReader::skip(numValues);
  if (byteRle_) {
    byteRle_->skip(numValues);
  } else {
    boolRle_->skip(numValues);
  }
  return numValues;
}


void SelectiveByteRleColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  prepareRead<int8_t>(offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;
  common::Filter* filter =
      scanSpec_->filter() ? scanSpec_->filter() : &Filters::alwaysTrue;
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        processValueHook<false>(rows, scanSpec_->valueHook());
      }
      return;
    }
    if (isDense) {
      processFilter<true>(filter, ExtractToReader(this), rows);
    } else {
      processFilter<false>(filter, ExtractToReader(this), rows);
    }
  } else {
    if (isDense) {
      processFilter<true>(filter, DropValues(), rows);
    } else {
      processFilter<false>(filter, DropValues(), rows);
    }
  }
}
  


}
