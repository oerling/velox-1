//
// Created by Ying Su on 2/24/22.
//
#pragma once


#include <thrift/transport/TVirtualTransport.h>
//#include "velox/dwio/common/MetricsLog.h"

namespace facebook::velox::parquet {

class ThriftBufferedTransport
    : public apache::thrift::transport::TVirtualTransport<
          ThriftBufferedTransport> {
 public:
  ThriftBufferedTransport(const void* inputBuf, uint64_t len)
      : inputBuf_(reinterpret_cast<const uint8_t*>(inputBuf)),
        size_(len),
        offset_(0) {}

  uint32_t read(uint8_t* outputBuf, uint32_t len) {
    DWIO_ENSURE(offset_ + len <= size_);
    memcpy(outputBuf, inputBuf_ + offset_, len);
    offset_ += len;
    return len;
  }

 private:
  const uint8_t* inputBuf_;
  const uint64_t size_;
  uint64_t offset_;
};

} // namespace facebook::velox::parquet
