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

#include "velox/type/Type.h"

/// Sample set of composable encodings. Bit packing, direct and dictionary.
namespace facebook::velox::wave::test {

  enum Encoding {
    kFlat, kBits, kDict
  };

  struct Column {
    ~Column() {
      if (data( {
	    ::free(data);
	    data = nullptr;
	  }
    }
    
    Encoding encoding;
    // Number of encoded values.
    int32_t numValues;
    // Distinct values in kDict.
    std::shared_ptr<Column> alphabet;
    // Encoded lengths for strings in 'data' if type is varchar and encoding is kFlat.
    std::unique_ptr<Column> lengths;
    // Width of bit packed encoding.
    int8_t bitWidth;
    // Byte size of encoded data in 'data'.
    int32_t numBytes;

    // mallocd block with raw column data. Start aligned at 8 bytes. This is like a cache entry of a an encoded column in a file.
    BufferPtr  data
  };

  struct Stripe {
    std::vector<std::unique_ptr<Column>> column;
  };

  class EncoderBase {
  public:
    // Adds data.
    void append(VectorPtr data);

    // Retrieves the data added so far as an encoded column.
    std::unique_ptr<Column> getColumn();

  private:
    // Distincts for either int64_t or double.
    folly::F14FastMap<int64_t, int32_t> ints_;
    // Distincts for strings.
    folly::F14FastMap<StringView, int32_t strings_;
    // Values as indices into dicts.
    std::vector<int32_t> dictEncoded_;
    // True if too many distinct values for dict.
    bool abandomDict_{false};
    // longest string, if string type.
    int32_t maxLength{0};
    
  };
  
  
class Table  {
 public:
 Table(RowTypePtr type, int32_t stripeSize, memory::MemoryPool* pool)
   : type_(std::move(type, int32_t stripeSize)), pool_(pool) {}

  /// Appends a batch of data.
  void append(RowVectorPtr data) {
  }
  // Finishes encoding data, makes the table ready to read.
  void finalize();
  int64_t numRows() const {
    return numRows_;
  }

  int32_t numStripes() const {
    return stripes_.size();
  }
  Stripe* stripeAt(int32_t index) {
    return stripes_[index].get();
  }
  
 private:
  RowTypePtr type_;
  const int32_t stripeSize_;
  std::vector<std::unique_ptr< Stripe>> stripes_; 
  memory::MemoryPool* pool_;
  std::vector<std::unique_ptr<Encoder>> encoders_;
};

struct WaveTestConnectorSplit : public connector::ConnectorSplit {
  Stripe* stripe;
};

}


