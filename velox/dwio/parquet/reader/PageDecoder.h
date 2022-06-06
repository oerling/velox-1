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

namespace facebook::dwio::parquet {

class PageDecoder {
  PageDecoder(std::unique_ptr<dwio::dwrf::SeekableInputStream> stream)
    : inputStream_(std::move(stream)) {}

  template <typename Visitor> readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    VELOX_CHECK(!nulls, "Parquet does not accept incoming nulls");
    
  }

private:

  std::unique_ptr<SeekableInputStream> inputStream_;
};
  const char* bufferStart_{nullptr};
  const char* bufferEnd_{nullptr};
  std::unique_ptr<dwrf::DirectDecoder> directDecoder_;
  
  
}
