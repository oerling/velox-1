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

#include <parquet/api/writer.h>

namespace facebook::velox::parquet {
  class Writer {
    public:
    Writer(const std::string& path, int32_t stripesize);

    void append(const RowVectorPtr& data);
    
  private:
    std::unique_ptr<::parquet::parquetFilewriter> fileWriter_; 
    
  };
  

}

