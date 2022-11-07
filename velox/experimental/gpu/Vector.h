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
#include "velox/experimental/gpu/Util.h"


namespace facebook::velox::cuda {

  struct Vector {
    void* FOLLY_NONNULL values;
    int64_t* offsets{nullptr};
    int64_t sizes{nullptr};
    uint64_t* FOLLY_NULLABLE nulls{nullptr};
  };
    
}



