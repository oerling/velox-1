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

#include "velox/experimental/wave/exec/WaveDriver.h"

namespace facebook::velox::wave {

class Operator {
 public:
  exec::BlockingReason isBlocked
  virtual bool canAdvance();
  virtual void advance();
  virtual void fullyConsumed();

  // Returns number of bytes tied up in completed or ongoing speculative execution.
  virtual int32_t speculativeSize() const {
    return 0;
  }

  virtual int32_t dropSpeculative() {
    VELOX_UNSUPPOTED();
  }
  std::vector<InputGroup> needsInput(); 
  
 protected:
  std::vector<exec::IdentityProjection> identityProjections_;
  TypePtr outputType_;
  std::unique_ptr<Vector> output_;
  std::unique_ptr<Stream> streamNoShared_;
  std::unique_ptr<Stream> streamWithShared_;
};
  
}




