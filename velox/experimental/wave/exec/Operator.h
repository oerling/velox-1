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
  virtual exec::BlockingReason isBlocked(exec::ContinueFuture& future);

  /// True if may reduce cardinality without duplicating input rows.
  bool isFilter() {
    return isFilter_;
  }

  /// True if a single input can produce zero to multiple outputs.
  bool isExpanding() const {
    return isExpanding_;
  }

  virtual bool canAdvance();
  virtual void advance();
  virtual void fullyConsumed();

  // Returns number of bytes tied up in completed or ongoing speculative
  // execution.
  virtual int32_t speculativeSize() const {
    return 0;
  }

  virtual int32_t dropSpeculative() {
    VELOX_UNSUPPOTED();
  }

 protected:
  bool isFilter_{false};

  bool isExpanding_{false};

  std::vector<exec::IdentityProjection> identityProjections_;

  TypePtr outputType_;
  std::unique_ptr<Vector> output_;
};

} // namespace facebook::velox::wave
