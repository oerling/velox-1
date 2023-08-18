

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

#include "velox/experimental/wave/WaveOperator.h"

namespace facebook::velox::wave {

  WaveDriver::WaveDriver(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::vector<std::unique_ptr<Operator> waveOperators,
      std::vector<exec::Operator*> cpuOperators,
	     SubfieldMap subfields)
    : exec::Operator(operatorId, driverCtx,),
      operators_(std::move(waveOperators)),
      cpuOperators_(std::move(cpuOperators)),
      subfields_(std::move(subfields)) {}
												 
  RowVectorPtr WaveDriver::getOutput() override {
  if (!runnable_) {
    return nullptr;
  }

  return nullptr;
}
  std::string WaveDriver::toString() const override {
    std::stringstream out;
    out << "{Wave" << std::endl;
    for (auto& op : operators_) {
      out << op->toString() << std::endl;
    }
    return out.str();
  }


  
} // namespace facebook::velox::wave
