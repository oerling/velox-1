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

#include "velox/experimental/common/BlockCost.h"
#include "velox/experimental/common/ResultStaging.h"
#include "velox/experimental/dwio/decode/DecodeStep.h"

namespace facebook::velox::wave {

  class SplitStaging;


  class DecodeSet : public BlockProxySet {
 public:
    DecodeSet(ResultStaging* deviceStaging, ResultStaging* resultStaging, SplitStaging* splitStaging)
      : deviceStaging_(deviceStaging), resultStaging_(resultStaging), splitStaging_(splitStaging) {}
        
    SplitStaging* splitStaging() const {
      return splitStaging_;
    }
    
    ResultStaging* resultStaging() const {
      return resultStaging_;
    };

        
    ResultStaging* deviceStaging() const {
      return deviceStaging_;
    };


    int32_t addAction(
		      Depends* depends,
		      std::unique_ptr<Decodestep> step);

    std::vector<std::unique_ptr<DecodeStep> allSteps_;
    std::vector<std::vector<DecodeStep*>> stepByOp_;

    SplitStaging* splitStaging_;
    SplitStaging* Staging_;
    SplitStaging* resultStaging_;
  
  };
  

}
