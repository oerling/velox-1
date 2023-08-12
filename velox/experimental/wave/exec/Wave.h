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

#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/vector/Vector.h"

namespace facebook::velox::wave {

  struct Channel {
    int32_t id;
    WaveVectorPtr vector;
  };

  struc InputGroup {
    // Input may be received at different times for different
    // channels. Indicates which inputs should be received first,
    // e.g. given a choice, a group by wants the grouping keys before
    // the aggregate arguments because computing the groups can start
    // before the aggregate arguments arrive. A lower value indicates
    // earlier preferred receipt. For a projection all values will be
    // 0 if all the expressions are independent. Channels that common
    // subexpressions depend on might be preferred before other
    // channels. All the channels in the group must be received though before the computation can start.
    int32_t precedence;
    std::vector<ChannelIndex> channels;
  };

  
  enm class DropletStatus {
    kDone;
    kMore,
    kError
  };
  
  /// The result of a droplet execution.
  struct Result {
    DropletStatus Status;
    int32_t numRows;
  };
  
  /// Common fields in Wave kernel launch.
  struct Droplet {
    int16_t op;
    Result* result;
  };

  class DropletStream : public Stream {
    virtual start(Droplet** droplets, 
  };
  
  };
  
  struct Wave {
    int64_t sequence;
    Span<Droplet*> dropets;
    Span<Channel> Channels;
    Span<Channel> identityChannels;
  };
  


struct Dependent {
  int64_t sequence;
  int32_t numMissing;
  span<Channel*> channels;
  std::function<void(Dependent*)> activate;
}

