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

#include "velox/experimental/gpu/UopExec.h"

namespace facebook::velox::cuda {

  using UopRun = void(*)(Uop* FOLLY_NONNULL uop);

  void Uop::inputReady() {
    int32_t newState = --state_;
    if (newState == 0) {
      task_->runnable->add(this);
      for (auto i = 0; i < optionalInputs.size(); ++i) {
	optionalInputs_[i]->optionalNeeded();
      }
    }
  }

  void outputReady() {
    for (auto i = 0; i < comsumers_.size(); ++i) {
      consumers_[i]->inputReady();
    }
  }

  // Called by 'this' after noMoreInput() has been called on 'this' and all
  // output generated.
  void noMoreOutput();

  // Called by 'this' to signal that th producers are free to rewrite the input
  // 'this' consumed.

  void inputConsumed();

  // Called by producers to signal that they will not produce any more.
  void noMoreInput();

  
  Uop::batchDone() {
    for(auto next : dependdents) {
      
    }
  }
  
  void Task::runThread() {
    while (numFinished_ < uops_.size()) {
      //__shared__ Uop* sharedUop;
      Uop* uop;
      //if (threadIdx.x == 0) {
	runnable_.dequeue(uop);
	//}
      // --syncThreads();
      uopRun[uop->uopCode](uop);
    }
  }

  void Task::inputReady(UopSpan leaves, Span<InputBuffer> input) {
    for (auto i = 0; i < leaves.size(); ++i) {

      leaves[i]->setInput(inputs[i]);
      leaves[i]->inputReady();
    }
  }


}
