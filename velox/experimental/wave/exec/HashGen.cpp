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

#include "velox/experimental/wave/exec/HashGen.h"

namespace  facebook::velox::wave {

  void makeKeyMembers(const std::vector<const AbstractOperand*>& keys, std::stringstream& out) {
    for (auto i = 0; i < keys.size(); ++i) {
      auto* key = keys[i];
      out << cudaTypeName(*key->type) << " key" << i << ";\n";
      
    }
  }


  void makeOperandHashFunction(CompileState* state, std::vector<const AbstractOperand*> keys, bool discardNulls, std::stringstream& out) {
    out << "uint64_t __device__ __forceinline__ hash(Operands** operands, ErrorCode& laneStatus) {\n";
    
		       
  }

  void makeRowHashFunction(std::vector<const AbstractOperand*> keys, bool discardNulls, std::stringstream& out) {
    
  }

  
  

}
