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

#include <gtest/gtest.h>
#include "velox/common/time/Timer.h"
#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/common/tests/BlockTest.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include <cuda.h>
#include "velox/experimental/wave/common/Exception.h"

#include <iostream>


namespace facebook::velox::wave {

  void testCuCheck(CUresult result) {
    if (result != CUDA_SUCCESS) {
      const char* str;
      cuGetErrorString(result, &str);
      waveError(
		std::string("Cuda error: ") + str);
    }
  }

  
class CompileTest : public testing::Test {
 protected:
  void SetUp() override {
    return;
    device_ = getDevice();
    setDevice(device_);
    allocator_ = getAllocator(device_);
    arena_ = std::make_unique<GpuArena>(1 << 28, allocator_);
    streams_.push_back(std::make_unique<BlockTestStream>());
  }


  Device* device_;
  GpuAllocator* allocator_;
  std::unique_ptr<GpuArena> arena_;
  std::vector<std::unique_ptr<BlockTestStream>> streams_;

};

  struct KernelParams {
    int32_t* array;
    int32_t size;
  };
   
const char* kernelText = 
"#include <cstdint>\n"
    "namespace facebook::velox::wave {\n"
    "  struct KernelParams {\n"
"    int32_t* array;\n"
"    int32_t size;\n"
"  };\n"
"\n"
"  void __global__ add1(KernelParams params) {\n"
"    for (auto i = threadIdx.x; i < params.size; i += blockDim.x) {\n"
"      ++params.array[i];\n"
"    }\n"
"  }\n"
"\n"
"  void __global__ add2(KernelParams params) {\n"
"    for (auto i = threadIdx.x; i < params.size; i += blockDim.x) {\n"
"      params.array[i] += 2;\n"
"    }\n"
  "  }\n"
  "} // namespace\n";
  


  TEST_F(CompileTest, module) {
    KernelSpec spec = KernelSpec{kernelText, {"add1", "add2"}};
    auto module = CompiledModule::create(spec);
    int32_t* ptr;
    testCuCheck(cuMemAllocManaged(reinterpret_cast<CUdeviceptr*>(&ptr), 1000 * sizeof(int32_t), CU_MEM_ATTACH_GLOBAL));
    KernelParams record{ptr, 1000};
    memset(ptr, 0, 1000 * sizeof(int32_t));
    void* recordPtr = &record;
    module->launch(0, 1, 256, 0, nullptr, &recordPtr);
    EXPECT_EQ(1, ptr[0]);
  }

#if 0
    TEST_F(CompileTest, basic) {
      kernelKey key{"pfaal", []() -> KernelSpec {
	return KernelSpec{kernelText,
			  {"add1", "add2"}};
      }};
      

    }
#endif
}

  
