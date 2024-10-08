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



#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/Exception.h"
#include <nvrtc.h>

#include <gflags/gflags.h>


DEFINE_string(wavegen_include_path, "", "path to velox/experimental/wave. Mustt contain the cuh headers for runtime compilation");

namespace facebook::velox::wave {

  void nvrtcCheck(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {                                
      waveError(nvrtcGetErrorString(result));
    }
  }

  class CompiledModuleImpl  : public CompiledModule {
    
    ~CompiledModuleImpl() {
      cuModuleUnload(module);
    }

    
    CUmodule module;
    std::vector<CUfunction> kernels;
  };
  
  std::shared_ptr<CompiledModule> CompiledModule::create(const KernelSpec& spec) {

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,
		       spec.code.c_str(),         // buffer
        "rtctest.cu",    // name
        0,             // numHeaders
        NULL,          // headers
		       NULL);         // includeNames
    for (auto& name : spec.entryPoints) {
      nvrtcCheck(nvrtcAddNameExpression(entry.c_str()));
    }
    const char *opts[] = {"--gpu-architecture=compute_80", "-g", "-G"};
    nvrtcCompileProgram(prog,     // prog
			3,        // numOptions
			opts);    // options
    

    size_t logSize;

    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    // Obtain PTX from the program.
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);
    std::vector<std::string> loweredNames;
    for (auto& entry : spec.entryPoints) {
      const char * temp;
      nvrtcCheck(nvrtcGetLoweredName(prog, entry.c_str(), &temp));
      loweredNames.push_back(std::string(temp));
    }
 
    nvrtcDestroyProgram(&prog);

    CUdevice cuDevice;
    CUcontext context;
    getDeviceAndContext(dvice, context);
    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    std::vector<CUfunction> funcs;
    for (auto& name : loweredNames) {
      funcs.emplace_back();
      cuModuleGetFunction(&funcs.back(), module, name.c_str());
    }
    return std::make_shared<CompiledModuleImpl>(module, std::move(funcs));
}


  CompiledModuleImpl::launch(int32_t kernelIdx, int32_t numBlocks, int32_t numThreads, int32_t shared, void* stream, void** args) {
			      
   cuLaunchKernel(kernels[idx],
            numBlocks, 1, 1,   // grid dim
            numThreads, 1, 1,    // block dim
		  shared, stream,             // shared mem and stream
            args,                // arguments
            0);
  };
