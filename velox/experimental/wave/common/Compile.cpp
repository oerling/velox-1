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



#include <nvrtc.h>

#include <gflags/gflags.h>


DEFINE_string(wavegen_include_path, "path to velox/experimental/wave. Mustt contain the cuh headers for runtime compilation");

namespace facebook::velox::wave {


  class CompiledModuleImpl  : public CompiledModule {
    
    ~CompiledModuleImpl() {
      cuModuleUnload(module);
    }

    
    CUmodule module;
    std::vector<CUfunction> kernels;
  };
  
  std::shared_ptr<CompiledModule> CompiledModule::create(const KernelSpec& spec) {

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, // prog
		       code.c_str(),         // buffer
        "saxpy.cu",    // name
        0,             // numHeaders
        NULL,          // headers
        NULL);         // includeNames

const char *opts[] = {"--gpu-architecture=compute_80"};
 nvrtcCompileProgram(prog,     // prog
		    1,        // numOptions
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

nvrtcDestroyProgram(&prog);

CUdevice cuDevice;
CUcontext context;
CUmodule module;
CUfunction kernel;
cuDeviceGet(&cuDevice, 0);
cuCtxCreate(&context, 0, cuDevice);
cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
 std::vector<CUfunction> funcs;
 for (auto& name : spec.entryPoints) {
   funcs.emplace_back();
   cuModuleGetFunction(&funcs.back(), module, entry.c_str());
size_t n = size_t n = NUM_THREADS * NUM_BLOCKS;
size_t bufferSize = n * sizeof(float);
float a = ...;
float *hX = ..., *hY = ..., *hOut = ...;
CUdeviceptr dX, dY, dOut;
cuMemAlloc(&dX, bufferSize);
cuMemAlloc(&dY, bufferSize);
cuMemAlloc(&dOut, bufferSize);
cuMemcpyHtoD(dX, hX, bufferSize);
cuMemcpyHtoD(dY, hY, bufferSize);
void *args[] = { &a, &dX, &dY, &dOut, &n };
cuLaunchKernel(kernel,
            NUM_THREADS, 1, 1,   // grid dim
            NUM_BLOCKS, 1, 1,    // block dim
            0, NULL,             // shared mem and stream
            args,                // arguments
            0);

 return std::make_shared<CompiledModuleImpl>(module, 
}
