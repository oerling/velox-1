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

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <iostream>
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/CudaUtil.cuh"
#include "velox/experimental/wave/common/Exception.h"

#include <sstream>

namespace facebook::velox::wave {

  void cuCheck(CUresult result, const char* file, int32_t line) {
    if (result != CUDA_SUCCESS) {
      const char* str;
      cuGetErrorString(result, &str);
      waveError(
      fmt::format("Cuda error: {}:{} {}", file, line, str));
    }
  }
  
void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  waveError(
      fmt::format("Cuda error: {}:{} {}", file, line, cudaGetErrorString(err)));
}

void cudaCheckFatal(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  auto error =
      fmt::format("Cuda error: {}:{} {}", file, line, cudaGetErrorString(err));
  std::cerr << err << std::endl;
  exit(1);
}

  namespace {
  std::mutex ctxMutex;
bool  driverInited = false;

  // A context for each device. Each is initialized on first use and made the primary context for the device.
std::vector<CUcontext> deviceContexts;
  // Device structs to 1:1 to contexts.
  std::vector<std::unique_ptr<Device>> devices;
  }
  
  void setDriverDevice(int32_t deviceId) {
    if (!driverInited) {
      std::lock_guard<std::mutex> l(ctxMutex);
      CU_CHECK(cuInit(0));
      int32_t cnt;
      CU_CHECK(cuDeviceGetCount(&cnt));
      contexts.resize(cnt);
      devices.resize(cnt);
      if (cnt == 0) {
	waveError("No Cuda devices found");
      }
    }
    if (deviceId >= contexts.size()) {
      waveError(std::string("Bad device id ") + deviceId);
    }
    if (contexts[deviceId] != nullptr) {
      cuCtxSetCurrent(contexts[deviceId]);
      return;
    }
    {
      std::lock_guard<std::mutex> l(ctxMutex);
      CUdevice dev;
      CU_CHECK(cuDeviceGet(&dev, deviceId));	   
      CU_CHECK(      cuDevicePrimaryCtxRetain(&contexts[deviceId], dev));
    }
    CU_CHECK(cuCtxSetCurrent(contexts[deviceId]));
  }


  Device* currentDevice() {
    CUdevice dev;
    CU_CHECK(cuCtxGetCurrent(&dev));
    if (!dev) {
      return nullptr;
    }
    for (auto i = 0; i < contexts 
  }

Device* getDevice(int32_t deviceId/) {
  Device* save = nullptr;
  if (driverInited) {
    save = currentDevice();
  }
    
  auto* dev = driverSetDevice(deviceId);
}

void setDevice(Device* device) {
  driverSetDevice(device->id);
  CUDA_CHECK(cudaSetDevice(device->deviceId));
}

    
namespace {
class CudaManagedAllocator : public GpuAllocator {
 public:
  void* allocate(size_t size) override {
    void* ret;
    CUDA_CHECK(cudaMallocManaged(&ret, size));
    return ret;
  }

  void free(void* ptr, size_t /*size*/) override {
    cudaFree(ptr);
  }
};

class CudaDeviceAllocator : public GpuAllocator {
 public:
  void* allocate(size_t size) override {
    void* ret;
    CUDA_CHECK(cudaMalloc(&ret, size));
    return ret;
  }

  void free(void* ptr, size_t /*size*/) override {
    cudaFree(ptr);
  }
  bool isDevice() const override {
    return true;
  }
};

class CudaHostAllocator : public GpuAllocator {
 public:
  void* allocate(size_t size) override {
    void* ret;
    CUDA_CHECK(cudaMallocHost(&ret, size));
    return ret;
  }

  void free(void* ptr, size_t /*size*/) override {
    cudaFreeHost(ptr);
  };

  bool isHost() const override {
    return true;
  }
};

} // namespace

GpuAllocator* getAllocator(Device* /*device*/) {
  static auto* allocator = new CudaManagedAllocator();
  return allocator;
}

GpuAllocator* getDeviceAllocator(Device* /*device*/) {
  static auto* allocator = new CudaDeviceAllocator();
  return allocator;
}
GpuAllocator* getHostAllocator(Device* /*device*/) {
  static auto* allocator = new CudaHostAllocator();
  return allocator;
}



  Stream::Stream(std::unique_ptr<StreamImpl> impl)
    : stream_(std::move(impl)) {}

  
Stream::Stream() {
  stream_ = std::make_unique<StreamImpl>();
  CUDA_CHECK(cudaStreamCreate(&stream_->stream));
}

Stream::~Stream() {
  if (stream_->stream) {
    cudaStreamDestroy(stream_->stream);
  }
  if (stream_->cuStream) {
    cuStreamDestroy((CUstream)stream_->cuStream);
  }
}

void Stream::wait() {
  CUDA_CHECK(cudaStreamSynchronize(stream_->stream));
}

void Stream::prefetch(Device* device, void* ptr, size_t size) {
  CUDA_CHECK(cudaMemPrefetchAsync(
      ptr, size, device ? device->deviceId : cudaCpuDeviceId, stream_->stream));
}

void Stream::memset(void* ptr, int32_t value, size_t size) {
  CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream_->stream));
}

void Stream::hostToDeviceAsync(
    void* deviceAddress,
    const void* hostAddress,
    size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
      deviceAddress,
      hostAddress,
      size,
      cudaMemcpyHostToDevice,
      stream_->stream));
  isTransfer_ = true;
}

void Stream::deviceToHostAsync(
    void* hostAddress,
    const void* deviceAddress,
    size_t size) {
  CUDA_CHECK(cudaMemcpyAsync(
      hostAddress,
      deviceAddress,
      size,
      cudaMemcpyDeviceToHost,
      stream_->stream));
}

namespace {
struct CallbackData {
  CallbackData(std::function<void()> callback)
      : callback(std::move(callback)){};
  std::function<void()> callback;
};

void readyCallback(void* voidData) {
  std::unique_ptr<CallbackData> data(reinterpret_cast<CallbackData*>(voidData));
  data->callback();
}
} // namespace

void Stream::addCallback(std::function<void()> callback) {
  auto cdata = new CallbackData(std::move(callback));
  CUDA_CHECK(cudaLaunchHostFunc(stream_->stream, readyCallback, cdata));
}

struct EventImpl {
  ~EventImpl() {
    auto err = cudaEventDestroy(event);
    if (err != cudaSuccess) {
      // Do not throw because it can shadow other more important exceptions.  As
      // a rule of thumb, we should not throw in any destructors.
      LOG(ERROR) << "cudaEventDestroy: " << cudaGetErrorString(err);
    }
  }
  cudaEvent_t event;
};

Event::Event(bool withTime) : hasTiming_(withTime) {
  event_ = std::make_unique<EventImpl>();
  CUDA_CHECK(cudaEventCreateWithFlags(
      &event_->event, withTime ? 0 : cudaEventDisableTiming));
}

Event::~Event() {}

void Event::record(Stream& stream) {
  CUDA_CHECK(cudaEventRecord(event_->event, stream.stream_->stream));
  recorded_ = true;
}

void Event::wait() {
  CUDA_CHECK(cudaEventSynchronize(event_->event));
}

bool Event::query() const {
  auto rc = cudaEventQuery(event_->event);
  if (rc == ::cudaErrorNotReady) {
    return false;
  }
  CUDA_CHECK(rc);
  return true;
}

void Event::wait(Stream& stream) {
  CUDA_CHECK(cudaStreamWaitEvent(stream.stream_->stream, event_->event));
}

/// Returns time in ms betweene 'this' and an earlier 'start'. Both events must
/// enable timing.
float Event::elapsedTime(const Event& start) const {
  float ms;
  if (!hasTiming_ || !start.hasTiming_) {
    waveError("Event timing not enabled");
  }
  CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_->event, event_->event));
  return ms;
}
namespace {
struct KernelEntry {
  const char* name;
  const void* func;
};

int32_t numKernelEntries = 0;
KernelEntry kernelEntries[200];
} // namespace

bool registerKernel(const char* name, const void* func) {
  kernelEntries[numKernelEntries].name = name;
  kernelEntries[numKernelEntries].func = func;
  ++numKernelEntries;
  if (numKernelEntries >= sizeof(kernelEntries) / sizeof(kernelEntries[0])) {
    LOG(ERROR) << "Reserve more space in kernelEntries";
    exit(1);
  }
  return true;
}

KernelInfo kernelInfo(const void* func) {
  cudaFuncAttributes attrs;
  CUDA_CHECK_FATAL(cudaFuncGetAttributes(&attrs, func));
  KernelInfo info;
  info.numRegs = attrs.numRegs;
  info.maxThreadsPerBlock = attrs.maxThreadsPerBlock;
  info.sharedMemory = attrs.sharedSizeBytes;
  int max;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max, func, 256, 0);
  info.maxOccupancy0 = max;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max, func, 256, 256 * 32);
  info.maxOccupancy32 = max;

  return info;
}

std::string KernelInfo::toString() const {
  std::stringstream out;
  out << "NumRegs=" << numRegs << " maxThreadsPerBlock= " << maxThreadsPerBlock
      << " sharedMemory=" << sharedMemory
      << " occupancy 256,  0=" << maxOccupancy0
      << " occupancy 256,32=" << maxOccupancy32;
  return out.str();
}

KernelInfo getRegisteredKernelInfo(const char* name) {
  for (auto i = 0; i < numKernelEntries; ++i) {
    if (strcmp(name, kernelEntries[i].name) == 0) {
      return kernelInfo(kernelEntries[i].func);
    }
  }
  return KernelInfo();
}

void printKernels() {
  for (auto i = 0; i < numKernelEntries; ++i) {
    std::cout << kernelEntries[i].name << " - "
              << getRegisteredKernelInfo(kernelEntries[i].name).toString()
              << std::endl;
  }
}

} // namespace facebook::velox::wave
