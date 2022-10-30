/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
//#include <cuda/semaphore>
#include <vector>
#include <thread>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  a[i] += i;
}

float maxError(float *a, int n) 
{
#if 0
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
#else
  return 0;
#endif
}

// Arguments: n-repeats n-streams kbytesperstream dev1 dev2 ...
int main(int argc, char **argv)
{
  int nRepeats = 1;
  	int 	nStreams = 4;
	int n256floats = 4096;
  if (argc >= 4) {
    nRepeats = atoi(argv[1]);
    nStreams = atoi(argv[2]);
		    n256floats = atoi(argv[3]);
		    }
const int blockSize = 256;
  const long n = n256floats * blockSize * nStreams;
  const long streamSize = n / nStreams;
  const long streamBytes = streamSize * sizeof(float);
  const long bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 4) devId = atoi(argv[4]);
  std::vector<int> devices;
  devices.push_back(devId);
  printf("Devices: %d", devId);
  for (auto i = 5; i< argc; ++i) {
    devices.push_back(atoi(argv[i]));
    printf(", %d", devices.back());
  }
  printf("\n");
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  std::vector<std::thread > threads;
  for (auto dev : devices) {
    threads.push_back(std::thread([&]() {
      checkCuda( cudaSetDevice(dev) );
      // Allocate unified memory
      int32_t managedBytes = 32 * 1024;
      char* managed;
      checkCuda( cudaMallocManaged((void**)&managed, managedBytes));
      

  
      // allocate pinned host memory and device memory
      float *a, *d_a;
      printf("%dKB\n", bytes / 1024);
      checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
      checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device
      
      float ms; // elapsed time in milliseconds
      float singleMs = 0;
      float async1Ms = 0;
      float async2Ms = 0;
      
      // create events and streams
      cudaEvent_t startEvent, stopEvent, dummyEvent;
      cudaStream_t* stream = (cudaStream_t*)malloc (nStreams * sizeof(cudaStream_t));
      checkCuda( cudaEventCreate(&startEvent) );
      checkCuda( cudaEventCreate(&stopEvent) );
      checkCuda( cudaEventCreate(&dummyEvent) );
      for (int i = 0; i < nStreams; ++i)
	checkCuda( cudaStreamCreate(&stream[i]) );
      
      // baseline case - sequential transfer and execute
      memset(a, 0, bytes);
      for (int repeat = 0; repeat < nRepeats; ++repeat) { 
	checkCuda( cudaEventRecord(startEvent,0) );
	checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
	kernel<<<n/blockSize, blockSize>>>(d_a, 0);
	checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	singleMs += ms;
      }
      float volume = bytes * nRepeats;
      printf("Time for sequential transfer and execute (ms): %f %f GB/s\n", singleMs, 2.0 * (volume / (1 << 30)) / (singleMs / 1000)  );
      printf("  max error: %e\n", maxError(a, n));
      
      // asynchronous version 1: loop over {copy, kernel, copy}
      memset(a, 0, bytes);
      for (int repeat = 0; repeat < nRepeats; ++repeat) {
	checkCuda( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; ++i) {
	  int offset = i * streamSize;
	  checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
				     streamBytes, cudaMemcpyHostToDevice, 
				     stream[i]) );
	  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
	  checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
				     streamBytes, cudaMemcpyDeviceToHost,
				     stream[i]) );
    }
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    async1Ms += ms;
  }
  volume = streamBytes * nRepeats * nStreams;

  printf("Time for asynchronous V1 transfer and execute (ms): %f %f GB/s\n", async1Ms, 2.0 * (volume / (1 << 30)) / (async1Ms / 1000)  );
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2: 
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  for (int repeat = 0; repeat < nRepeats; ++repeat) {
    checkCuda( cudaEventRecord(startEvent,0) );
    for (int i = 0; i < nStreams; ++i)
      {
	int offset = i * streamSize;
	checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
				   streamBytes, cudaMemcpyHostToDevice,
				   stream[i]) );
      }
    for (int i = 0; i < nStreams; ++i)
      {
	int offset = i * streamSize;
	kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
      }
    for (int i = 0; i < nStreams; ++i)
      {
	int offset = i * streamSize;
	checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
				   streamBytes, cudaMemcpyDeviceToHost,
				   stream[i]) );
      }
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    async2Ms += ms;
  }
    printf("Time for asynchronous V2 transfer and execute (ms): %f\n", async2Ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);
  cudaFree(managed);
    }));
  }
    for (auto& thread : threads) {
	thread.join();
      }
      printf("Completed %d threads", threads.size());
      return 0;
}
