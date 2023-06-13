#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuError_t hipError_t
#define gpuFree hipFree
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetErrorName hipGetErrorName
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemset hipMemset
#define gpuSetDevice hipSetDevice
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuSuccess hipSuccess
#define WARPSIZE warpSize
#endif

#ifdef __NVCOMPILER
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuError_t cudaError_t
#define gpuFree cudaFree
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetErrorName cudaGetErrorName
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemset cudaMemset
#define gpuSetDevice cudaSetDevice
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuSuccess cudaSuccess
#define WARPSIZE 32
#endif

static void checkGPU(const gpuError_t err, const char *const file, const int line)
{
  if (err == gpuSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,gpuGetErrorName(err),gpuGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) checkGPU(X,__FILE__,__LINE__)

#if defined(__HIPCC__) || defined(__CUDACC__)

static constexpr int gpuMaxThreads = WARPSIZE;
static constexpr int gpuMinThreads = WARPSIZE;

template <typename F>
__global__ __launch_bounds__(gpuMinThreads) void gpuRun(F f)
{ f(); }

template <typename F>
void gpuLaunch(F f, const gpuStream_t stream = 0)
{
  gpuRun<<<1,1,0,stream>>>(f);
}

template <typename F>
void gpuLaunch(const dim3 &gridDim, const dim3 &blockDim, F f, const gpuStream_t stream = 0)
{
  assert(blockDim.x*blockDim.y*blockDim.z <= gpuMaxThreads);
  gpuRun<<<gridDim,blockDim,0,stream>>>(f);
}

static inline void set(const std::initializer_list<int> il, int &lo, int &d)
{
  assert(il.size() > 0);
  assert(il.size() < 3);
  if (il.size() == 2) {
    lo = *il.begin();
    d = *(il.begin()+1)-lo;
  } else {
    lo = 0;
    d = *il.begin();
  }
  assert(d > 0);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun0x1(F f, const int lo0)
{ f(threadIdx.x+lo0); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, F f, const gpuStream_t stream = 0)
{
  int d0;
  int lo0;
  set(il0,lo0,d0);
  assert(d0 <= gpuMaxThreads);
  gpuRun0x1<<<1,d0,0,stream>>>(f,lo0);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun1x1(F f, const int lo0, const int lo1)
{ f(threadIdx.x+lo0,blockIdx.x+lo1); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun0x2(F f, const int lo0, const int lo1)
{ f(threadIdx.x+lo0,threadIdx.y+lo1); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, F f, const gpuStream_t stream = 0)
{
  int d0, d1;
  int lo0, lo1;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  assert(d0 <= gpuMaxThreads);
  if (d0*d1 > gpuMaxThreads) gpuRun1x1<<<d1,d0,0,stream>>>(f,lo0,lo1);
  else gpuRun0x2<<<1,dim3(d0,d1),0,stream>>>(f,lo0,lo1);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun2x1(F f, const int lo0, const int lo1, const int lo2)
{ f(threadIdx.x+lo0,blockIdx.x+lo1,blockIdx.y+lo2); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun1x2(F f, const int lo0, const int lo1, const int lo2)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,blockIdx.x+lo2); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun0x3(F f, const int lo0, const int lo1, const int lo2)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,threadIdx.z+lo2); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, F f, const gpuStream_t stream = 0)
{
  int d0, d1, d2;
  int lo0, lo1, lo2;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  set(il2,lo2,d2);
  assert(d0 <= gpuMaxThreads);
  const int d01 = d0*d1;
  const int d012 = d01*d2;
  if (d01 > gpuMaxThreads) gpuRun2x1<<<dim3(d1,d2),d0,0,stream>>>(f,lo0,lo1,lo2);
  else if (d012 > gpuMaxThreads) gpuRun1x2<<<d2,dim3(d0,d1),0,stream>>>(f,lo0,lo1,lo2);
  else gpuRun0x3<<<1,dim3(d0,d1,d2),0,stream>>>(f,lo0,lo1,lo2);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun3x1(F f, const int lo0, const int lo1, const int lo2, const int lo3)
{ f(threadIdx.x+lo0,blockIdx.x+lo1,blockIdx.y+lo2,blockIdx.z+lo3); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun2x2(F f, const int lo0, const int lo1, const int lo2, const int lo3)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,blockIdx.x+lo2,blockIdx.y+lo3); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun1x3(F f, const int lo0, const int lo1, const int lo2, const int lo3)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,threadIdx.z+lo2,blockIdx.x+lo3); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, const std::initializer_list<int> il3, F f, const gpuStream_t stream = 0)
{
  int d0, d1, d2, d3;
  int lo0, lo1, lo2, lo3;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  set(il2,lo2,d2);
  set(il3,lo3,d3);
  assert(d0 <= gpuMaxThreads);
  const int d01 = d0*d1;
  const int d012 = d01*d2;
  if (d01 > gpuMaxThreads) gpuRun3x1<<<dim3(d1,d2,d3),d0,0,stream>>>(f,lo0,lo1,lo2,lo3);
  else if (d012 > gpuMaxThreads) gpuRun2x2<<<dim3(d2,d3),dim3(d0,d1),0,stream>>>(f,lo0,lo1,lo2,lo3);
  else gpuRun1x3<<<d3,dim3(d0,d1,d2),0,stream>>>(f,lo0,lo1,lo2,lo3);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun4x1(F f, const int lo0, const int lo1, const int lo2, const int lo3, const int lo4, const int d1)
{
  const int i1 = lo1+blockIdx.x%d1;
  const int i2 = lo2+blockIdx.x/d1;
  f(threadIdx.x+lo0,i1,i2,blockIdx.y+lo3,blockIdx.z+lo4);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun2x3(F f, const int lo0, const int lo1, const int lo2, const int lo3, const int lo4)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,threadIdx.z+lo2,blockIdx.x+lo3,blockIdx.y+lo4); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun3x2(F f, const int lo0, const int lo1, const int lo2, const int lo3, const int lo4)
{ f(threadIdx.x+lo0,threadIdx.y+lo1,blockIdx.x+lo2,blockIdx.y+lo3,blockIdx.z+lo4); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, const std::initializer_list<int> il3, const std::initializer_list<int> il4, F f, const gpuStream_t stream = 0)
{
  int d0, d1, d2, d3, d4;
  int lo0, lo1, lo2, lo3, lo4;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  set(il2,lo2,d2);
  set(il3,lo3,d3);
  set(il4,lo4,d4);
  assert(d0 <= gpuMaxThreads);
  const int d01 = d0*d1;
  const int d012 = d01*d2;
  if (d01 > gpuMaxThreads) {
    const int d12 = d1*d2;
    gpuRun4x1<<<dim3(d12,d3,d4),d0,0,stream>>>(f,lo0,lo1,lo2,lo3,lo4,d1);
  } else if (d012 > gpuMaxThreads) {
    gpuRun3x2<<<dim3(d2,d3,d4),dim3(d0,d1),0,stream>>>(f,lo0,lo1,lo2,lo3,lo4); 
  } else {
    gpuRun2x3<<<dim3(d3,d4),dim3(d0,d1,d2),0,stream>>>(f,lo0,lo1,lo2,lo3,lo4);
  }
}

// Multi-Loop-Block Kernels

struct Range {
  Range(const int hi): lo(0), n(hi) {}
  Range(const std::initializer_list<int> il): 
    lo(*(il.begin())),
    n(*(il.begin()+1)-lo)
  { assert(il.size() == 2); }
  int lo, n;
};

#define GPU_LAMBDA [=] __device__

#endif
