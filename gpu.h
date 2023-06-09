#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "hip/hip_runtime.h"
#ifdef __NVCC__
#define LAUNCH(...) hipLaunchKernelGGL(__VA_ARGS__)
#else
#include "hip/hip_ext.h"
#define LAUNCH(...) hipLaunchKernelGGL(__VA_ARGS__)
//#define LAUNCH(F,G,B,M,S,...) hipExtLaunchKernelGGL(F,G,B,M,S,nullptr,nullptr,1,__VA_ARGS__)
#endif

static void checkHip(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) checkHip(X,__FILE__,__LINE__)

#ifdef __HIPCC__

static constexpr int gpuMaxThreads = 64;
static constexpr int gpuMinThreads = 64;

static void __attribute__((unused)) print(const dim3 g, const dim3 b)
{
  std::cout<<"Launching block "<<b.x<<'x'<<b.y<<'x'<<b.z<<" on grid "<<g.x<<'x'<<g.y<<'x'<<g.z<<std::endl;
}

template <typename F>
__global__ __launch_bounds__(gpuMinThreads) void gpuRun(F f)
{ f(); }

template <typename F>
void gpuLaunch(F f, const hipStream_t stream = 0)
{
  LAUNCH(gpuRun,1,1,0,stream,f);
}

template <typename F>
void gpuLaunch(const dim3 &gridDim, const dim3 &blockDim, F f, const hipStream_t stream = 0)
{
  assert(blockDim.x*blockDim.y*blockDim.z <= gpuMaxThreads);
  LAUNCH(gpuRun,gridDim,blockDim,0,stream,f);
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
void gpuFor(const std::initializer_list<int> il0, F f, const hipStream_t stream = 0)
{
  int d0;
  int lo0;
  set(il0,lo0,d0);
  assert(d0 <= gpuMaxThreads);
  LAUNCH(gpuRun0x1,1,d0,0,stream,f,lo0);
}

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun1x1(F f, const int lo0, const int lo1)
{ f(threadIdx.x+lo0,blockIdx.x+lo1); }

template <typename F>
__global__ __launch_bounds__(gpuMaxThreads) void gpuRun0x2(F f, const int lo0, const int lo1)
{ f(threadIdx.x+lo0,threadIdx.y+lo1); }

template <typename F>
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, F f, const hipStream_t stream = 0)
{
  int d0, d1;
  int lo0, lo1;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  assert(d0 <= gpuMaxThreads);
  if (d0*d1 > gpuMaxThreads) LAUNCH(gpuRun1x1,d1,d0,0,stream,f,lo0,lo1);
  else LAUNCH(gpuRun0x2,1,dim3(d0,d1),0,stream,f,lo0,lo1);
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
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, F f, const hipStream_t stream = 0)
{
  int d0, d1, d2;
  int lo0, lo1, lo2;
  set(il0,lo0,d0);
  set(il1,lo1,d1);
  set(il2,lo2,d2);
  assert(d0 <= gpuMaxThreads);
  const int d01 = d0*d1;
  const int d012 = d01*d2;
  if (d01 > gpuMaxThreads) LAUNCH(gpuRun2x1,dim3(d1,d2),d0,0,stream,f,lo0,lo1,lo2);
  else if (d012 > gpuMaxThreads) LAUNCH(gpuRun1x2,d2,dim3(d0,d1),0,stream,f,lo0,lo1,lo2);
  else LAUNCH(gpuRun0x3,1,dim3(d0,d1,d2),0,stream,f,lo0,lo1,lo2);
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
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, const std::initializer_list<int> il3, F f, const hipStream_t stream = 0)
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
  if (d01 > gpuMaxThreads) LAUNCH(gpuRun3x1,dim3(d1,d2,d3),d0,0,stream,f,lo0,lo1,lo2,lo3);
  else if (d012 > gpuMaxThreads) LAUNCH(gpuRun2x2,dim3(d2,d3),dim3(d0,d1),0,stream,f,lo0,lo1,lo2,lo3);
  else LAUNCH(gpuRun1x3,d3,dim3(d0,d1,d2),0,stream,f,lo0,lo1,lo2,lo3);
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
void gpuFor(const std::initializer_list<int> il0, const std::initializer_list<int> il1, const std::initializer_list<int> il2, const std::initializer_list<int> il3, const std::initializer_list<int> il4, F f, const hipStream_t stream = 0)
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
    LAUNCH(gpuRun4x1,dim3(d12,d3,d4),d0,0,stream,f,lo0,lo1,lo2,lo3,lo4,d1);
  } else if (d012 > gpuMaxThreads) {
    LAUNCH(gpuRun3x2,dim3(d2,d3,d4),dim3(d0,d1),0,stream,f,lo0,lo1,lo2,lo3,lo4); 
  } else {
    LAUNCH(gpuRun2x3,dim3(d3,d4),dim3(d0,d1,d2),0,stream,f,lo0,lo1,lo2,lo3,lo4);
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

static constexpr int gpuBlockSize = 64;
static constexpr int gpuBlocksPerWG = 8;
static constexpr int gpuPerThread = 1;
static constexpr int gpuPerBlock = gpuBlockSize*gpuPerThread;

#if 0

template <typename ...Ts>
struct Loop {
  static constexpr int N = sizeof...(Ts)-1;
  using F = typename std::tuple_element<N,std::tuple<Ts...>>::type;
  F body;
  std::array<int,N> lo;
  std::array<int,N> n;
  int nb;

  Loop(Ts ...ts):
    body(std::get<N>(std::forward_as_tuple(ts...)))
  {
    init(0,ts...);
    nb = (n.back()-1)/gpuPerBlock+1;
    //std::cout<<' '<<nb;
  }

  void init(const int i, F)
  {
    assert(i == N);
  }

  template <typename ...Rs>
  void init(const int i, const Range r, const Rs ...rs)
  {
    lo[i] = r.lo;
    n[i] = r.n;
    if (i > 0) n[i] *= n[i-1];
    init(i+1,rs...);
  }

  __device__ void run(int j) const
  {
    std::array<int,N> js;
    for (int i = N-1; i > 0; i--) {
      js[i] = j/n[i-1];
      j -= js[i]*n[i-1];
      js[i] += lo[i];
    }
    js[0] = lo[0]+j;
    std::apply(body,js);
  }
};

static inline __device__ void runLoops(int) {}

template <typename L, typename ...Ls>
__device__ void runLoops(const int j, const L &loop, const Ls &...loops)
{
  if (j < loop.nb) {
    const int di = loop.nb*gpuBlockSize;
    const int n = loop.n.back();
    for (int i = j*gpuBlockSize+threadIdx.x; i < n; i += di) loop.run(i);
  } else {
    runLoops(j-loop.nb,loops...);
  }
}

template <typename ...Ts>
__global__ __launch_bounds__(gpuBlockSize*gpuBlocksPerWG) void gpuLoops(const Ts ...ts)
{
  const int j = blockIdx.x*gpuBlocksPerWG+threadIdx.y;
  runLoops(j,ts...);
}

template <typename L>
int numBlocks(const L &loop) { return loop.nb; }

template <typename L, typename ...Ls>
int numBlocks(const L &loop, const Ls &...loops)
{
  return loop.nb+numBlocks(loops...);
}

template <typename ...Ts>
void kernel(const hipStream_t stream, const Ts &...ts)
{
  const int nb = numBlocks(ts...);
  //std::cout<<" = "<<nb<<std::endl;
  LAUNCH(gpuLoops<Ts...>,(nb-1)/gpuBlocksPerWG+1,dim3(gpuBlockSize,gpuBlocksPerWG),0,stream,ts...);
  CHECK(hipPeekAtLastError());
}
#endif

#define GPU_LAMBDA [=] __device__

#endif
