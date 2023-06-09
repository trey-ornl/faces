#pragma once

#include <vector>

#include "gpu.h"


template <typename T, int N>
class DArray {
  public:
    template <typename I, typename... Is>
    DArray(const I size, const Is... sizes):
      values_(nullptr),
      first_(this)
    {
      strides_[0] = size;
      init<1,Is...>(sizes...);
      alloc();
   }

    DArray(const int size):
      strides_{size},
      values_(nullptr),
      first_(this)
      { alloc(); }

    ~DArray()
    {
      if (this == first_) CHECK(hipFree(values_));
      values_ = nullptr;
      for (int i = 0; i < N; i++) strides_[i] = 0;
    }

    __forceinline__ __host__ __device__ long index(const int i) { return i; }

    template <typename I, typename... Is>
    __forceinline__ __host__ __device__ long index(const I i, const Is... is) const
    { return i+stride<0,Is...>(is...); }

    __forceinline__ __device__ T &operator[](const long i) const { return values_[i]; }

    template <typename... Is>
    __forceinline__ __device__ T &operator()(const Is... is) const { return values_[index(is...)]; }

    void copy(std::vector<T> &that) const
    {
      that.resize(strides_[N-1]);
      const long bytes = sizeof(T)*that.size();
      CHECK(hipMemcpy(that.data(),values_,bytes,hipMemcpyDeviceToHost));
    }

    const T *data() const { return values_; }
    T *data() { return values_; }

    template <typename ...Is>
    const T *data(const Is ...is) const
    { return values_+index(is...); }

    template <typename ...Is>
    T *data(const Is ...is)
    { return values_+index(is...); }

    __host__ __device__ long size() const { return strides_[N-1]; }

    __host__ __device__ long size(const int i) const
    {
      assert((i >= 0) && (i < N));
      if (i == 0) return strides_[0];
      return strides_[i]/strides_[i-1];
    }

  protected:
    long strides_[N];
    T * __restrict values_;
    DArray<T,N> *first_;

    void alloc()
    {
      const long bytes = sizeof(T)*strides_[N-1];
      CHECK(hipMalloc(const_cast<T**>(&values_),bytes));
      CHECK(hipMemset(values_,0,bytes));
    }
 
    template <int M, typename I, typename... Is>
    void init(const I size, const Is... sizes)
    {
      strides_[M] = size*strides_[M-1];
      init<M+1,Is...>(sizes...);
    }

    template <int M, typename I>
    void init(const I size)
    {
      static_assert(M == N-1,"bad init");
      if (N > 1) {
        // Page align largest dimension to work around Infiniband limitation with AMD GPU memory for MPI
        constexpr int page = 4096/sizeof(T);
        strides_[M-1] = ((strides_[M-1]-1)/page+1)*page;
      }
      strides_[M] = size*strides_[M-1];
    }

    template <int M, typename I, typename... Is>
    __forceinline__ __host__ __device__ long stride(const I i, const Is... is) const
    { return i*strides_[M]+stride<M+1,Is...>(is...); }

    template <int M, typename I>
    __forceinline__ __host__ __device__ long stride(const I i) const
    {
      static_assert(M == N-2,"bad stride");
      return i*strides_[M];
    }
};

