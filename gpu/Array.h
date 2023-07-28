#pragma once

#include <cstring>


template <typename T, int N>
class Array {
  public:
    template <typename I, typename... Is>
    Array(const I size, const Is... sizes)
    {
      strides_[0] = sizes_[0] = size;
      init<1,Is...>(sizes...);
      values_ = new T[strides_[N-1]];
      memset(values_,0,strides_[N-1]*sizeof(T));
    }

    ~Array()
    {
      delete [] values_;
      values_ = nullptr;
    }

    template <typename I, typename... Is>
    long index(const I i, const Is... is) const
    { return i+stride<0,Is...>(is...); }

    T &operator[](const long i) { return values_[i]; }
    T operator[](const long i) const { return values_[i]; }

    template <typename... Is>
    T &operator()(const Is... is) { return values_[index(is...)]; }

    template <typename... Is>
    T operator()(const Is... is) const { return values_[index(is...)]; }

    T *data() { return values_; }
    const T *data() const { return values_; }

    long size() const { return strides_[N-1]; }

    long size(const int i) const { return sizes_[i]; }

    const long *sizes() const { return sizes_; }

    const long *strides() const { return strides_; }

  protected:
    long sizes_[N];
    long strides_[N];
    T * __restrict values_;

    template <int M, typename I, typename... Is>
    void init(const I size, const Is... sizes)
    {
      sizes_[M] = size;
      strides_[M] = size*strides_[M-1];
      init<M+1,Is...>(sizes...);
    }

    template <int M, typename I>
    void init(const I size)
    {
      static_assert(M == N-1,"bad init");
      sizes_[M] = size;
      strides_[M] = size*strides_[M-1];
    }

    template <int M, typename I, typename... Is>
    long stride(const I i, const Is... is) const
    { return i*strides_[M]+stride<M+1,Is...>(is...); }

    template <int M, typename I>
    long stride(const I i) const
    {
      static_assert(M == N-2,"bad stride");
      return i*strides_[M];
    }
};


