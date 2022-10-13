/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2022, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <algorithm>
#include <cstdlib>
#if _WIN32 || __USE_ISOC11
#include <malloc.h>
#else
#include <malloc/malloc.h>
#endif
#include <numeric>
#include <vector>
#include "options.h"

#include "dimensions.h"

namespace sadl
{
// tensor between layers: depth height width (or width height?)
template<typename T, std::size_t Alignment> struct aligned_allocator
{
  using pointer         = T *;
  using const_pointer   = const T *;
  using reference       = T &;
  using const_reference = const T &;
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  pointer       address(reference r) const { return &r; }
  const_pointer address(const_reference s) const { return &s; }
  size_type     max_size() const { return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T); }
  template<typename U> struct rebind
  {
    typedef aligned_allocator<U, Alignment> other;
  };

  bool operator!=(const aligned_allocator &other) const { return !(*this == other); }
  void construct(pointer p, const_reference t) const
  {
    void *const pv = static_cast<void *>(p);
    new (pv) T(t);
  }
  void destroy(T *const p) const { p->~T(); }
  bool operator==(const aligned_allocator & /*other*/) const { return true; }

  aligned_allocator()                          = default;
  aligned_allocator(const aligned_allocator &) = default;
  ~aligned_allocator()                         = default;
  aligned_allocator &operator=(const aligned_allocator &) = delete;

  template<typename U> aligned_allocator(const aligned_allocator<U, Alignment> &) {}

  pointer allocate(const std::size_t n) const
  {
    if (n == 0)
      return nullptr;
    size_t s = ((n * sizeof(T) + Alignment - 1) / Alignment) * Alignment;

#if _WIN32
#if __MINGW32__
    void *const pv = __mingw_aligned_malloc(s, Alignment);
#else
    void *const pv = _aligned_malloc(s, Alignment);
#endif
#else
#if __USE_ISOC11
    void *const pv = aligned_alloc(Alignment, s);
#else
    void *pv = nullptr;
    if (posix_memalign(&pv, Alignment, s))
    {
      throw std::bad_alloc();
    }
#endif
#endif

    if (!pv)
      throw std::bad_alloc();
    return static_cast<T *>(pv);
  }

#ifdef _WIN32
  void deallocate(T *const p, const std::size_t n) const { _aligned_free(p); }
#else
  void deallocate(T *const p, const std::size_t /*n*/) const { free(p); }
#endif

  template<typename U> pointer allocate(const std::size_t n, const U * /* const hint */) const { return allocate(n); }
};

template<typename T> struct ComputationType
{
};

// predecl for friendness
template<typename T> class Tensor;
template<typename T> void swap(Tensor<T> &t0, Tensor<T> &t1);
template<typename T> void swapData(Tensor<T> &t0, Tensor<T> &t1);

template<typename T> class Tensor
{
public:
  using value_type     = T;
  using Data           = std::vector<value_type, aligned_allocator<value_type, 64>>;
  using iterator       = typename Data::iterator;
  using const_iterator = typename Data::const_iterator;
  static bool skip_border;   // to replace by inline global C++17

  Tensor() = default;
  explicit Tensor(Dimensions d);

  void resize(Dimensions d);

  // lineqar access
  value_type &operator[](int i);
  value_type  operator[](int i) const;

  // tensor access
  value_type &operator()(int i);
  value_type  operator()(int i) const;

  value_type &operator()(int i, int j);
  value_type  operator()(int i, int j) const;

  value_type &operator()(int i, int j, int k);
  value_type  operator()(int i, int j, int k) const;

  value_type &      operator()(int i, int j, int k, int l);
  value_type        operator()(int i, int j, int k, int l) const;
  const value_type *addr(int i, int j, int k, int l) const;

  bool in(int i) const;
  bool in(int i, int j) const;
  bool in(int i, int j, int k) const;
  bool in(int i, int j, int k, int l) const;
  void fill(value_type value);

  const Dimensions &dims() const;
  int64_t size() const;

  const value_type *data() const { return data_.data(); }
  value_type *      data() { return data_.data(); }

  iterator       begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin(); }
  iterator       end() { return data_.end(); }
  const_iterator end() const { return data_.end(); }

  int                  quantizer   = 0;   // for int
  int                  border_skip = 0;
  static constexpr int64_t kMaxSize    = 32LL*1024*1024*1024;

  Data &getData() { return data_; }

private:
  Dimensions  dims_;
  Data        data_;
  friend void swap<>(Tensor<T> &t0, Tensor<T> &t1);
  friend void swapData<>(Tensor<T> &t0, Tensor<T> &t1);
#if DEBUG_PRINT
public:
  static bool verbose_;
#endif
};

// spe
template<> struct ComputationType<float>
{
  using type                = float;
  static constexpr type max = std::numeric_limits<float>::max();
  static void           quantize(type, int) {}     // nothing to do
  static void           shift_left(type, int) {}   // nothing to do
};

template<> struct ComputationType<int32_t>
{
  using type                = int64_t;
  static constexpr type max = std::numeric_limits<int32_t>::max();
  static void           quantize(type &z, int q) { z >>= q; }
  static void           shift_left(type &z, int q) { z <<= q; }
  static void           quantize(int32_t &z, int q) { z >>= q; }
  static void           shift_left(int32_t &z, int q) { z <<= q; }
};

template<> struct ComputationType<int16_t>
{
  using type                = int32_t;
  static constexpr type max = std::numeric_limits<int16_t>::max();
  static void           quantize(type &z, int q) { z >>= q; }
  static void           shift_left(type &z, int q) { z <<= q; }
  static void           quantize(int16_t &z, int q) { z >>= q; }
  static void           shift_left(int16_t &z, int q) { z <<= q; }
};

// impl
template<typename T> bool Tensor<T>::skip_border = false;

template<typename T> void swap(Tensor<T> &t0, Tensor<T> &t1)
{
  std::swap(t0.dims_, t1.dims_);
  std::swap(t0.data_, t1.data_);
  std::swap(t0.quantizer, t1.quantizer);
  std::swap(t0.border_skip, t1.border_skip);
}

template<typename T> void swapData(Tensor<T> &t0, Tensor<T> &t1)
{
  assert(t0.size() == t1.size());
  std::swap(t0.data_, t1.data_);
  std::swap(t0.quantizer, t1.quantizer);
  std::swap(t0.border_skip, t1.border_skip);
}

template<typename T> Tensor<T>::Tensor(Dimensions d)
{
  resize(d);
}

template<typename T> const Dimensions &Tensor<T>::dims() const
{
  return dims_;
}

template<typename T> int64_t Tensor<T>::size() const
{
  return data_.size();
}

template<typename T> void Tensor<T>::resize(Dimensions d)
{
  dims_ = d;
  int64_t m = dims_.nbElements();
  //    for(auto x: dims_) m*=x;
  assert(m < kMaxSize);
  data_.resize(m);
}

// TODO: variadic template to define all accesors
template<typename T> T &Tensor<T>::operator[](int i)
{
  return data_[i];
}

template<typename T> T &Tensor<T>::operator()(int i)
{
  assert(dims_.size() == 1);
  assert(i < dims_[0] && i >= 0);

  return data_[i];
}

template<typename T> bool Tensor<T>::in(int i) const
{
  return dims_.size() == 1 && i < dims_[0] && i >= 0;
}

template<typename T> T Tensor<T>::operator[](int i) const
{
  return data_[i];
}

template<typename T> T Tensor<T>::operator()(int i) const
{
  assert(dims_.size() == 1);
  assert(i < dims_[0] && i >= 0);

  return data_[i];
}

template<typename T> T &Tensor<T>::operator()(int i, int j)
{
  assert(dims_.size() == 2);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);

  return data_[(int64_t)dims_[1] * i + j];
}

template<typename T> T Tensor<T>::operator()(int i, int j) const
{
  assert(dims_.size() == 2);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);

  return data_[(int64_t)dims_[1] * i + j];
}

template<typename T> bool Tensor<T>::in(int i, int j) const
{
  return dims_.size() == 2 && i < dims_[0] && i >= 0 && j < dims_[1] && j >= 0;
}

template<typename T> T &Tensor<T>::operator()(int i, int j, int k)
{
  assert(dims_.size() == 3);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);
  assert(k < dims_[2] && k >= 0);

  return data_[(int64_t)dims_[2] * (dims_[1] * i + j) + k];
}

template<typename T> T Tensor<T>::operator()(int i, int j, int k) const
{
  assert(dims_.size() == 3);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);
  assert(k < dims_[2] && k >= 0);

  return data_[(int64_t)dims_[2] * (dims_[1] * i + j) + k];
}

template<typename T> bool Tensor<T>::in(int i, int j, int k) const
{
  return dims_.size() == 3 && i < dims_[0] && i >= 0 && j < dims_[1] && j >= 0 && k < dims_[2] && k >= 0;
}

template<typename T> T &Tensor<T>::operator()(int i, int j, int k, int l)
{
  assert(dims_.size() == 4);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);
  assert(k < dims_[2] && k >= 0);
  assert(l < dims_[3] && l >= 0);

  return data_[(int64_t)dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

template<typename T> bool Tensor<T>::in(int i, int j, int k, int l) const
{
  return dims_.size() == 4 && i < dims_[0] && i >= 0 && j < dims_[1] && j >= 0 && k < dims_[2] && k >= 0 && l < dims_[3] && l >= 0;
}

template<typename T> const T *Tensor<T>::addr(int i, int j, int k, int l) const
{
  assert(dims_.size() == 4);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);
  assert(k < dims_[2] && k >= 0);
  assert(l < dims_[3] && l >= 0);
  return &data_[(int64_t)dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

template<typename T> T Tensor<T>::operator()(int i, int j, int k, int l) const
{
  assert(dims_.size() == 4);
  assert(i < dims_[0] && i >= 0);
  assert(j < dims_[1] && j >= 0);
  assert(k < dims_[2] && k >= 0);
  assert(l < dims_[3] && l >= 0);
  return data_[(int64_t)dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
}

template<typename T> void Tensor<T>::fill(value_type value)
{
  std::fill(data_.begin(), data_.end(), value);
}

}   // namespace sadl

#include <iostream>
#include <sstream>

#if DEBUG_PRINT
template<typename T> bool sadl::Tensor<T>::verbose_ = true;

#define SADL_DBG(X)                                                                                                                                            \
  if (sadl::Tensor<T>::verbose_)                                                                                                                               \
  {                                                                                                                                                            \
    X;                                                                                                                                                         \
  }
#else
#define SADL_DBG(X)
#endif

namespace sadl
{
template<typename T> std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
  // adhoc
  if (t.dims().size() == 4u)
  {
    out << "[";
    if (t.dims()[0] > 1)
      out << '\n';
    for (int k = 0; k < t.dims()[0]; ++k)
    {
      out << " [";
      if (t.dims()[1] > 1)
        out << '\n';
      for (int d = 0; d < t.dims()[1]; ++d)
      {
        out << "  [";
        if (t.dims()[2] > 1)
          out << '\n';
        for (int i = 0; i < t.dims()[2]; ++i)
        {
          out << "   [";
          for (int j = 0; j < t.dims()[3]; ++j)
            out << t(k, d, i, j) << ' ';
          out << "   ]";
          if (t.dims()[2] > 1)
            out << '\n';
        }
        out << "  ]";
        if (t.dims()[1] > 1)
          out << '\n';
      }
      out << " ]";
      if (t.dims()[0] > 1)
        out << '\n';
    }
    out << "]";
  }
  else if (t.dims().size() == 3u)
  {
    out << "[";
    for (int d = 0; d < t.dims()[0]; ++d)
    {
      out << " [";
      if (t.dims()[0] > 1)
        out << '\n';
      for (int i = 0; i < t.dims()[1]; ++i)
      {
        out << "[";
        if (t.dims()[1] > 1)
          out << '\n';
        for (int j = 0; j < t.dims()[2]; ++j)
          out << t(d, i, j) << '\t';
        out << "  ]";
        if (t.dims()[1] > 1)
          out << '\n';
      }
      out << " ]";
      if (t.dims()[0] > 1)
        out << '\n';
    }
    out << "]";
  }
  else if (t.dims().size() == 2u)
  {
    out << "[";
    for (int i = 0; i < t.dims()[0]; ++i)
    {
      out << "[";
      if (t.dims()[0] > 1)
        out << '\n';
      for (int j = 0; j < t.dims()[1]; ++j)
        out << t(i, j) << ' ';
      out << " ]";
      if (t.dims()[0] > 1)
        out << '\n';
    }
    out << "]\n";
  }
  else if (t.dims().size() == 1u)
  {
    out << "[";
    for (int j = 0; j < t.dims()[0]; ++j)
      out << t(j) << ' ';
    out << "]";
  }
  else
  {
    out << "TODO\n";
  }
  return out;
}

}   // namespace sadl
