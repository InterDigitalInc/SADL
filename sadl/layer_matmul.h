/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2021, ITU/ISO/IEC
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
#include "layer.h"
#if __AVX2__
#include <immintrin.h>
#endif

namespace sadl {
namespace layers {

template <typename T>
class MatMul : public Layer<T> {
 public:
  using Layer<T>::Layer;
  using Layer<T>::out_;  // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

  PROTECTED : virtual bool loadInternal(std::istream &file, Version v) override;
  bool apply_dim2(std::vector<Tensor<T> *> &in);
  bool apply_dim3(std::vector<Tensor<T> *> &in);
#if __AVX2__
  bool apply_dim2_simd8(std::vector<Tensor<T> *> &in);
  bool apply_dim2_simd16(std::vector<Tensor<T> *> &in);
#endif
  int q_ = 0;
  DUMP_MODEL_EXT;
};

template <typename T>
bool MatMul<T>::apply(std::vector<Tensor<T> *> &in) {
  assert(in.size() == 2);
#if __AVX2__
#define MULT8_DIM2 apply_dim2_simd8
#define MULT16_DIM2 apply_dim2_simd16
#else
#define MULT8_DIM2 apply_dim2
#define MULT16_DIM2 apply_dim2
#endif
  const Tensor<T> &A{*in[0]};
  const Tensor<T> &B{*in[1]};
  out_.quantizer = A.quantizer - q_;
  assert(out_.quantizer >= 0);
  assert(in[1]->quantizer + q_ >= 0);
  int dum = A.dims().size();
  // cases:
  // A: always a tensor
  // B: tensor or const
  // 1- A [x] B[x] || A [x,y] B[y,z] || A [x,y,z] B[x,z,t]
  // 2- A [1,x] B[x] || A [1,x,y] B[y,z] || A [1,x,y,z] B[x,z,t]
  if (A.dims().size() - 1 == B.dims().size()) dum--;
  const int H{A.dims().back()};  // to be chnaged if SIMD for more than dim1 and dim2

  switch (dum) {
    case 2:
      if (H % 16 == 0)
        return MULT16_DIM2(in);
      else if (H % 8 == 0)
        return MULT8_DIM2(in);
      else
        return apply_dim2(in);
      break;
    case 3:
      return apply_dim3(in);
      break;
    default:
      std::cerr << "Logical error MatMul::apply(std::vector<Tensor<T> *> &in)" << A.dims() << ' ' << B.dims() << std::endl;
      return false;
  }
}

#if __AVX2__
template <>
inline bool MatMul<float>::apply_dim2_simd8(std::vector<Tensor<float> *> &in) {
  using T = float;
  const Tensor<T> &A{*in[0]};
  const Tensor<T> &B{*in[1]};
  const int last = A.dims().size() - 1;
  const int N{A.dims()[last - 1]};
  const int H{A.dims()[last]};
  const int R{B.dims()[1]};
  assert(H % 8 == 0);
  for (int b = 0; b < N; ++b) {
    float *optr=out_.data()+R*b;
    for (int t = 0; t < R; ++t) {
      __m256 s = _mm256_setzero_ps();
      const float *aptr = A.data() + b * H;
      const float *bptr = B.data() + t * H;  // T * i + t  (i, t); => B[t*H+i] if transposed
      for (int i = 0; i < H; i += 8, aptr += 8, bptr += 8) {
        __m256 a = _mm256_load_ps(aptr);
        __m256 b = _mm256_load_ps(bptr);
#if __FMA__
        s = _mm256_fmadd_ps(a, b, s);
#else
        s = _mm256_add_ps(s, _mm256_mul_ps(a, b)); // s+= _mm256_mul_ps(a, b); // _mm256_hadd_ps(s, _mm256_mul_ps(a, b));
#endif
      }
      optr[t] = sum8_float(s);// out_(b, t) = sum8_float(s);
    }
  }
  return true;
}

#if __AVX512F__
template <>
inline bool MatMul<float>::apply_dim2_simd16(std::vector<Tensor<float> *> &in)
{
  const Tensor<float> &A{*in[0]};
  const Tensor<float> &B{*in[1]};
  const int last = A.dims().size() - 1;
  const int N{A.dims()[last - 1]};
  const int H{A.dims()[last]};
  const int R{B.dims()[1]};
  assert(H % 16 == 0);
  for (int b = 0; b < N; ++b)
  {
    float* optr = out_.data() + R*b;
    for (int t = 0; t < R; ++t)
    {
      __m512 s = _mm512_setzero_ps();
      const float* aptr = A.data() + b*H;
      
      // The matrix of weights is transposed.
      const float* bptr = B.data() + t*H;
      for (int i = 0; i < H; i += 16, aptr += 16, bptr += 16)
      {
        __m512 a = _mm512_load_ps(aptr);
        __m512 b = _mm512_load_ps(bptr);
#if __FMA__
        s = _mm512_fmadd_ps(a, b, s);
#else
        s = _mm512_add_ps(s, _mm512_mul_ps(a, b));
#endif
      }
      optr[t] = sum16_float(s);
    }
  }
  return true;
}
#else
template <>
inline bool MatMul<float>::apply_dim2_simd16(std::vector<Tensor<float> *> &in) {
  return apply_dim2_simd8(in);
}
#endif

#if 0 // TODO
template <>
inline bool MatMul<int16_t>::apply_dim2_simd16(std::vector<Tensor<int16_t> *> &in) {
  using T = int16_t;
  const Tensor<T> &A{*in[0]};
  const Tensor<T> &B{*in[1]};
  const int H{A.dims().back()};
  const int R{B.dims()[1]};
  const int shift{in[1]->quantizer + q_};
  assert(H % 16 == 0);
  for (int t = 0; t < R; ++t) {
    __m256i s = _mm256_setzero_si256();
    const T *aptr = A.data();
    const T *bptr = B.data() + t * H;  // T * i + t  (i, t); => B[t*H+i] if transposed
    for (int i = 0; i < H; i += 16, aptr += 16, bptr += 16) {
      __m256i a = _mm256_load_si256((const __m256i *)aptr);
      __m256i b = _mm256_load_si256((const __m256i *)bptr);
      const __m256i mad0 = _mm256_madd_epi16(a, b);  // res in si32
      s = _mm256_add_epi32(s, mad0);
    }
    typename ComputationType<int32_t>::type z = (sum32_int16(s) >> shift);
    COUNTERS(z);
    SATURATE(z);
    out_[t] = z;
  }
  return true;
}
#endif

// to do
template <typename T>
bool MatMul<T>::apply_dim2_simd8(std::vector<Tensor<T> *> &in) {
  return apply_dim2(in);
}

// to do
template <typename T>
bool MatMul<T>::apply_dim2_simd16(std::vector<Tensor<T> *> &in) {
  return apply_dim2(in);
}

#endif

template <typename T>
bool MatMul<T>::apply_dim2(std::vector<Tensor<T> *> &in) {
  const Tensor<T> &A{*in[0]};
  const Tensor<T> &B{*in[1]};
  const int shift{in[1]->quantizer + q_};
  const int last = A.dims().size() - 1;
  const int N{A.dims()[last - 1]};
  const int H{A.dims()[last]};
  const int R{B.dims().back()};
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version matmul dim2 " << H<< std::endl;
#endif  // SIMD
  if (A.dims().size() == 2) {
    for (int b = 0; b < N; ++b) {
      const T *aptr = A.data() + H * b;  // A(b,i)   => A[H*b]
      for (int t = 0; t < R; ++t) {
        typename ComputationType<T>::type x = 0;
        const T *bptr = B.data() + t * H;  // T * i + t  (i, t); => B[t*H+i] if transposed
        for (int i = 0; i < H; ++i) {
          x += (typename ComputationType<T>::type)aptr[i] * bptr[i];  // A(b,i)*B(i, t);
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(b, t) = (T)x;
      }
    }
  } else {
    for (int b = 0; b < N; ++b) {
      const T *aptr = A.data() + H * b;  // A(0,b,i)  => A[H*b]
      for (int t = 0; t < R; ++t) {
        typename ComputationType<T>::type x = 0;
        const T *bptr = B.data() + t * H;  // T * i + t  (i, t); => B[t*H+i] if transposed
        for (int i = 0; i < H; ++i) {
          x += (typename ComputationType<T>::type)aptr[i] * bptr[i];  // A(0,b,i)*B(i, t);
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(0, b, t) = (T)x;
      }
    }
  }
  return true;
}

template <typename T>
bool MatMul<T>::apply_dim3(std::vector<Tensor<T> *> &in) {
  const Tensor<T> &A{*in[0]};
  const Tensor<T> &B{*in[1]};
  const int shift{in[1]->quantizer + q_};
  const int last = A.dims().size() - 1;
  const int N{A.dims()[last - 2]};
  const int H{A.dims()[last - 1]};
  const int W{A.dims()[last]};
  const int R{B.dims().back()};
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version matmul dim3" << std::endl;
#endif  // SIMD
  if (A.dims().size() == 3) {
    for (int b = 0; b < N; ++b) {
      for (int i = 0; i < H; ++i) {
        for (int t = 0; t < R; ++t) {
          typename ComputationType<T>::type x = 0;
          for (int j = 0; j < W; ++j) {
            x += (typename ComputationType<T>::type)A(b, i, j) * B(b, j, t);
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(b, i, t) = (T)x;
        }
      }
    }
  } else {  // size==4
    for (int b = 0; b < N; ++b) {
      for (int i = 0; i < H; ++i) {
        for (int t = 0; t < R; ++t) {
          typename ComputationType<T>::type x = 0;
          for (int j = 0; j < W; ++j) {
            x += (typename ComputationType<T>::type)A(0, b, i, j) * B(b, j, t);
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(0,b, i, t) = (T)x;
        }
      }
    }
  }
  return true;
}

template <typename T>
bool MatMul<T>::init(const std::vector<Tensor<T> *> &in) {
  // old:
  // multiply matrix of inner dim [a b ] or [x a b] or [ x a b y] (the [a b] matrix)
  // x and y should be same
  // new:
  // output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.

  SADL_DBG(std::cout << "  - input matmul: " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);

  if (in.size() != 2) {
    return false;
  }
  // cases:
  // A: always a tensor
  // B: const (because assumed transposed)
  // 1- A [x,y] B[y,z] || A [x,y,z] B[x,z,t] || A [1,x,y,z] B[1,x,z,t]
  // 2- A [1,x,y] B[y,z] || A [1,x,y,z] B[x,z,t]
  if (in[1]->dims().size() < 2 || in[1]->dims().size() > 3) {
    return false;
  }

  if (in[0]->dims().size() != in[1]->dims().size() && !(in[0]->dims().size() - 1 == in[1]->dims().size() && in[0]->dims()[0] == 1)) {
    return false;
  }

  if (in[0]->dims().size() != in[1]->dims().size() && !(in[0]->dims().size() - 1 == in[1]->dims().size() && in[0]->dims()[0] == 1)) {
    return false;
  }
  Dimensions dim = in[0]->dims();
  const int last = in[0]->dims().size() - 1;

  if (in[0]->dims().size() - 1 == in[1]->dims().size()) {
    for (int k = 1; k < last - 1; ++k) {
      if (in[0]->dims()[k] != in[1]->dims()[k - 1]) {
        return false;
      }
    }
    if (in[0]->dims()[last] != in[1]->dims()[last - 2]) {
      return false;
    }
  } else {
#if DEBUG_MODEL
    if (in[0]->dims()[0]!=1) std::cout<<"[WARN] suspicious operation (likely second input not a Const)"<<std::endl;
#endif
    // Excluding the last two dimensions, the dimension
    // of index i in the first input Tensor<T> must be equal
    // to the dimension of index i in the second input
    // Tensor<T>.
    for (int k = 0; k < last - 1; ++k) {
      if (in[0]->dims()[k] != in[1]->dims()[k]) {
        return false;
      }
    }
    if (in[0]->dims()[last] != in[1]->dims()[last - 1]) {
      return false;
    }
  }
  dim[last] = in[1]->dims().back();
  out_.resize(dim);
  SADL_DBG(std::cout << "  - output matmul: " << out_.dims() << std::endl);
  initDone_ = true;
  return true;
}

template <typename T>
bool MatMul<T>::loadInternal(std::istream &file, Version v) {
  if (v == Version::sadl01) {
    file.read((char *)&q_, sizeof(q_));
    SADL_DBG(std::cout << "  - q: " << q_ << std::endl);
  }
  return true;
}

}  // namespace layers
}  // namespace sadl



