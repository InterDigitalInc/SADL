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

namespace sadl {
namespace layers {

template <typename T>
class Mul : public Layer<T> {
 public:
  using Layer<T>::Layer;
  using Layer<T>::out_;  // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T>*>& in) override;
  virtual bool init(const std::vector<Tensor<T>*>& in) override;
  virtual bool mutateInput() const override { return true; }

  PROTECTED : virtual bool loadInternal(std::istream& file, Version v) override;
  int q_ = 0;
  bool apply_same_dim(std::vector<Tensor<T>*>& in);
  bool apply_singleton(std::vector<Tensor<T>*>& in);
  bool apply_dim2(std::vector<Tensor<T>*>& in);
  bool apply_dim3(std::vector<Tensor<T>*>& in);
  bool apply_dim4(std::vector<Tensor<T>*>& in);
#if __AVX2__
  bool apply_singleton_simd8(std::vector<Tensor<T>*>& in);
#endif
  DUMP_MODEL_EXT;
};

template <typename T>
bool Mul<T>::apply(std::vector<Tensor<T>*>& in) {
  assert(in.size() == 2);
  if (in[0] == in[1]) {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }
  swap(*in[0], out_);
  out_.quantizer -= q_;  // q0-q
  assert(out_.quantizer >= 0);
  assert(in[1]->quantizer + q_ >= 0);

  if (in[0]->dims() == in[1]->dims()) {  // product wise
    return apply_same_dim(in);
  } else if (in[1]->size() == 1) {  // broadcast single element
#if __AVX2__
    if (std::is_same<T,float>::value && (out_.size()%8==0) )
     return apply_singleton_simd8(in);
#endif
    return apply_singleton(in);
 } else if (in[0]->dims().size() == 2) {
    return apply_dim2(in);
 } else if (in[0]->dims().size() == 3) {
    return apply_dim3(in);
 } else if (in[0]->dims().size() == 4) {
    return apply_dim4(in);
}
 return false;
}

template <typename T>
bool Mul<T>::apply_same_dim(std::vector<Tensor<T>*>& in) {
  const int shift = in[1]->quantizer + q_;

#if __AVX2__ && DEBUG_MODEL
  std::cout << "[WARN] generic version mul " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif  // SIMD
  for (auto it0 = out_.begin(), it1 = in[1]->begin(); it0 != out_.end(); ++it0, ++it1) {
    typename ComputationType<T>::type x = *it0;
    x *= *it1;
    ComputationType<T>::quantize(x, shift);
    COUNTERS(x);
    SATURATE(x);
    *it0 = (T)x;
  }
  return true;
}

template <typename T>
bool Mul<T>::apply_singleton(std::vector<Tensor<T>*>& in) {
  const int shift = in[1]->quantizer + q_;
  const Tensor<T>& B = *in[1];
#if __AVX2__ && DEBUG_MODEL
  std::cout << "[WARN] generic version mul " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif  // SIMD
  const T value{B[0]};
  for (auto it0 = out_.begin(); it0 != out_.end(); ++it0) {
    typename ComputationType<T>::type x = *it0;
    x *= value;
    ComputationType<T>::quantize(x, shift);
    COUNTERS(x);
    SATURATE(x);
    *it0 = (T)x;
  }
  return true;
}

template <typename T>
bool Mul<T>::apply_dim2(std::vector<Tensor<T>*>& in) {
  const int shift = in[1]->quantizer + q_;

#if __AVX2__ && DEBUG_MODEL
  std::cout << "[WARN] generic version mul " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif  // SIMD

  const Tensor<T>& B = *in[1];
  const int N = in[0]->dims()[0];
  const int H = in[0]->dims()[1];
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i) {
      typename ComputationType<T>::type x = out_(n, i);
      x *= B[i];
      ComputationType<T>::quantize(x, shift);
      COUNTERS(x);
      SATURATE(x);
      out_(n, i) = (T)x;
    }
  return true;
}

template <typename T>
bool Mul<T>::apply_dim3(std::vector<Tensor<T>*>& in) {
  const int shift = in[1]->quantizer + q_;

#if __AVX2__ && DEBUG_MODEL
  std::cout << "[WARN] generic version mul " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif  // SIMD

  const Tensor<T>& B = *in[1];
  const int N = in[0]->dims()[0];
  const int H = in[0]->dims()[1];
  const int W = in[0]->dims()[2];
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j) {
        typename ComputationType<T>::type x = out_(n, i, j);
        x *= B[j];
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(n, i, j) = (T)x;
      }
  return true;
}

template <typename T>
bool Mul<T>::apply_dim4(std::vector<Tensor<T>*>& in) {
  const int shift = in[1]->quantizer + q_;

#if __AVX2__ && DEBUG_MODEL
  std::cout << "[WARN] generic version mul " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif  // SIMD
  assert(in[0]->dims()[0] == 1);

  const Tensor<T>& B = *in[1];
  const int N = in[0]->dims()[0];
  const int H = in[0]->dims()[1];
  const int W = in[0]->dims()[2];
  const int K = in[0]->dims()[3];
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        for (int k = 0; k < K; ++k) {
          typename ComputationType<T>::type x = out_(n, i, j, k);
          x *= B[k];
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(n, i, j, k) = (T)x;
        }

  return true;
}

#if __AVX2__
template <>
inline bool Mul<float>::apply_singleton_simd8(std::vector<Tensor<float>*>& in) {
  using T=float;
  const Tensor<T>& B = *in[1];
  const __m256 value = _mm256_set1_ps(B[0]);
  for (int k = 0; k < out_.size(); k += 8) {
    float* aptr = out_.data() + k;
    __m256 a = _mm256_load_ps(aptr);
    __m256 v = _mm256_mul_ps(a, value);
    _mm256_store_ps(aptr, v);
  }
  return true;
}





template <typename T>
bool Mul<T>::apply_singleton_simd8(std::vector<Tensor<T>*>& in) {
  std::cerr << "[ERROR] should not be called apply_singleton_simd8" << std::endl;
  exit(-1);
  return false;
}
#endif

// data in in[0]
// bias in in[1]
// assume data shape [N,W,H,D]
// assume bias shape [D]
template <typename T>
bool Mul<T>::init(const std::vector<Tensor<T>*>& in) {
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in.size() != 2) return false;

  // cases:
  // same dim: element wise
  // if B as only one element-> bradcast to all A element
  // B has dim [n] or [1,n] and A[...,n]
  /*
  If the bias a single dimension dimension and it
  is not a singleton, the last dimension of the input
  tensor has to be equal to the bias dimension.
  */

  if (in[1]->size() == 1) {
    // ok
  } else if (in[1]->dims().size() == 1 || (in[1]->dims().size() == 2 && in[1]->dims()[0] == 1)) {
    if (in[0]->dims().back() != in[1]->dims().back()) return false;
  } else {
    if (!(in[0]->dims() == in[1]->dims())) return false;
  }
  out_.resize(in[0]->dims());
  initDone_ = true;
  return true;
}

template <typename T>
bool Mul<T>::loadInternal(std::istream& file, Version v) {
  if (v == Version::sadl01) {
    file.read((char*)&q_, sizeof(q_));
    SADL_DBG(std::cout << "  - q: " << q_ << std::endl);
  }

  return true;
}

}  // namespace layers
}  // namespace sadl
