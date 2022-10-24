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
#include <cmath>
#if __AVX2__
#include <immintrin.h>
#endif

#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class Conv2DTranspose : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version /*v*/) override;
  Dimensions   strides_;
  Dimensions   pads_;
  Dimensions   out_pads_;
  int          q_ = 0;
  // should never be used
  void conv2dtranspose(int nb_filters, int in_D,Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);
#if __AVX512F__
  template<int in_D> void conv2dtranspose_simd512(int nb_filters,  Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);
#endif
  using T2=typename ComputationType<T>::type;
  Tensor<T2> tempo_;
  DUMP_MODEL_EXT;
};

// assume data in in[0] and kernel in in[1]
// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2DTranspose<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  assert(in[0]->dims().size() == 4);
  assert(in[1]->dims().size() == 4);
  const Tensor<T> &A      = *in[0];
  const Tensor<T> &kernel = *in[1];
  out_.quantizer          = A.quantizer - q_;
  out_.border_skip        = A.border_skip;

  assert(out_.quantizer >= 0);
  assert(kernel.quantizer + q_ >= 0);

   const int nb_filters{ out_.dims()[3] }; // kernel.dims()[2] };
//   int       in_H{ A.dims()[1] };
//   int       in_W{ A.dims()[2] };
   const int in_D{ A.dims()[3] };
   //const int half_size{ kernel.dims()[0] / 2 };
   //const int top{ pads_[0] };
   //const int left{ pads_[1] };
  // int       start_h{ half_size - top };
  // int       start_w{ half_size - left };

#if __AVX512F__
   switch(in_D) {
   case 32: conv2dtranspose_simd512<32>(nb_filters,out_,A,kernel); break;
   case 64: conv2dtranspose_simd512<64>(nb_filters,out_,A,kernel); break;
   case 128: conv2dtranspose_simd512<128>(nb_filters,out_,A,kernel); break;
   case 192: conv2dtranspose_simd512<192>(nb_filters,out_,A,kernel); break;
   case 256: conv2dtranspose_simd512<256>(nb_filters,out_,A,kernel); break;
   default:
       conv2dtranspose(nb_filters,in_D,out_,A,kernel);
   }
#else
  conv2dtranspose(nb_filters,in_D,out_,A,kernel);
#endif
  return true;
}



// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2DTranspose<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - input conv2dtranspose: " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in[0]->dims().size() != 4)
    return false;
  if (in[1]->dims().size() != 4)
    return false;
  if (in[1]->dims()[0] != in[1]->dims()[1])
    return false;
  if ((in[1]->dims()[0]) % 2 == 0)
    return false;

  // The spatial dimensions of a convolutional kernel must be 3 or 5
  if (
          (in[1]->dims()[0] !=3 )
          && (in[1]->dims()[0] !=5 ))
    return false;

  if (! ((in[1]->dims()[0] ==3 && pads_[1] == 1) || (in[1]->dims()[0] == 5 && pads_[1] == 2)) ) {
       return false;
  }
  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];
  dim[1] = (int) ceil(in[0]->dims()[1] * (float) strides_[1]);
  dim[2] = (int) ceil(in[0]->dims()[2] * (float) strides_[2]);
  dim[3] = in[1]->dims()[2];
  out_.resize(dim);
  SADL_DBG(std::cout << "  - output Conv2DTranspose: " << out_.dims() << std::endl);
  // init tempo
  int half_size=in[1]->dims()[0]/2;
  dim[1]+=half_size*2;
  dim[2]+=half_size*2;
  tempo_.resize(dim);
  initDone_ = true;
  return true;
}

template<typename T> bool Conv2DTranspose<T>::loadInternal(std::istream &file, Version /*v*/)
{
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  strides_.resize(x);
  for (int k = 0; k < strides_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    strides_[k] = x;
  }
  if (strides_.size() == 2)
  {
    strides_ = Dimensions({ 1, strides_[0], strides_[1], 1 });
  }
  if (strides_.size() != 4)
  {
    std::cerr << "[ERROR] invalid strides: " << strides_.size() << std::endl;
    return false;
  }
  if (strides_[0] != 1)
  {
    std::cerr << "[ERROR] invalid strides[0]: " << strides_[0] << std::endl;
    return false;
  }
  if (strides_[3] != 1)
  {
    std::cerr << "[ERROR] invalid strides[3]: " << strides_[3] << std::endl;
    return false;
  }
  if (strides_[1] != 2 || strides_[1] != 2)
  {
    std::cerr << "[ERROR] stride not 2: to check " << strides_ << std::endl;
    return false;
  }
  SADL_DBG(std::cout << "  - strides: " << strides_ << std::endl);

  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  pads_.resize(x);
  for (int k = 0; k < pads_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    pads_[k] = x;
    if (x != 1 && x!=2) {
        std::cerr << "[ERROR] pads values not supported: " << x << std::endl;
        return false;
    }
  }
  SADL_DBG(std::cout << "  - pads: " << pads_ << std::endl);

  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  out_pads_.resize(x);
  for (int k = 0; k < out_pads_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    out_pads_[k] = x;
    if (x != 1) {
        std::cerr << "[ERROR] output pads !=1 " << x << std::endl;
        return false;
    }
  }
  SADL_DBG(std::cout << "  - out_pads: " << out_pads_ << std::endl);

  {
    file.read((char *) &q_, sizeof(q_));
    SADL_DBG(std::cout << "  - q: " << q_ << std::endl);
  }

  return true;
}

// should never be used for perf reasons
template<typename T>
void Conv2DTranspose<T>::conv2dtranspose(int nb_filters, int in_D,Tensor<T> &out_, const Tensor<T> &A,
                       const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  const  int       in_H{ A.dims()[1] };
  const  int       in_W{ A.dims()[2] };
  std::cout << "\n[WARN] debug generic version convtranspose inD=" << in_D << " outD=" << nb_filters <<
               //" s=[" << s_w << ' ' << s_h << "] "
                in_H << 'x' << in_W << " " <<
           // << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H /) * (in_W / s_w) / 1000 << " kMAC"
               std::endl;
#endif
  constexpr int im_nb = 0;
  assert(strides_[1]==2);
  assert(strides_[2]==2);
  int     half_size=kernel.dims()[0]/2;
  constexpr int sw=2;
  constexpr int sh=2;

  const int     shift = kernel.quantizer + q_;
  const int out_h=out_.dims()[1];
  const int out_w=out_.dims()[2];
  tempo_.fill(T2{});

  for (int filter = 0; filter < nb_filters; ++filter)
  {
      for (int im_i = 0; im_i < out_h; im_i += sh)
      {
          for (int im_j = 0; im_j < out_w; im_j += sw)
          {
              const int i1=im_i/sh;
              const int j1=im_j/sw;
              assert(A.in(im_nb,i1,j1,0));
              for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
              {
                  // fixed
                  for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
                  {
                      // fixed
                      const int ki = half_size + filter_i;
                      const int kj = half_size + filter_j;
                      const int ii = im_i + ki;
                      const int jj = im_j + kj;
                      T2 s{};
                      for (int filter_d = 0; filter_d < in_D; ++filter_d)
                      {
                          s+= A(im_nb, i1, j1, filter_d) * kernel(ki, kj, filter, filter_d);
                          COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
                      }
                      tempo_(im_nb,ii,jj,filter) += s;
                  }
              }
          }
      }
      for (int im_i = 0; im_i < out_h; ++im_i)
      {
          for (int im_j = 0; im_j < out_w; ++im_j)
          {
              auto x=tempo_(im_nb,im_i+half_size,im_j+half_size,filter);
              ComputationType<T>::quantize(x, shift);
              COUNTERS(x);
              SATURATE(x);
              out_(im_nb,im_i,im_j,filter)=static_cast<T>(x);
          }
      }
  }
}

#if __AVX512F__
template<>
template<int in_D>
void Conv2DTranspose<float>::conv2dtranspose_simd512(int nb_filters,  Tensor<float> &out_, const Tensor<float> &A, const Tensor<float> &kernel)
{
  constexpr int im_nb = 0;
  assert(strides_[1]==2);
  assert(strides_[2]==2);
  assert(kernel.dims()[0]==kernel.dims()[1]);
  int     half_size=kernel.dims()[0]/2;
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
  constexpr int sw=2;
  constexpr int sh=2;

  const int out_h=out_.dims()[1];
  const int out_w=out_.dims()[2];
  tempo_.fill(T2{});

  for (int filter = 0; filter < nb_filters; ++filter)
  {
      for (int im_i = 0; im_i < out_h; im_i += sh)
      {
          for (int im_j = 0; im_j < out_w; im_j += sw)
          {
              const int i1=im_i/sh;
              const int j1=im_j/sw;
              assert(A.in(im_nb,i1,j1,0));
              for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
              {
                  // fixed
                  for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
                  {
                      __m512 s = _mm512_setzero_ps();
                      const int ki = half_size + filter_i;
                      const int kj = half_size + filter_j;
                      const int ii = im_i + ki;
                      const int jj = im_j + kj;
                      const float *kptr = kernel.addr(ki, kj, filter, 0);
                      const float *aptr = A.addr(im_nb, i1, j1, 0);
                      // fixed
                      for (int filter_d = 0; filter_d < in_D; filter_d+=16)
                      {
                          const __m512 k0   = _mm512_loadu_ps(kptr+filter_d); // not always aligned
#if __FMA__
                          s = _mm512_fmadd_ps(k0, _mm512_load_ps(aptr+filter_d), s);
#else
                          const __m512 m0 = _mm512_mul_ps(k0, _mm512_load_ps(aptr+filter_d));
                          s               = _mm512_add_ps(s, m0);
#endif
                      }
                      tempo_(im_nb,ii,jj,filter) += sum16_float(s);
                  }
              }
          }
      }
      for (int im_i = 0; im_i < out_h; ++im_i)
      {
          for (int im_j = 0; im_j < out_w; ++im_j)
          {
              auto x=tempo_(im_nb,im_i+half_size,im_j+half_size,filter);
              out_(im_nb,im_i,im_j,filter)=x;
          }
      }
  }
}
#endif

#if __AVX512BW__
template<>
template<int in_D>
void Conv2DTranspose<int16_t>::conv2dtranspose_simd512(int nb_filters,  Tensor<int16_t> &out_, const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  constexpr int im_nb = 0;
  assert(strides_[1]==2);
  assert(strides_[2]==2);
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int16_t;
#endif
  assert(kernel.dims()[0]==kernel.dims()[1]);
  const int     half_size=kernel.dims()[0]/2;
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
  constexpr int sw=2;
  constexpr int sh=2;
  const int out_h=out_.dims()[1];
  const int out_w=out_.dims()[2];
  tempo_.fill(T2{});
  const int     shift = kernel.quantizer + q_;

  for (int filter = 0; filter < nb_filters; ++filter)
  {
      for (int im_i = 0; im_i < out_h; im_i += sh)
      {
          for (int im_j = 0; im_j < out_w; im_j += sw)
          {
              const int i1=im_i/sh;
              const int j1=im_j/sw;
              assert(A.in(im_nb,i1,j1,0));
              const T *aptr = A.addr(im_nb, i1, j1, 0);
              for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
              {
                  // fixed
                  for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
                  {
                      __m512i s = _mm512_setzero_si512();
                      const int ki = half_size + filter_i;
                      const int kj = half_size + filter_j;
                      const int ii = im_i + ki;
                      const int jj = im_j + kj;
                      const T *kptr = kernel.addr(ki, kj, filter, 0);
                      // fixed
                      for (int filter_d = 0; filter_d < in_D; filter_d+=32)
                      {
                          const __m512i k0   = _mm512_loadu_si512(kptr+filter_d); // not always aligned
                          const __m512i v0   = _mm512_load_si512(aptr+filter_d);
                          const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
                          s                  = _mm512_add_epi32(s, mad0);
                      }
                      tempo_(im_nb,ii,jj,filter) += _mm512_reduce_add_epi32(s);
                  }
              }
          }
      }
      for (int im_i = 0; im_i < out_h; ++im_i)
      {
          for (int im_j = 0; im_j < out_w; ++im_j)
          {
              auto z=tempo_(im_nb,im_i+half_size,im_j+half_size,filter)>>shift;
              SATURATE(z);
              out_(im_nb,im_i,im_j,filter)=z;
          }
      }
  }
}
#endif

#if __AVX512BW__ || __AVX512F__
template<typename T>
template<int in_D> void Conv2DTranspose<T>::conv2dtranspose_simd512(int nb_filters,  Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel) {
    std::cerr<<"TODO "<<std::endl;
    exit(-1);
}
#endif

}   // namespace layers
}   // namespace sadl
