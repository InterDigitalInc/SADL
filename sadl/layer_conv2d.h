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
template<typename T> class Conv2D : public Layer<T>
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
  int          q_ = 0;

  template<int s_h, int s_w> bool apply_s(const Tensor<T> &A, const Tensor<T> &kernel);

  // should never be used
  void conv2d(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, int s_h, int s_w, Tensor<T> &out_, const Tensor<T> &A,
              const Tensor<T> &kernel);

  // 1x1
  template<int s_h, int s_w>
  void conv2d_1x1_s_dispatch(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                             const Tensor<T> &kernel);

  template<int in_D, int s_h, int s_w>
  void conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w>
  void conv2d_1x1_s(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

  // 3x3
  template<int s_h, int s_w>
  void conv2d_3x3_s_peel(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w>
  void conv2d_3x3_s_core_dispatch(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                  const Tensor<T> &kernel);

  template<int s_h, int s_w>
  void conv2d_3x3_s_core(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

  template<int in_D, int s_h, int s_w>
  void conv2d_3x3_s_d_core(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

#if __AVX2__
  template<int in_D, int s_h, int s_w>
  void simd8_conv2d_1x1_s_d(int /*nb_filters*/, int /*in_H*/, int /*in_W*/, int /*start_h*/, int /*start_w*/, Tensor<T> & /*out_*/, const Tensor<T> & /*A*/,
                            const Tensor<T> & /*kernel*/)
  {
    assert(false);
    exit(-1);
  }
  template<int in_D, int s_h, int s_w>
  void simd16_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd8_conv2d_1x1_s_d<in_D, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel);
  }
  template<int in_D, int s_h, int s_w>
  void simd32_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd16_conv2d_1x1_s_d<in_D, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel);
  }

  template<int in_D, int s_h, int s_w>
  void simd8_conv2d_3x3_s_d(int /*nb_filters*/, int /*in_H*/, int /*in_W*/, int /*start_h*/, int /*start_w*/, Tensor<T> & /*out_*/, const Tensor<T> & /*A*/,
                            const Tensor<T> & /*kernel*/)
  {
    assert(false);
    exit(-1);
  }
  template<int in_D, int s_h, int s_w>
  void simd16_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd8_conv2d_3x3_s_d<in_D, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel);
  }
  template<int in_D, int s_h, int s_w>
  void simd32_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd16_conv2d_3x3_s_d<in_D, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel);
  }
#endif
  DUMP_MODEL_EXT;
};

// assume data in in[0] and kernel in in[1]
// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2D<T>::apply(std::vector<Tensor<T> *> &in)
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
  if (strides_[1] == 1 && strides_[2] == 1)
  {
    return apply_s<1, 1>(A, kernel);
  }
  else if (strides_[1] == 1 && strides_[2] == 2)
  {
    return apply_s<1, 2>(A, kernel);
  }
  else if ((strides_[1] == 2 && strides_[2] == 1))
  {
    return apply_s<2, 1>(A, kernel);
  }
  else if (strides_[1] == 2 && strides_[2] == 2)
  {
    return apply_s<2, 2>(A, kernel);
  }
  else
  {
    std::cerr << "[ERROR] stride = (" << strides_[1] << ", " << strides_[2] << ")" << std::endl;
    assert(false);
    exit(-1);
  }
  return false;
}

template<typename T> template<int s_h, int s_w> bool Conv2D<T>::apply_s(const Tensor<T> &A, const Tensor<T> &kernel)
{
  int       in_H{ A.dims()[1] };
  int       in_W{ A.dims()[2] };
  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };
  const int half_size{ kernel.dims()[0] / 2 };
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  int       start_h{ half_size - top };
  int       start_w{ half_size - left };
  assert(in_H > 1);
  assert(in_W > 1);

  if (half_size == 0)
  {
    conv2d_1x1_s_dispatch<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel);
  }
  else if (half_size == 1)
  {
    if (!Tensor<T>::skip_border)
    {
      conv2d_3x3_s_peel<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel);
    }
    else
    {   // skip border
      if (s_h == 1 && s_w == 1)
      {
        start_h += out_.border_skip;
        start_w += out_.border_skip;
        in_H -= out_.border_skip;
        in_W -= out_.border_skip;
        out_.border_skip++;
      }
    }
    conv2d_3x3_s_core_dispatch<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel);
  }
  else
  {
    assert(false);
    // conv2d()
    return false;
  }
  return true;
}

// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2D<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - input conv2d: " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in[0]->dims().size() != 4)
    return false;
  if (in[1]->dims().size() != 4)
    return false;
  if (in[1]->dims()[0] != in[1]->dims()[1])
    return false;
  if ((in[1]->dims()[0]) % 2 == 0)
    return false;

  // The spatial dimensions of a convolutional kernel must be either
  // 1x1 or 3x3.
  if (in[1]->dims()[0] / 2 > 1)
    return false;
  if (in[0]->dims()[0] != 1)
    return false;
  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];
  dim[1] = (int) ceil(in[0]->dims()[1] / (float) strides_[1]);
  dim[2] = (int) ceil(in[0]->dims()[2] / (float) strides_[2]);
  dim[3] = in[1]->dims()[2];
  out_.resize(dim);
  SADL_DBG(std::cout << "  - output Conv2D: " << out_.dims() << std::endl);
  initDone_ = true;
  return true;
}

template<typename T> bool Conv2D<T>::loadInternal(std::istream &file, Version /*v*/)
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
  if (strides_[1] != 1 && strides_[1] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << strides_ << std::endl;
    return false;
  }
  if (strides_[2] != 1 && strides_[2] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << strides_ << std::endl;
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
  }
  SADL_DBG(std::cout << "  - pads: " << pads_ << std::endl);
  {
    file.read((char *) &q_, sizeof(q_));
    SADL_DBG(std::cout << "  - q: " << q_ << std::endl);
  }

  return true;
}

// should never be used for perf reasons
template<typename T>
void Conv2D<T>::conv2d(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, int s_h, int s_w, Tensor<T> &out_, const Tensor<T> &A,
                       const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] debug generic version conv inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb = 0;
  const int     half_size{ kernel.dims()[0] / 2 };
  const int     shift = kernel.quantizer + q_;
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
    {
      for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {
          // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size + filter_i;
              int kj = half_size + filter_j;
              if (A.in(im_nb, ii, jj, filter_d))
              {
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
              }
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1x1
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_1x1_s_dispatch(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                      const Tensor<T> &kernel)
{
#if __AVX2__
#define CONV_MOD8 simd8_conv2d_1x1_s_d
#define CONV_MOD16 simd16_conv2d_1x1_s_d
#define CONV_MOD32 simd32_conv2d_1x1_s_d
#else
#define CONV_MOD8 conv2d_1x1_s_d
#define CONV_MOD16 conv2d_1x1_s_d
#define CONV_MOD32 conv2d_1x1_s_d
#endif
  switch (in_D)
  {
  case 1: conv2d_1x1_s_d<1, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 2: conv2d_1x1_s_d<2, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 4: conv2d_1x1_s_d<4, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 8: CONV_MOD8<8, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 16: CONV_MOD16<16, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 24: CONV_MOD8<24, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 32: CONV_MOD32<32, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 48: CONV_MOD16<48, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 64: CONV_MOD32<64, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 72:
    // better do 64 and than 8
    CONV_MOD8<72, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel);
    break;
  case 96: CONV_MOD32<96, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 128: CONV_MOD32<128, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 384: CONV_MOD32<384, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 480: CONV_MOD32<480, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  default: conv2d_1x1_s<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel); break;
  }
#undef CONV_MOD8
#undef CONV_MOD16
#undef CONV_MOD32
}

template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_1x1_s(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                             const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv1x1 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D; ++filter_d)
        {
          {
            x += (typename ComputationType<T>::type) A(im_nb, im_i, im_j, filter_d) * kernel(0, 0, filter_nb, filter_d);
            COUNTERS_MAC(kernel(0, 0, filter_nb, filter_d));
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_w, im_j / s_h, filter_nb) = static_cast<T>(x);
      }
    }
  }
}

template<typename T>
template<int in_D, int s_h, int s_w>
void Conv2D<T>::conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv 1x1 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif

  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
  for (int im_i = start_h; im_i < in_H; im_i += s_h)
  {
    for (int im_j = start_w; im_j < in_W; im_j += s_w)
    {
      for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D; ++filter_d)
        {
          x += (typename ComputationType<T>::type) A(im_nb, im_i, im_j, filter_d) * kernel(0, 0, filter_nb, filter_d);
          COUNTERS_MAC(kernel(0, 0, filter_nb, filter_d));
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
      }
    }
  }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 3x3
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_3x3_s_peel(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                  const Tensor<T> &kernel)
{
  constexpr int im_nb      = 0;
  const int     shift      = kernel.quantizer + q_;
  constexpr int ihalf_size = 1;
  for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
  {
    // corners
    {
      int  im_i;
      int  im_j;
      auto loop_with_cond = [&, filter_nb, shift](int i0, int i1, int j0, int j1)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = i0; filter_i <= i1; ++filter_i)
        {
          for (int filter_j = j0; filter_j <= j1; ++filter_j)
          {
            for (int filter_d = 0; filter_d < in_D; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = ihalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter_nb, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
      };

      im_j = start_w;
      if (im_j < in_W)
      {   // left side
        im_i = start_h;
        if (im_i < in_H)
        {   // top left corner
          loop_with_cond(-start_h, ihalf_size, -start_w, ihalf_size);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom left corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -start_w, ihalf_size);
        }
      }

      im_j            = ((in_W - ihalf_size - start_w) / s_w) * s_w + start_w;
      const int end_j = (im_j + 1 < in_W) ? 1 : 0;
      if (im_j > 0 && im_j < in_W && im_j != start_w)
      {   // rihgt side
        im_i = start_h;
        if (im_i < in_H)
        {   // top right corner
          loop_with_cond(-start_h, ihalf_size, -ihalf_size, end_j);
        }

        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom right corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -ihalf_size, end_j);
        }
      }
    }

    // vertical borders
    {
      for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
      {
        int im_j = start_w;   // can be only 0 or 1
        if (im_j < in_W)
        {   // left side
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -start_w; filter_j <= ihalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = ihalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }

        im_j = ((in_W - ihalf_size - start_w) / s_w) * s_w + start_w;
        if (im_j > 0 && im_j < in_W && im_j != start_w)
        {   // rihgt side
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_j + 1) < in_W ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -ihalf_size; filter_j <= end_filter; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = ihalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
    {
      // horizontal borders
      for (int im_j = s_w + start_w; im_j < in_W - ihalf_size; im_j += s_w)
      {
        int im_i = start_h;   // 0 or 1 -> adapt filter start
        if (im_i < in_H)
        {   // top line
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -start_h; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -ihalf_size; filter_j <= ihalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = ihalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }

          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom line
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_i + 1) < in_H ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= end_filter; ++filter_i)
          {
            for (int filter_j = -ihalf_size; filter_j <= ihalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = ihalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
  }   // filter_nb
}

template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_3x3_s_core_dispatch(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                           const Tensor<T> &kernel)
{
#if __AVX2__
#define CONV_MOD8 simd8_conv2d_3x3_s_d
#define CONV_MOD16 simd16_conv2d_3x3_s_d
#define CONV_MOD32 simd32_conv2d_3x3_s_d
#else
#define CONV_MOD8 conv2d_3x3_s_d_core
#define CONV_MOD16 conv2d_3x3_s_d_core
#define CONV_MOD32 conv2d_3x3_s_d_core
#endif

  switch (in_D)
  {
  case 1: conv2d_3x3_s_d_core<1, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 2: conv2d_3x3_s_d_core<2, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 4: conv2d_3x3_s_d_core<4, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 8: CONV_MOD8<8, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 16: CONV_MOD16<16, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 24: CONV_MOD8<24, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 32: CONV_MOD32<32, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 48: CONV_MOD16<48, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 64: CONV_MOD32<64, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 72:
    CONV_MOD8<72, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A,
                            kernel);   // better do 64 and than 8
    break;
  case 96: CONV_MOD32<96, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  case 128: CONV_MOD32<128, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  default: conv2d_3x3_s_core<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel); break;
  }
#undef CONV_MOD8
#undef CONV_MOD16
#undef CONV_MOD32
}

template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_3x3_s_core(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                  const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size + filter_i;
              int kj = half_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

template<typename T>
template<int in_D, int s_h, int s_w>
void Conv2D<T>::conv2d_3x3_s_d_core(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D; ++filter_d)
        {
          for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
          {   // fixed
            for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
            {   // fixed
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size + filter_i;
              int kj = half_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

#if __AVX2__
static inline float sum8_float(__m256 x)
{
  const __m128 hiQuad  = _mm256_extractf128_ps(x, 1);
  const __m128 loQuad  = _mm256_castps256_ps128(x);
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  const __m128 loDual  = sumQuad;
  const __m128 hiDual  = _mm_movehl_ps(sumQuad, sumQuad);
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  const __m128 lo      = sumDual;
  const __m128 hi      = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  const __m128 sum     = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}
// int32
static inline typename ComputationType<int32_t>::type sum64_int32(__m256i x)
{   //  to optiz
  return _mm256_extract_epi64(x, 0) + _mm256_extract_epi64(x, 1) + _mm256_extract_epi64(x, 2) + _mm256_extract_epi64(x, 3);
}
static inline typename ComputationType<int16_t>::type hsum_epi32_avx(__m128i x)
{
  __m128i hi64 = _mm_unpackhi_epi64(x, x);   // 3-operand non-destructive AVX lets us save a
                                             // byte without needing a movdqa
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));   // Swap the low two elements
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);   // movd
}

static inline typename ComputationType<int16_t>::type sum32_int16(__m256i x)
{
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  return hsum_epi32_avx(sum128);
}

static inline typename ComputationType<int16_t>::type sum32_int16(__m128i s)
{
    __m128i hi64 =
      _mm_unpackhi_epi64(s, s);   // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
    __m128i sum64 = _mm_add_epi32(hi64, s);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));   // Swap the low two elements
    __m128i sum32 = _mm_add_epi32(sum64, hi32);

    typename ComputationType<int16_t>::type z = _mm_cvtsi128_si32(sum32);
    return z;
}
#if __AVX512F__
static inline float sum16_float(const __m512 vec_in)
{
  const __m128 vec_low_quad_0  = _mm512_extractf32x4_ps(vec_in, 0);
  const __m128 vec_high_quad_0 = _mm512_extractf32x4_ps(vec_in, 1);
  const __m128 vec_sum_quad_0  = _mm_add_ps(vec_low_quad_0, vec_high_quad_0);
  const __m128 vec_low_quad_1  = _mm512_extractf32x4_ps(vec_in, 2);
  const __m128 vec_high_quad_1 = _mm512_extractf32x4_ps(vec_in, 3);
  const __m128 vec_sum_quad_1  = _mm_add_ps(vec_low_quad_1, vec_high_quad_1);
  const __m128 vec_sum_quad    = _mm_add_ps(vec_sum_quad_0, vec_sum_quad_1);
  const __m128 vec_moved       = _mm_movehl_ps(vec_sum_quad, vec_sum_quad);
  const __m128 vec_sum_dual    = _mm_add_ps(vec_sum_quad, vec_moved);
  const __m128 vec_shuffled    = _mm_shuffle_ps(vec_sum_dual, vec_sum_dual, 0x1);
  const __m128 vec_sum_single  = _mm_add_ss(vec_sum_dual, vec_shuffled);
  return _mm_cvtss_f32(vec_sum_single);
}
#endif

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1x1
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<>
template<int in_D, int s_h, int s_w>
inline void Conv2D<float>::simd8_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<float> &out_, const Tensor<float> &A,
                                                const Tensor<float> &kernel)
{
  static_assert(in_D % 8 == 0, "Should be used with mod8 filters.");
#if DEBUG_SIMD && __AVX512F__
  if (in_D >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 1x1 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb = 0;
  for (int im_i = start_h ; im_i < in_H ; im_i += s_h)
  {
    for (int im_j = start_w ; im_j < in_W ; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256 s = _mm256_setzero_ps();
        for (int filter_d = 0; filter_d < in_D; filter_d += 8)
        {
          const float *kptr = kernel.addr(0, 0, filter, filter_d);
          const float *aptr = A.addr(im_nb, im_i, im_j, filter_d);
          const __m256 k0   = _mm256_load_ps(kptr);
#if __FMA__
          s = _mm256_fmadd_ps(k0, _mm256_load_ps(aptr), s);
#else
          const __m256 m0 = _mm256_mul_ps(k0, _mm256_load_ps(aptr));
          s               = _mm256_add_ps(s, m0);
          // s + m0; // s = _mm256_hadd_ps(s, m0);
#endif
        }
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = sum8_float(s);
      }
    }
  }
}

#if __AVX512F__
template<>
template<int in_D, int s_h, int s_w>
inline void Conv2D<float>::simd16_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<float> &out_, const Tensor<float> &A,
                                                 const Tensor<float> &kernel)
{
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
  constexpr int im_nb = 0;
  for (int im_i = start_h ; im_i < in_H ; im_i += s_h)
  {
    for (int im_j = start_w ; im_j < in_W ; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512 s = _mm512_setzero_ps();
        for (int filter_d = 0; filter_d < in_D; filter_d += 16)
        {
          const float *kptr = kernel.addr(0, 0, filter, filter_d);
          const float *aptr = A.addr(im_nb, im_i, im_j, filter_d);
          const __m512 k0   = _mm512_load_ps(kptr);
#if __FMA__
          s = _mm512_fmadd_ps(k0, _mm512_load_ps(aptr), s);
#else
          const __m512 m0 = _mm512_mul_ps(k0, _mm512_load_ps(aptr));
          s               = _mm512_add_ps(s, m0);
#endif
        }
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = sum16_float(s);
      }
    }
  }
}
#endif

// int16
template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd8_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                           const Tensor<int16_t> &kernel)
{   // should be sse42
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int16_t;
#endif
  static_assert(in_D % 8 == 0, "Should be used with mod16 filters.");
#if DEBUG_SIMD && __AVX2__
  if (in_D >= 8)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
  for (int im_i = start_h ; im_i < in_H ; im_i += s_h)
  {
    for (int im_j = start_w ; im_j < in_W ; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m128i s = _mm_setzero_si128();
        for (int filter_d = 0; filter_d < in_D; filter_d += 8)
        {
          const __m128i *kptr = (const __m128i *) kernel.addr(0, 0, filter, filter_d);
          const __m128i  k0   = _mm_load_si128(kptr);   // or loadu ?
          const __m128i *aptr = (const __m128i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m128i  v0   = _mm_load_si128(aptr);

          const __m128i mad0 = _mm_madd_epi16(k0, v0);   // res in si32
          s                  = _mm_add_epi32(s, mad0);
        }
        typename ComputationType<int32_t>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd16_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                            const Tensor<int16_t> &kernel)
{
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int16_t;
#endif
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
#if DEBUG_SIMD && __AVX512BW__
  if (in_D >= 32)
  {
    std::cout << "\n[WARN] suboptimal SIMD16 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
  for (int im_i = start_h ; im_i < in_H ; im_i += s_h)
  {
    for (int im_j = start_w ; im_j < in_W ; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256i s = _mm256_setzero_si256();
        for (int filter_d = 0; filter_d < in_D; filter_d += 16)
        {
          const __m256i *kptr = (const __m256i *) kernel.addr(0, 0, filter, filter_d);
          const __m256i  k0   = _mm256_load_si256(kptr);   // or loadu ?
          const __m256i *aptr = (const __m256i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m256i  v0   = _mm256_load_si256(aptr);

          const __m256i mad0 = _mm256_madd_epi16(k0, v0);   // res in si32
          s                  = _mm256_add_epi32(s, mad0);
        }
        typename ComputationType<int32_t>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

#if __AVX512BW__
template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd32_conv2d_1x1_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                            const Tensor<int16_t> &kernel)
{
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
  using T             = int16_t;
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
  for (int im_i = start_h ; im_i < in_H ; im_i += s_h)
  {
    for (int im_j = start_w ; im_j < in_W ; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512i s = _mm512_setzero_si512();
        for (int filter_d = 0; filter_d < in_D; filter_d += 32)
        {
          const __m512i *kptr = (const __m512i *) kernel.addr(0, 0, filter, filter_d);
          const __m512i  k0   = _mm512_load_si512(kptr);
          const __m512i *aptr = (const __m512i *) A.addr(im_nb, im_i, im_j, filter_d);
          const __m512i  v0   = _mm512_load_si512(aptr);

          const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
          s                  = _mm512_add_epi32(s, mad0);
        }
        typename ComputationType<int32_t>::type z = (_mm512_reduce_add_epi32(s) >> shift);
        COUNTERS(z);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = z;
      }
    }
  }
}
#endif

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 3x3
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<>
template<int in_D, int s_h, int s_w>
inline void Conv2D<float>::simd8_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<float> &out_, const Tensor<float> &A,
                                                const Tensor<float> &kernel)
{
  static_assert(in_D % 8 == 0, "Should be used with mod8 filters.");
#if DEBUG_SIMD && __AVX512F__
  if (in_D >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256 s = _mm256_setzero_ps();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            const int ii = im_i + filter_i;
            const int jj = im_j + filter_j;
            const int ki = half_size + filter_i;
            const int kj = half_size + filter_j;

            for (int filter_d = 0; filter_d < in_D; filter_d += 8)
            {
              const float *kptr = kernel.addr(ki, kj, filter, filter_d);
              const __m256 k0   = _mm256_load_ps(kptr);
              const float *aptr = A.addr(im_nb, ii, jj, filter_d);
#if __FMA__
              s = _mm256_fmadd_ps(k0, _mm256_load_ps(aptr), s);
#else
              const __m256 m0 = _mm256_mul_ps(k0, _mm256_load_ps(aptr));
              s               = _mm256_add_ps(s, m0);
              ;   // s + m0; // s = _mm256_hadd_ps(s, m0);
#endif
            }
          }
        }
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = sum8_float(s);
      }
    }
  }
}

#if __AVX512F__
template<>
template<int in_D, int s_h, int s_w>
inline void Conv2D<float>::simd16_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<float> &out_, const Tensor<float> &A,
                                                 const Tensor<float> &kernel)
{
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");

  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512 s = _mm512_setzero_ps();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            const int ii = im_i + filter_i;
            const int jj = im_j + filter_j;
            const int ki = half_size + filter_i;
            const int kj = half_size + filter_j;

            for (int filter_d = 0; filter_d < in_D; filter_d += 16)
            {
              const float *kptr = kernel.addr(ki, kj, filter, filter_d);
              const __m512 k0   = _mm512_load_ps(kptr);
              const float *aptr = A.addr(im_nb, ii, jj, filter_d);
#if __FMA__
              s = _mm512_fmadd_ps(k0, _mm512_load_ps(aptr), s);
#else
              const __m512 m0 = _mm512_mul_ps(k0, _mm512_load_ps(aptr));
              s               = _mm512_add_ps(s, m0);
#endif
            }
          }
        }
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = sum16_float(s);
      }
    }
  }
}
#endif

template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int32_t>::simd8_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int32_t> &out_, const Tensor<int32_t> &A,
                                           const Tensor<int32_t> &kernel)
{
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int32_t;
#endif
  static_assert(in_D % 8 == 0, "Should be used with mod8 filters.");
#if DEBUG_SIMD && __AVX512F__
  if (in_D >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256i s = _mm256_setzero_si256();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 8)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = half_size + filter_i;
              const int      kj   = half_size + filter_j;
              const __m256i *kptr = (const __m256i *) kernel.addr(ki, kj, filter, filter_d);
              const __m256i  k0   = _mm256_load_si256(kptr);
              const __m256i *aptr = (const __m256i *) A.addr(im_nb, ii, jj, filter_d);
              const __m256i  v0   = _mm256_load_si256(aptr);
              const __m256i  m0   = _mm256_mul_epi32(k0, v0);

              const __m256i k1 = _mm256_shuffle_epi32(k0, 0b11110101);
              const __m256i v1 = _mm256_shuffle_epi32(v0, 0b11110101);

              s = _mm256_add_epi64(s, m0);

              const __m256i m1 = _mm256_mul_epi32(k1, v1);
              s                = _mm256_add_epi64(s, m1);
            }
          }
        }
        typename ComputationType<int32_t>::type z = (sum64_int32(s) >> shift);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int32_t>(z);
      }
    }
  }
}

// int16
template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd8_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                           const Tensor<int16_t> &kernel)
{
  // should be sse42
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int16_t;
#endif
  static_assert(in_D % 8 == 0, "Should be used with mod8 filters.");
#if DEBUG_SIMD
  if (in_D >= 8)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m128i s = _mm_setzero_si128();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 8)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = half_size + filter_i;
              const int      kj   = half_size + filter_j;
              const __m128i *kptr = (const __m128i *) kernel.addr(ki, kj, filter, filter_d);
              const __m128i  k0   = _mm_load_si128(kptr);   // or loadu ?
              const __m128i *aptr = (const __m128i *) A.addr(im_nb, ii, jj, filter_d);
              const __m128i  v0   = _mm_load_si128(aptr);

              const __m128i mad0 = _mm_madd_epi16(k0, v0);   // res in si32
              s                  = _mm_add_epi32(s, mad0);
            }
          }
        }
        typename ComputationType<int32_t>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd16_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                            const Tensor<int16_t> &kernel)
{
#if DEBUG_COUNTERS || SATURATE_RESULT
  using T = int16_t;
#endif
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
#if DEBUG_SIMD && __AVX512BW__
  if (in_D >= 32)
  {
    std::cout << "\n[WARN] suboptimal SIMD16 version conv 3x3 inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x'
              << in_W << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
  }
#endif
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256i s = _mm256_setzero_si256();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 16)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = half_size + filter_i;
              const int      kj   = half_size + filter_j;
              const __m256i *kptr = (const __m256i *) kernel.addr(ki, kj, filter, filter_d);
              const __m256i  k0   = _mm256_load_si256(kptr);   // or loadu ?
              const __m256i *aptr = (const __m256i *) A.addr(im_nb, ii, jj, filter_d);
              const __m256i  v0   = _mm256_load_si256(aptr);

              const __m256i mad0 = _mm256_madd_epi16(k0, v0);   // res in si32
              s                  = _mm256_add_epi32(s, mad0);
            }
          }
        }
        typename ComputationType<int32_t>::type z = (sum32_int16(s) >> shift);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

#if __AVX512BW__
template<>
template<int in_D, int s_h, int s_w>
void Conv2D<int16_t>::simd32_conv2d_3x3_s_d(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<int16_t> &out_, const Tensor<int16_t> &A,
                                            const Tensor<int16_t> &kernel)
{
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
  using T                 = int16_t;
  constexpr int im_nb     = 0;
  constexpr int half_size = 1;
  const int     shift     = kernel.quantizer + q_;
  for (int im_i = start_h + s_h; im_i < in_H - s_h; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - s_w; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512i s = _mm512_setzero_si512();
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {   // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 32)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = half_size + filter_i;
              const int      kj   = half_size + filter_j;
              const __m512i *kptr = (const __m512i *) kernel.addr(ki, kj, filter, filter_d);
              const __m512i  k0   = _mm512_load_si512(kptr);
              const __m512i *aptr = (const __m512i *) A.addr(im_nb, ii, jj, filter_d);
              const __m512i  v0   = _mm512_load_si512(aptr);

              const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
              s                  = _mm512_add_epi32(s, mad0);
            }
          }
        }
        typename ComputationType<int32_t>::type z = (_mm512_reduce_add_epi32(s) >> shift);
        COUNTERS(z);
        SATURATE(z);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = z;
      }
    }
  }
}
#endif

#endif   // SIMD

}   // namespace layers
}   // namespace sadl
