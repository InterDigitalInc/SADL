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
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class MaxPool : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  Dimensions   kernel_;
  Dimensions   strides_;
  Dimensions   pads_;
  DUMP_MODEL_EXT;
};

// assume data in in[0]
// data [batch, in_height, in_width, in_channels]
// kernel [1, kernel_height, kernel_width, 1]
// stride [1, stride_height, stride_width, 1]
template<typename T> bool MaxPool<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  assert(in[0]->dims().size() == 4);

  const Tensor<T> &A            = *in[0];
  const int        N            = out_.dims()[0];
  const int        H            = out_.dims()[1];
  const int        W            = out_.dims()[2];
  const int        D            = out_.dims()[3];
  const int        offset_end   = kernel_[1] / 2;
  const int        offset_start = kernel_[1] - 1 - offset_end;
  const int        step         = strides_[1];
  const int        in_H         = in[0]->dims()[1];

  // currently adhoc start
  int start = 0;
  if (step == 1)
  {
    start = 0;
  }
  else if (step == 2)
  {
    //  if (in_H % 2 == 0)
    //    start = 1;
    //  else
    start = 0;
  }
  else if (step == 3)
  {
    if (in_H % 2 == 0)
      start = 0;
    else
      start = 1;
  }
  else
  {
    std::cerr << "[ERROR] to do" << std::endl;
    assert(false);
    exit(-1);
  }

  out_.quantizer   = in[0]->quantizer;     // adapt output width to bias
  out_.border_skip = in[0]->border_skip;   // to check

  for (int im_nb = 0; im_nb < N; ++im_nb)
  {
    // loop on out
    for (int im_i = 0; im_i < H; ++im_i)
    {
      for (int im_j = 0; im_j < W; ++im_j)
      {
        for (int im_d = 0; im_d < D; ++im_d)
        {
          T xx = -std::numeric_limits<T>::max();
          for (int filter_i = -offset_start; filter_i <= offset_end; ++filter_i)
          {
            for (int filter_j = -offset_start; filter_j <= offset_end; ++filter_j)
            {
              int ii = im_i * step + filter_i + start;
              int jj = im_j * step + filter_j + start;
              if (A.in(im_nb, ii, jj, im_d))
              {
                T x = A(im_nb, ii, jj, im_d);
                if (xx < x)
                  xx = x;
              }
            }
          }
          out_(im_nb, im_i, im_j, im_d) = xx;
        }
      }
    }
  }

  return true;
}

// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool MaxPool<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  SADL_DBG(std::cout << "  - input maxpool: " << in[0]->dims() << std::endl);
  SADL_DBG(std::cout << "  - stride: " << strides_ << std::endl);
  SADL_DBG(std::cout << "  - kernel: " << kernel_ << std::endl);
  if (in[0]->dims().size() != 4)
    return false;

  // convervative check
  if (kernel_.size() != 4)
    return false;
  // no pooling on batch and depth
  if (kernel_[0] != 1 || kernel_[3] != 1)
    return false;

  // no stride on batch and depth
  if (strides_.size() != 4)
    return false;
  if (strides_[0] != 1 || strides_[3] != 1)
    return false;

  // square filter
  if (kernel_[1] != kernel_[2])
    return false;
  // square stride
  if (strides_[1] != strides_[2])
    return false;

  Dimensions dim;

  dim.resize(4);
  dim[0]                   = in[0]->dims()[0];
  constexpr int dilatation = 1;
  dim[1]                   = (int) floor((in[0]->dims()[1] + pads_[0] + pads_[2] - ((kernel_[1] - 1) * dilatation + 1)) / (float) strides_[1] + 1);
  dim[2]                   = (int) floor((in[0]->dims()[2] + pads_[1] + pads_[3] - ((kernel_[2] - 1) * dilatation + 1)) / (float) strides_[2] + 1);
  dim[3]                   = in[0]->dims()[3];

  out_.resize(dim);
  SADL_DBG(std::cout << "  - output: " << out_.dims() << std::endl);

  initDone_ = true;
  return true;
}

template<typename T> bool MaxPool<T>::loadInternal(std::istream &file, Version v)
{
  // load values
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions strides: " << x << std::endl;
    return false;
  }
  strides_.resize(x);
  for (int k = 0; k < strides_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    strides_[k] = x;
  }
  SADL_DBG(std::cout << "  - strides: " << strides_ << std::endl);
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
  if (strides_[1] != strides_[2])
  {
    std::cerr << "[ERROR] invalid stride H Vs: " << strides_ << std::endl;
    return false;
  }

  x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions kernel: " << x << std::endl;
    return false;
  }
  kernel_.resize(x);
  for (int k = 0; k < kernel_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    kernel_[k] = x;
  }
  SADL_DBG(std::cout << "  - kernel: " << kernel_ << std::endl);
  if (kernel_.size() != 4)
  {
    std::cerr << "[ERROR] invalid kernel: " << kernel_.size() << std::endl;
    return false;
  }
  if (kernel_[0] != 1)
  {
    std::cerr << "[ERROR] invalid kernel[0]: " << kernel_[0] << std::endl;
    return false;
  }
  if (kernel_[3] != 1)
  {
    std::cerr << "[ERROR] invalid kernel[3]: " << kernel_[3] << std::endl;
    return false;
  }
  if (kernel_[1] != kernel_[2])
  {
    std::cerr << "[ERROR] invalid kernel H V: " << kernel_ << std::endl;
    return false;
  }
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
  return true;
}

}   // namespace layers
}   // namespace sadl
