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
#include "layer_add.h"

namespace sadl
{
namespace layers
{
template<typename T> class BiasAdd : public Add<T>
{
public:
  using Add<T>::Add;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
};

template<typename T> bool BiasAdd<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }

  const int shift = in[0]->quantizer - in[1]->quantizer;
  swap(*in[0], out_);
  // adapt output width to second input (which are the bias) in order to be able to rescale as desired the input
  out_.quantizer = in[1]->quantizer;

  if (shift < 0)
  {
    if (in[0]->dims() == in[1]->dims())
    {
      for (auto it0 = out_.begin(), it1 = in[1]->begin(); it0 != out_.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it0;
        ComputationType<T>::shift_left(z, -shift);
//        ComputationType<T>::quantize(z, shift);
        z += *it1;
        COUNTERS(z);
        SATURATE(z);
        *it0 = z;
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // ie in[0]->dims().size() == 1? happen if in[1] is a Const
        const Tensor<T> &B     = *in[1];
        const T          value = B[0];
        for (auto &x: out_)
        {
          typename ComputationType<T>::type z = x;
          ComputationType<T>::shift_left(z, -shift);
          z += value;
          COUNTERS(z);
          SATURATE(z);
          x = z;
        }
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = out_(n, i);
            ComputationType<T>::shift_left(z, -shift);
            z += B[i];
            COUNTERS(z);
            SATURATE(z);
            out_(n, i) = z;
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = out_(n, i, j);
              ComputationType<T>::shift_left(z, -shift);
              z += B[j];
              COUNTERS(z);
              SATURATE(z);
              out_(n, i, j) = z;
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
        const int        K = in[0]->dims()[3];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = out_(n, i, j, k);
                ComputationType<T>::shift_left(z, -shift);
                z += B[k];
                COUNTERS(z);
                SATURATE(z);
                out_(n, i, j, k) = z;
              }
      }
    }
  }
  else
  {
    if (in[0]->dims() == in[1]->dims())
    {
      for (auto it0 = out_.begin(), it1 = in[1]->begin(); it0 != out_.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it0;
        ComputationType<T>::quantize(z, shift);
        z += *it1;
        COUNTERS(z);
        SATURATE(z);
        *it0 = z;
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // for constant
        const Tensor<T> &B     = *in[1];
        const T          value = B[0];
        for (auto &x: out_)
        {
          typename ComputationType<T>::type z = x;
          ComputationType<T>::quantize(z, shift);
          z += value;
          COUNTERS(z);
          SATURATE(z);
          x = z;
        }
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = out_(n, i);
            ComputationType<T>::quantize(z, shift);
            z += B[i];
            COUNTERS(z);
            SATURATE(z);
            out_(n, i) = z;
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];

        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = out_(n, i, j);
              ComputationType<T>::quantize(z, shift);
              z += B[j];
              COUNTERS(z);
              SATURATE(z);
              out_(n, i, j) = z;
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];
        const int K = in[0]->dims()[3];

        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = out_(n, i, j, k);
                ComputationType<T>::quantize(z, shift);
                z += B[k];
                COUNTERS(z);
                SATURATE(z);
                out_(n, i, j, k) = z;
              }
      }
    }
  }
  return true;
}

template<typename T> bool BiasAdd<T>::init(const std::vector<Tensor<T> *> &in)
{
  // convervative check
  if (in.size() != 2)
    return false;
  if (in[1]->dims().size() != 1)
    return false;
  if (in[0]->dims()[in[0]->dims().size() - 1] != in[1]->dims()[0])
    return false;

  out_.resize(in[0]->dims());
  initDone_ = true;
  return true;
}

}   // namespace layers
}   // namespace sadl
