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
template<typename T> class Maximum : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version) override;
};

template<typename T> bool Maximum<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }
  const int shift = -(in[1]->quantizer - in[0]->quantizer);
  swap(*in[0], out_);

  /*
  Looking at the initialization, if the condition
  below is false, necessarily, `in[1]->dims().size()`
  is equal to 1.
  */
  if (in[0]->dims() == in[1]->dims())
  {
    for (auto it0 = out_.begin(), it1 = in[1]->begin(); it0 != out_.end(); ++it0, ++it1)
    {
      T z = *it1;
      ComputationType<T>::shift_left(z, shift);
      *it0 = std::max(*it0, z);
    }
  }
  else
  {
    const Tensor<T> &B{ *in[1] };
    if (B.size() == 1)
    {
      T value{ B[0] };
      ComputationType<T>::shift_left(value, shift);
      for (auto it0 = out_.begin(); it0 != out_.end(); ++it0)
      {
        *it0 = std::max(*it0, value);
      }
    }
    else if (in[0]->dims().size() == 2)
    {
      const int N{ in[0]->dims()[0] };
      const int H{ in[0]->dims()[1] };
      for (int n = 0; n < N; ++n)
        for (int i = 0; i < H; ++i)
        {
          T z = B[i];
          ComputationType<T>::shift_left(z, shift);
          out_(n, i) = std::max(out_(n, i), z);
        }
    }
    else if (in[0]->dims().size() == 3)
    {
      const int N{ in[0]->dims()[0] };
      const int H{ in[0]->dims()[1] };
      const int W{ in[0]->dims()[2] };
      for (int n = 0; n < N; ++n)
        for (int i = 0; i < H; ++i)
          for (int j = 0; j < W; ++j)
          {
            T z = B[j];
            ComputationType<T>::shift_left(z, shift);
            out_(n, i, j) = std::max(out_(n, i, j), z);
          }
    }
    else if (in[0]->dims().size() == 4)
    {
      const int N{ in[0]->dims()[0] };
      const int H{ in[0]->dims()[1] };
      const int W{ in[0]->dims()[2] };
      const int K{ in[0]->dims()[3] };
      for (int n = 0; n < N; ++n)
        for (int i = 0; i < H; ++i)
          for (int j = 0; j < W; ++j)
            for (int k = 0; k < K; ++k)
            {
              T z = B[k];
              ComputationType<T>::shift_left(z, shift);
              out_(n, i, j, k) = std::max(out_(n, i, j, k), z);
            }
    }
  }
  return true;
}

template<typename T> bool Maximum<T>::init(const std::vector<Tensor<T> *> &in)
{
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in.size() != 2)
  {
    return false;
  }

  /*
  Broadcasting is supported. This means that either
  the two input Tensor<T>s have the same shape or the
  second input Tensor<T> is a singleton or the second
  input Tensor<T> is a vector and the last dimension
  of the first input Tensor<T> is equal to the size
  of the second input Tensor<T>.
  */
  if (in[1]->size() == 1)
  {   // singleton
      // ok
  }
  else if (in[1]->dims().size() == 1 || (in[1]->dims().size() == 2 && in[1]->dims()[0] == 1))
  {
    if (in[1]->size() != in[0]->dims().back())
    {   // broadcast last tdim
      return false;
    }
  }
  else
  {
    if (!(in[0]->dims() == in[1]->dims()))
    {   // same sim
      return false;
    }
  }
  out_.resize(in[0]->dims());
  initDone_ = true;
  return true;
}

template<typename T> bool Maximum<T>::loadInternal(std::istream &, Version)
{
  return true;
}

}   // namespace layers
}   // namespace sadl
