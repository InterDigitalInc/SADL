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
template<typename T> class Expand : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
};

// assume data in in[0] and shape in in[1]
template<typename T> bool Expand<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  // second layer is reshape prms, already process in init
  out_.border_skip = in[0]->border_skip;   // adapt output width to bias
  out_.quantizer=in[0]->quantizer;

  if (in[0]->size() == 1)
  {   // broadcast
    const auto v = (*in[0])[0];
    fill(out_.begin(), out_.end(), v);
  }
  else
  {
    // quick hack: to improve
    if (out_.dims().size() == 4)
    {
      const Dimensions d = out_.dims();
      assert(d[0] == 1);
      assert(in[0]->dims()[3] == 1);
      for (int i = 0; i < 1 /*d[0]*/; ++i)
      {
        for (int j = 0; j < d[1]; ++j)
        {
          for (int k = 0; k < d[2]; ++k)
          {
            const auto offset_in0 = (d[2] * (d[1] * i + j) + k);
            const auto offset_in1 = d[3] * offset_in0;
            const auto v          = in[0]->data()[offset_in0];
            for (int l = 0; l < out_.dims()[3]; ++l)
            {
              out_.data()[offset_in1 + l] = v;
            }
          }
        }
      }
    }
    else
    {
      SADL_DBG(std::cout << "TODO" << std::endl);
      exit(-1);
    }
  }
  return true;
}

template<typename T> bool Expand<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  // second layer is always reshape prms: value as int inside the tensor
  if (in[1]->dims().size() != 1)
    return false;
  Dimensions dim;
  dim.resize(in[1]->size());
  if (!std::is_same<float,T>::value&&in[1]->quantizer!=0) {
      std::cerr << "[ERROR] quantizer on reshape dimensions data layer" << std::endl;
      return false;
  }
  copy(in[1]->begin(), in[1]->end(), dim.begin());
  // current restriction: broadcast only scalar to shape or expand last channel =1 of a tensor of dim 4
  bool ok = false;
  if (in[0]->size() == 1)
  {
    ok = true;
  }
  else
  {
    if (in[0]->dims().size() != dim.size() || dim.size() != 4)
    {
      ok = false;
    }
    else
    {
      ok = (in[0]->dims().back() == 1);
      for (int k = 0; k < dim.size() - 1; ++k)
        if (in[0]->dims()[k] != dim[k])
          ok = false;
    }
  }
  if (!ok)
  {
    std::cerr << "[ERROR] value to expand not supported " << in[0]->dims() << " expand to " << dim << std::endl;
    return false;
  }
  out_.resize(dim);
  SADL_DBG(std::cout << "  - new shape: " << dim << std::endl);
  initDone_ = true;
  return true;
}

template<typename T> bool Expand<T>::loadInternal(std::istream &, Version)
{
  return true;
}

}   // namespace layers
}   // namespace sadl
