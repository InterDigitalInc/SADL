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
template<typename T> class Flatten : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  int32_t      axis_;
  Dimensions   dim_;   // dims after flatten
  DUMP_MODEL_EXT;
};

template<typename T> bool Flatten<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  assert(in[0]->size() == out_.size());
  // resize done at init
  swapData(*in[0], out_);

  return true;
}

template<typename T> bool Flatten<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << std::endl);
  int nb_dim = axis_ + 1;
  dim_.resize(nb_dim);
  for (int k = 0; k < axis_; ++k)
    dim_[k] = in[0]->dims()[k];
  int s = 1;
  for (int k = axis_; k < in[0]->dims().size(); ++k)
    s *= in[0]->dims()[k];
  dim_[axis_] = s;
  SADL_DBG(std::cout << "  - new shape: " << dim_ << std::endl);
  out_.resize(dim_);
  initDone_ = true;
  return true;
}

template<typename T> bool Flatten<T>::loadInternal(std::istream &file, Version)
{
  // load values
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid axis: " << x << std::endl;
    return false;
  }
  axis_ = x;
  SADL_DBG(std::cout << "  - start axis: " << axis_ << std::endl);
  return true;
}

}   // namespace layers
}   // namespace sadl
