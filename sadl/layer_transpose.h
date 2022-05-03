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
template<typename T> class Transpose : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool                        loadInternal(std::istream &file, Version v) override;
  std::array<int, Dimensions::MaxDim> perm_;
};

// assume data in in[0] and shape in in[1]
template<typename T> bool Transpose<T>::apply(std::vector<Tensor<T> *> &in)
{
  Dimensions d = out_.dims();   // {in[0]->dims()[0], in[0]->dims()[3],
                                // in[0]->dims()[1], in[0]->dims()[2]};
  const auto &A  = *in[0];
  Dimensions  Ad = A.dims();
  if (d.size() == 1)
    swapData(*in[0], out_);
  else if (d.size() == 4)
  {
    std::array<int, 4>   index;
    std::array<int *, 4> index_mapped;
    for (int k = 0; k < 4; ++k)
      index_mapped[k] = &index[perm_[k]];

    for (index[0] = 0; index[0] < Ad[0]; ++index[0])
      for (index[1] = 0; index[1] < Ad[1]; ++index[1])
        for (index[2] = 0; index[2] < Ad[2]; ++index[2])
          for (index[3] = 0; index[3] < Ad[3]; ++index[3])
          {
            auto offsetA    = (Ad[3] * (Ad[2] * (Ad[1] * index[0] + index[1]) + index[2]) + index[3]);
            auto offsetOut  = (d[3] * (d[2] * (d[1] * *index_mapped[0] + *index_mapped[1]) + *index_mapped[2]) + *index_mapped[3]);
            out_[offsetOut] = A[offsetA];
          }
  }
  else if (d.size() == 6)
  {   // very naive version
    std::array<int, 6>   index;
    std::array<int *, 6> index_mapped;
    for (int k = 0; k < 6; ++k)
      index_mapped[k] = &index[perm_[k]];

    for (index[0] = 0; index[0] < Ad[0]; ++index[0])
      for (index[1] = 0; index[1] < Ad[1]; ++index[1])
        for (index[2] = 0; index[2] < Ad[2]; ++index[2])
          for (index[3] = 0; index[3] < Ad[3]; ++index[3])
            for (index[4] = 0; index[4] < Ad[4]; ++index[4])
              for (index[5] = 0; index[5] < Ad[5]; ++index[5])
              {
                auto offsetA = Ad[5] * (Ad[4] * (Ad[3] * (Ad[2] * (Ad[1] * index[0] + index[1]) + index[2]) + index[3]) + index[4]) + index[5];
                auto offsetOut =
                  d[5] * (d[4] * (d[3] * (d[2] * (d[1] * *index_mapped[0] + *index_mapped[1]) + *index_mapped[2]) + *index_mapped[3]) + *index_mapped[4])
                  + *index_mapped[5];
                out_[offsetOut] = A[offsetA];
              }
  }
  else
  {
    std::cerr << "\nTODO Transpose case: " << in[0]->dims() << " => " << out_.dims() << std::endl;
    exit(-1);
  }
  //  }
  return true;
}

template<typename T> bool Transpose<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  // second layer is always reshape prms: value as int inside the tensor
  if (in[1]->dims().size() != 1)
    return false;
  Dimensions dim;
  dim.resize(in[1]->size());
  for (int k = 0; k < in[1]->size(); ++k)
  {
    if ((*in[1]) (k) == -1)
    {   // keep dim of org
      dim[k]   = in[0]->dims()[k];
      perm_[k] = k;
    }
    else
    {
      dim[k]   = in[0]->dims()[(int) ((*in[1]) (k))];
      perm_[k] = (int) ((*in[1]) (k));
    }
  }
  if (dim.nbElements() != in[0]->dims().nbElements())
  {
    std::cerr << "[ERROR] transpose incompatible sizes shuffle=[" << dim << "] input shape: " << in[0]->dims() << std::endl;
    return false;
  }
  SADL_DBG(std::cout << "  - new shape: " << dim << std::endl);
  out_.resize(dim);
  initDone_ = true;
  return true;
}

template<typename T> bool Transpose<T>::loadInternal(std::istream &, Version)
{
  return true;
}

}   // namespace layers
}   // namespace sadl
