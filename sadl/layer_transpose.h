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
class Transpose : public Layer<T> {
 public:
  using Layer<T>::Layer;
  using Layer<T>::out_; // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

 protected:
  virtual bool loadInternal(std::istream &file,Version v) override;
};

// assume data in in[0] and shape in in[1]
template <typename T>
bool Transpose<T>::apply(std::vector<Tensor<T> *> &in) {
  // NHWC to NCHW
  Dimensions d = {in[0]->dims()[0], in[0]->dims()[3], in[0]->dims()[1], in[0]->dims()[2]};

  if (d[3] == 1 || d[1] * d[2] == 1) {
    swapData(*in[0], out_);
    // resize done at init
  } else {
   // d[0]=1 for (int i = 0; i < d[0]; ++i) {
      for (int j = 0; j < d[1]; ++j) { // was l
        for (int k = 0; k < d[2]; ++k) { // was j
          for (int l = 0; l < d[3]; ++l) { // was k
            out_.data()[d[3] * (d[2] * j + k) + l] = in[0]->data()[d[1] * (d[3] * k + l) + j];
          }
        }
      }
   // }
  }
  return true;
}

template <typename T>
bool Transpose<T>::init(const std::vector<Tensor<T> *> &in) {
  //  Only use transpose for Pytorch: NHWC => NCHW
  if (in.size() != 2) return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  // second layer is always reshape prms: value as int inside the tensor
  if (in[1]->dims().size() != 1) return false;
  Dimensions dim;
  dim.resize(in[1]->size());
  for (int k = 0; k < in[1]->size(); ++k) {
    if ((*in[1])(k) == -1) {  // keep dim of org
      dim[k] = in[0]->dims()[k];
    } else {
      dim[k] = in[0]->dims()[(int)((*in[1])(k))];
    }
  }
  if (dim.nbElements() != in[0]->dims().nbElements()) {
    std::cerr << "[ERROR] transpose incompatible sizes " << dim << ' ' << in[0]->dims() << std::endl;
    std::cerr << "[ERROR] ";
    for (int k = 0; k < in[1]->dims()[0]; ++k) std::cerr << (*in[1])(k) << ' ';
    std::cerr << std::endl;

    return false;
  }
  SADL_DBG(std::cout << "  - new shape: "<<dim << std::endl);
  // currently only support NHWCtoNCHW
  if (dim[0]!=in[0]->dims()[0]||
      dim[1]!=in[0]->dims()[3]||
      dim[2]!=in[0]->dims()[1]||
      dim[3]!=in[0]->dims()[2]) {
    std::cerr << "[ERROR] transpose values not supported" << std::endl;
    return false;
  }
  out_.resize(dim);
  initDone_ = true;
  return true;
}

template <typename T>
bool Transpose<T>::loadInternal(std::istream &,Version ) {
  return true;
}

}  // namespace layers
}  // namespace sadl
