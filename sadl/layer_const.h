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
#include <cmath>

#include "layer.h"

namespace sadl {
namespace layers {

template <typename T>
class Const : public Layer<T> {
 public:
  using Layer<T>::Layer;
  using Layer<T>::out_; // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

 protected:
  virtual bool loadInternal(std::istream &file,Version v) override;
  template<typename U> void readTensor(std::istream &file, Tensor<T> &out);
  DUMP_MODEL_EXT;
};

template <typename T>
bool Const<T>::apply(std::vector<Tensor<T> *> &in) {
  assert(in.size() == 0);
  (void)in;
  // assert(ptr==ptr)
  return true;
}

template <typename T>
bool Const<T>::init(const std::vector<Tensor<T> *> &in) {
  if (in.size() != 0) return false;
  initDone_ = true;
  return true;
}

template <typename T>
template<typename U>
void Const<T>::readTensor(std::istream &file, Tensor<T> &out) {
  if (std::is_same<T, U>::value) file.read((char *)out.data(), sizeof(T) * out.size());
  else {
    std::vector<U> data(out.size());
    file.read((char *)data.data(),sizeof(U)*data.size());
    for(int k=0;k<(int)data.size();++k) out[k]=static_cast<T>(data[k]);
  }
}




template <typename T>
bool Const<T>::loadInternal(std::istream &file,Version v) {
  // load values
  int32_t x = 0;
  file.read((char *)&x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim) {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  Dimensions d;
  d.resize(x);
  for (int k = 0; k < d.size(); ++k) {
    file.read((char *)&x, sizeof(x));
    d[k] = x;
  }

  if (d.nbElements() >= Tensor<T>::kMaxSize) {
    std::cerr << "[ERROR] tensor too large? " << d.nbElements() << std::endl;
    return false;
  }
  out_.resize(d);
  SADL_DBG(std::cout << "  - tensor: " << out_.dims() << std::endl);

  file.read((char *)&x, sizeof(x));

  // cannot check internal type because tensor also used by reshape etc.
  switch (x) {
    case TensorInternalType::Int32:
      //assert((std::is_same<T,int32_t>::value));
      file.read((char *)&out_.quantizer, sizeof(out_.quantizer));
      readTensor<int32_t>(file, out_);
      break;
    case TensorInternalType::Float:
      //assert((std::is_same<T, float>::value));
      readTensor<float>(file, out_);
      break;
    case TensorInternalType::Int16:
      //assert((std::is_same<T, int16_t>::value));
      file.read((char *)&out_.quantizer, sizeof(out_.quantizer));
      readTensor<int16_t>(file, out_);
      break;
    default:
      std::cerr << "[ERROR] unknown internal type " << x << std::endl;
      return false;
  }

  SADL_DBG(std::cout << "  - data: "; for (int k = 0; k < 4 && k < out_.size(); ++k) std::cout << out_[k] << ' ';
             std::cout << " ...\n");
  SADL_DBG(std::cout << "  - quantizer: " << out_.quantizer << std::endl);
  // SADL_DBG(std::cout<<out_<<std::endl;)
  return true;
}

}  // namespace layers
}  // namespace sadl
