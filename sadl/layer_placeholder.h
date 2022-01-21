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

// like an identity
template<typename T>
class Placeholder : public Layer<T> {
public:
    using Layer<T>::Layer;
    using Layer<T>::out_;
    using Layer<T>::initDone_;

    virtual bool apply(std::vector<Tensor<T>*> &in) override;
    virtual bool init(const std::vector<Tensor<T> *> &in) override;
    virtual bool mutateInput() const override { return true; }
    int quantizer() const { return q_; }
    Dimensions dims() const { return dims_; }

 protected:
    virtual bool loadInternal(std::istream &file,Version v) override;
    int q_=-1000;    // will override user input
    Dimensions dims_; // can be use as a hint by user
    DUMP_MODEL_EXT;
};

template<typename T>
bool Placeholder<T>::apply(std::vector<Tensor<T> *> &in)
{
    assert(in.size()==1);
    swap(*in[0],out_);
    if (q_>=0) { // v2
      out_.quantizer=q_;
    }
    out_.border_skip=0;
    return true;
}

template<typename T>
bool Placeholder<T>::init(const std::vector<Tensor<T> *> &in)
{
    if (in.size()!=1) return false;
    out_.resize(in[0]->dims());
    dims_=in[0]->dims();
    initDone_=true;
    return true;
}

template<typename T>
bool Placeholder<T>::loadInternal(std::istream &file, Version v)
{
  if (v == Version::sadl01) {
    int32_t x = 0;
    file.read((char*)&x, sizeof(x));
    if (x <= 0 || x > Dimensions::MaxDim) {
      std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
      return false;
    }
    dims_.resize(x);
    file.read((char*)dims_.begin(), sizeof(int)*x);
    // HACK
    if (dims_.size()==1) {
      x=dims_[0];
      dims_.resize(2);
      dims_[0]=1;
      dims_[1]=x;
    }
    // END HACK
    file.read((char*)&q_, sizeof(q_));
    SADL_DBG(std::cout << "  - dim: " <<dims_ << std::endl);
    SADL_DBG(std::cout << "  - q: " <<q_ << std::endl);
  }
  return true;
}

}
}
