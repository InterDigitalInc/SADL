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
class Concat : public Layer<T> {
 public:
  using Layer<T>::Layer;
  using Layer<T>::out_;  // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

 protected:
  virtual bool loadInternal(std::istream &file,Version v) override;
};

template <typename T>
bool Concat<T>::apply(std::vector<Tensor<T> *> &in) {
  assert(in.size() >= 3);
  const int nb_in = (int)in.size() - 1;                                                                        // without axis inputs
  assert((*in[nb_in]).size() == 1 && ((*in[nb_in])[0] == in[0]->dims().size() - 1 || (*in[nb_in])[0] == -1));  // currently concat on last axis
  int shift[16] = {};
  int qmin=in.front()->quantizer;
  if (!std::is_same<T, float>::value) {
    assert(nb_in < 16);
    for(int i = 0; i < nb_in; ++i) {
      if (in[i]->quantizer<qmin) qmin=in[i]->quantizer;
    }
    for (int i = 0; i < nb_in; ++i) {
      shift[i] = in[i]->quantizer-qmin;
    }
  }
  out_.quantizer = qmin;  // adapt output width to last input
  out_.border_skip=in[0]->border_skip;
  for(int i=1;i<nb_in;++i) out_.border_skip=std::max(out_.border_skip,in[i]->border_skip);

  const Dimensions dim = in[0]->dims();
  if (dim.size() == 2) {
    for (int i = 0; i < dim[0]; ++i) {
      int offset = 0;
      for (int n = 0; n < nb_in; ++n) {
        const Tensor<T> &A = *(in[n]);
        for (int j = 0; j < A.dims()[1]; ++j, ++offset) {
          T z = A(i, j);
          ComputationType<T>::quantize(z, shift[n]);
          out_(i, offset) = z;
        }
      }
    }
  } else if (dim.size() == 3) {
    for (int i = 0; i < dim[0]; ++i) {
      for (int j = 0; j < dim[1]; ++j) {
        int offset = 0;
        for (int n = 0; n < nb_in; ++n) {
          const Tensor<T> &A = *(in[n]);
          for (int k = 0; k < A.dims()[2]; ++k, ++offset) {
             T z = A(i, j, k);
            ComputationType<T>::quantize(z, shift[n]);
            out_(i, j, offset) = z;
          }
        }
      }
    }
  } else if (dim.size() == 4) {
    for (int i = 0; i < dim[0]; ++i) {
      for (int j = 0; j < dim[1]; ++j) {
        for (int k = 0; k < dim[2]; ++k) {
          int offset = 0;
          for (int n = 0; n < nb_in; ++n) {
            const Tensor<T> &A = *(in[n]);
            for (int l = 0; l < A.dims()[3]; ++l, ++offset) {
              T z = A(i, j, k, l);
              ComputationType<T>::quantize(z, shift[n]);
              out_(i, j, k, offset) =z;
            }
          }
        }
      }
    }
  } else {
    // TO DO
    return false;
  }
  return true;
}

template <typename T>
bool Concat<T>::init(const std::vector<Tensor<T> *> &in) {
  /*
  The axis of the concatenation is the third tensor
  in `in`.
  */
  if (in.size() < 3) return false;
  if (in[0]->dims().size() < 1) return false;
  const int last_axis = in[0]->dims().size() - 1;

  // Currently, the concatenation is along the last axis.
  int axis_idx = (int)in.size() - 1;
  if (!((*in[axis_idx]).size() == 1 && ((*in[axis_idx])[0] == last_axis || (*in[axis_idx])[0] == -1))) return false;

  // should have same shape
  int sum_dim = 0;
  for (int i = 1; i < axis_idx; i++) {
    if (in[0]->dims().size() != in[i]->dims().size()) return false;
    sum_dim += in[i]->dims()[last_axis];
    for (int k = 0; k < last_axis; ++k)
      if (in[0]->dims()[k] != in[i]->dims()[k]) return false;
  }
  Dimensions dim = in[0]->dims();
  dim[last_axis] += sum_dim;
  out_.resize(dim);
  initDone_ = true;
  return true;
}

template <typename T>
bool Concat<T>::loadInternal(std::istream &,Version ) {
  return true;
}

}  // namespace layers
}  // namespace sadl
