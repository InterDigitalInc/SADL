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
#include <vector>
#include <cstdint>
#include <array>
#include <iostream>
#include "tensor.h"

namespace sadl {

namespace layers {

// should be similar to python dumper:
struct OperationType {
  enum Type {
    Copy = -1,  // internal layer
    Const = 1,  // important to have const first
    Placeholder = 2,
    Identity = 3,
    BiasAdd = 4,
    MaxPool = 5,
    MatMul = 6,
    Reshape = 7,
    Relu = 8,
    Conv2D = 9,
    Add = 10,
    Concat = 11,
    Mul = 12,
    Maximum = 13,
    LeakyRelu = 14,
    Transpose = 15,
    Flatten = 16,
    Shape = 17,
    Expand = 18,
    OperationTypeCount = 19
  };
};

struct TensorInternalType {
  enum Type {
    Int32 = 0,
    Float = 1,
    Int16 = 2,
  };
};

template <typename T>
class Layer {
 public:
  using Id = int32_t;
  using value_type = T;

  Layer(Id iid, OperationType::Type iop) : id_(iid), op_(iop) {}
  virtual ~Layer() = default;

  virtual bool apply(std::vector<Tensor<T> *> &in) = 0;       // note: we ca modify inputs for optiz purpose
  virtual bool init(const std::vector<Tensor<T> *> &in) = 0;  // run it once
  bool load(std::istream &file, Version v);

  bool initDone() const;
  virtual bool mutateInput() const { return false; }
  Tensor<T> &output();
  const std::string &name() const;
  Id id() const;
  const std::vector<Id> &inputsId() const;
  OperationType::Type op() const;
  void replaceInputId(Id old, Id newid);
#if DEBUG_MODEL
  bool computed_ = false;
#endif
#if DEBUG_KEEP_OUTPUT
  Tensor<T> outcopy_;
#endif
#if DEBUG_COUNTERS
  int64_t cpt_op = 0;
  int64_t cpt_mac_nz = 0;
  int64_t cpt_mac = 0;
  int64_t cpt_overflow = 0;
#endif
protected :
  bool loadPrefix(std::istream &file, Version v);
  virtual bool loadInternal(std::istream &file, Version v) = 0;
  Tensor<T> out_;
  const Id id_;
  const OperationType::Type op_;
  std::string name_;
  std::vector<Id> inputs_id_;
  bool initDone_ = false;
  DUMP_MODEL_EXT;
};

template <typename T>
bool Layer<T>::load(std::istream &file, Version v) {
  return loadPrefix(file, v) && loadInternal(file, v);
}

template <typename T>
bool Layer<T>::initDone() const {
  return initDone_;
}

template <typename T>
sadl::Tensor<T> &Layer<T>::output() {
  return out_;
}

template <typename T>
const std::string &Layer<T>::name() const {
  return name_;
}

template <typename T>
typename Layer<T>::Id Layer<T>::id() const {
  return id_;
}

template <typename T>
const std::vector<typename Layer<T>::Id> &Layer<T>::inputsId() const {
  return inputs_id_;
}

template <typename T>
OperationType::Type Layer<T>::op() const {
  return op_;
}

template <typename T>
void Layer<T>::replaceInputId(Layer<T>::Id old, Layer<T>::Id newid) {
  std::replace(inputs_id_.begin(), inputs_id_.end(), old, newid);
}

template <typename T>
bool Layer<T>::loadPrefix(std::istream &file, Version v) {
  initDone_ = false;
  int32_t L = 0;
  file.read((char *)&L, sizeof(int32_t));
  constexpr int maxLength = 2048;
  assert(L > 0 && L + 1 < maxLength);  // max name size
  char s[maxLength];
  file.read(s, L);
  s[L] = '\0';
  name_ = s;
  SADL_DBG(std::cout << "  - name: " << name_ << '\n');

  file.read((char *)&L, sizeof(int32_t));
  assert(L >= 0 && L < 6);
  inputs_id_.resize(L);
  SADL_DBG(std::cout << "  - inputs: ");
  for (auto &x : inputs_id_) {
    file.read((char *)&x, sizeof(int32_t));
    SADL_DBG(std::cout << x << ' ');
  }
  SADL_DBG(std::cout << '\n');
  return static_cast<bool>(file);
}

}  // namespace layers

}  // namespace sadl
