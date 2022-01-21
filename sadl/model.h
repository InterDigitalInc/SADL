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
#include <memory>
#include <vector>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include "layer.h"
#include "layers.h"
#include "tensor.h"

namespace sadl {
// input Tensor<T> dims: depth, nb rows, nb col

template <typename T>
class Model {
private:

  struct LayerData {
    std::unique_ptr<layers::Layer<T>> layer;
    std::vector<Tensor<T> *> inputs;

  };
  std::vector<LayerData> data_;
  int32_t nb_inputs_ = 0;
  static constexpr int kMaxInputByLayer = 2;
  static constexpr int kMaxLayers = 256;
  std::vector<typename layers::Layer<T>::Id> getLayerIdsWithInput(typename layers::Layer<T>::Id id) const;
  void insertCopyLayers();
  void reshapeConv2DFilters();
  void reshapeMatrix();
  LayerData &getLayer(const typename layers::Layer<T>::Id &id);
  Version version_ = Version::unknown;
  std::vector<typename layers::Layer<T>::Id> ids_input, ids_output;

 public:
  bool load(std::istream &in);
  bool init(std::vector<Tensor<T>> &in);
  bool apply(std::vector<Tensor<T>> &in); // change input for optiz
  const Tensor<T> &result(int idx_out=0) const { return getLayer(ids_output[idx_out]).layer->output(); }

  // aditionnal info
  std::vector<Tensor<T>> getInputsTemplate() const;
  const std::vector<typename layers::Layer<T>::Id> &getIdsOutput() const { return ids_output; }
  std::vector<typename layers::Layer<T>::Id> getLayersId() const;
  const LayerData &getLayer(const typename layers::Layer<T>::Id &id) const;
  Version version() const { return version_; }

#if DEBUG_COUNTERS
  struct Stat {
      uint64_t overflow=0;
      uint64_t op=0;
      uint64_t mac=0;
      uint64_t mac_nz=0;
  };
  void resetCounters();
  Stat printOverflow(bool printinfo=false) const;

#endif

  DUMP_MODEL_EXT;
};

template <typename T>
std::unique_ptr<layers::Layer<T>> createLayer(int32_t id, layers::OperationType::Type op) {
  switch (op) {
    case layers::OperationType::Copy:
      return std::unique_ptr<layers::Layer<T>>(new layers::Copy<T>{id, op});
      break;
    case layers::OperationType::Const:
      return std::unique_ptr<layers::Layer<T>>(new layers::Const<T>{id, op});
      break;
    case layers::OperationType::Placeholder:
      return std::unique_ptr<layers::Layer<T>>(new layers::Placeholder<T>{id, op});
      break;
    case layers::OperationType::Reshape:
      return std::unique_ptr<layers::Layer<T>>(new layers::Reshape<T>{id, op});
      break;
    case layers::OperationType::Identity:
      return std::unique_ptr<layers::Layer<T>>(new layers::Identity<T>{id, op});
      break;
    case layers::OperationType::MatMul:
      return std::unique_ptr<layers::Layer<T>>(new layers::MatMul<T>{id, op});
      break;
    case layers::OperationType::BiasAdd:
      return std::unique_ptr<layers::Layer<T>>(new layers::BiasAdd<T>{id, op});
      break;
    case layers::OperationType::Conv2D:
      return std::unique_ptr<layers::Layer<T>>(new layers::Conv2D<T>{id, op});
      break;
    case layers::OperationType::Add:
      return std::unique_ptr<layers::Layer<T>>(new layers::Add<T>{id, op});
      break;
    case layers::OperationType::Relu:
      return std::unique_ptr<layers::Layer<T>>(new layers::Relu<T>{id, op});
      break;
    case layers::OperationType::MaxPool:
      return std::unique_ptr<layers::Layer<T>>(new layers::MaxPool<T>{id, op});
      break;
    case layers::OperationType::Mul:
      return std::unique_ptr<layers::Layer<T>>(new layers::Mul<T>{id, op});
      break;
    case layers::OperationType::Concat:
      return std::unique_ptr<layers::Layer<T>>(new layers::Concat<T>{id, op});
      break;
    case layers::OperationType::Maximum:
      return std::unique_ptr<layers::Layer<T>>(new layers::Maximum<T>{id, op});
      break;
    case layers::OperationType::LeakyRelu:
      return std::unique_ptr<layers::Layer<T>>(new layers::LeakyRelu<T>{id, op});
      break;
    case layers::OperationType::Transpose:
      return std::unique_ptr<layers::Layer<T>>(new layers::Transpose<T>{id, op});
      break;
    case layers::OperationType::Flatten:
      return std::unique_ptr<layers::Layer<T>>(new layers::Flatten<T>{id, op});
      break;
    case layers::OperationType::OperationTypeCount:
      break;  // no default on purpose
  }
  std::cerr << "[ERROR] unknown layer " << op << std::endl;
  exit(-1);
}

template <typename T>
bool Model<T>::load(std::istream &file) {
  if (!file) {
    std::cerr << "[ERROR] Pb reading model" << std::endl;
    return false;
  }

  SADL_DBG(std::cout << "[INFO] start model loading" << std::endl);
  char magic[9];
  file.read(magic, 8);
  magic[8] = '\0';
  SADL_DBG(std::cout << "[INFO] read magic " << magic << std::endl);
  std::string magic_s = magic;
  if (magic_s == "SADL0001") {
    version_ = Version::sadl01;
  } else {
    if (!file) {
      std::cerr << "[ERROR] Pb reading model" << std::endl;
      return false;
    }
    std::cerr << "[ERROR] Pb reading model: wrong magic " << magic_s << std::endl;
    return false;
  }

  if (version_ == Version::sadl01) {
    int32_t x = 0;
    file.read((char *)&x, sizeof(int32_t));
    if ((std::is_same<T, float>::value && x != layers::TensorInternalType::Float) || (std::is_same<T, int32_t>::value && x != layers::TensorInternalType::Int32) ||
        (std::is_same<T, int16_t>::value && x != layers::TensorInternalType::Int16)) {
      std::cerr << "[ERROR] wrong model type and Model<T>" << std::endl;
      return false;
    }
    SADL_DBG(std::cout << "[INFO] Model type: " << (int)x << std::endl);
  }

  int32_t nb_layers = 0;
  file.read((char *)&nb_layers, sizeof(int32_t));
  SADL_DBG(std::cout << "[INFO] Num layers: " << nb_layers << std::endl);
  if (nb_layers <= 0 || nb_layers > kMaxLayers) {
    std::cerr << "[ERROR] Pb reading model: nb layers " << nb_layers << std::endl;
    return false;
  }
  data_.clear();
  data_.resize(nb_layers);

  if (version_ == Version::sadl01) {
    int32_t nb;
    file.read((char *)&nb, sizeof(int32_t));
    ids_input.resize(nb);
    file.read((char *)ids_input.data(), sizeof(int32_t) * nb);
    file.read((char *)&nb, sizeof(int32_t));
    ids_output.resize(nb);
    file.read((char *)ids_output.data(), sizeof(int32_t) * nb);
    SADL_DBG(std::cout << "[INFO] input id: " );
    for (auto id : ids_input) {
      SADL_DBG(std::cout << id << ' ');
      (void)id;
    }
    SADL_DBG(std::cout << std::endl);
    SADL_DBG(std::cout << "[INFO] output id: " );
    for (auto id : ids_output) {
      SADL_DBG(std::cout <<  id << ' ');
      (void)id;
    }
    SADL_DBG(std::cout << std::endl);
  }

  for (int k = 0; k < nb_layers; ++k) {
    typename layers::Layer<T>::Id id = 0;
    file.read((char *)&id, sizeof(int32_t));
    int32_t op = 0;
    file.read((char *)&op, sizeof(int32_t));
    if (!(op > 0 && op < layers::OperationType::OperationTypeCount)) {
      std::cerr << "[ERROR] Pb reading model: layer op " << op << std::endl;
      return false;
    }
    SADL_DBG(std::cout << "[INFO] id: " << id << " op " << ' ' << layers::opName((layers::OperationType::Type)op)
                         << std::endl);  // opName((layers::OperationType::Type)op)<<std::endl);
    data_[k].layer = createLayer<T>(id, (layers::OperationType::Type)op);
    data_[k].inputs.clear();
    if (!data_[k].layer->load(file, version_)) {
      data_.clear();
      return false;
    }
  }

  if (data_.empty()) {
    std::cerr << "[ERROR] Pb reading model: no layer" << std::endl;
    return false;
  }
  SADL_DBG(std::cout << "[INFO] end model loading\n" << std::endl);

  return true;
}

template <typename T>
bool Model<T>::init(std::vector<Tensor<T>> &in) {
  SADL_DBG(std::cout << "[INFO] start model init" << std::endl);

  if (std::is_same<T, float>::value) {
    SADL_DBG(std::cout << "[INFO] float mode" << std::endl);
  } else if (std::is_same<T, int32_t>::value) {
    SADL_DBG(std::cout << "[INFO] int32 mode" << std::endl);
  } else if (std::is_same<T, int16_t>::value) {
    SADL_DBG(std::cout << "[INFO] int16 mode" << std::endl);
  } else {
    std::cerr << "[ERROR] unsupported type" << std::endl;
    return false;
  }
#if __AVX2__
  SADL_DBG(std::cout << "[INFO] use SIMD code" << std::endl);
#endif
#if __AVX512F__
  SADL_DBG(std::cout << "[INFO] use SIMD512 code" << std::endl);
#endif
#if __FMA__
  SADL_DBG(std::cout << "[INFO] use FMA" << std::endl);
#endif
  SADL_DBG(std::cout << "[INFO] use swapped tensor" << std::endl);

  if (data_.empty()) {
    std::cerr << "[ERROR] Empty model" << std::endl;
    return false;
  }
  nb_inputs_ = (int)in.size();
  if ((version_ == Version::sadl01) && nb_inputs_ != (int)ids_input.size()) {
    std::cerr << "[ERROR] inconsistent input dimension" << std::endl;
    return false;
  }
  insertCopyLayers();

  reshapeConv2DFilters();
  reshapeMatrix();

  // first solve inputs for placeholders (the inputs)
  bool ok = true;
  int placeholders_cnt = 0;
  for (int layer_cnt = 0; layer_cnt < (int)data_.size() && ok; ++layer_cnt) {
    if (data_[layer_cnt].layer->op() == layers::OperationType::Placeholder) {
      if (placeholders_cnt >= (int)in.size()) {
        std::cerr << "[ERROR] more placeholders than inputs" << std::endl;
        ok = false;
        break;
      }
      if (data_[layer_cnt].layer->inputsId().size() != 0) {
        std::cerr << "[ERROR] placeholders should have only 0 input" << std::endl;
        ok = false;
        break;
      }
      std::vector<Tensor<T> *> v = {&in[placeholders_cnt]};
      ++placeholders_cnt;
      SADL_DBG(std::cout << "[INFO] init layer " << data_[layer_cnt].layer->id() << ' ' << layers::opName((layers::OperationType::Type)(data_[layer_cnt].layer->op())) << ' '
                           << data_[layer_cnt].layer->name() << std::endl);
      data_[layer_cnt].layer->init(v);
    }
  }
  if (!ok) return false;
  if (placeholders_cnt != (int)in.size()) {
    std::cerr << "[ERROR] less placeholders than inputs" << std::endl;
    return false;
  }

  // then solve inputs of other layers: make the link between id of inputs and tensor ptr
  for (int layer_cnt = 0; layer_cnt < (int)data_.size() && ok; ++layer_cnt) {
    if (data_[layer_cnt].layer->op() == layers::OperationType::Placeholder) continue;
    int nb_inputs = (int)data_[layer_cnt].layer->inputsId().size();
    data_[layer_cnt].inputs.resize(nb_inputs);
    std::vector<layers::OperationType::Type> op_type(nb_inputs);
    for (int inputs_cnt = 0; inputs_cnt < nb_inputs; ++inputs_cnt) {
       
      typename layers::Layer<T>::Id id_input = data_[layer_cnt].layer->inputsId()[inputs_cnt];
      auto &L = getLayer(id_input);
      if (!L.layer->initDone()) {
        std::cerr << "[ERROR] init not done yet on " << L.layer->id() << " while init of " << data_[layer_cnt].layer->id() << std::endl;
        return false;
      }
      data_[layer_cnt].inputs[inputs_cnt] = &(L.layer->output());
      op_type[inputs_cnt] = L.layer->op();

      // always put data layers first when const layers
      if (inputs_cnt > 0 && op_type[inputs_cnt - 1] == layers::OperationType::Const && op_type[inputs_cnt] != layers::OperationType::Const) {
        std::cerr << "[ERROR] data layers should be first" << std::endl;
        return false;
      }
    }
    SADL_DBG(std::cout << "[INFO] init layer " << data_[layer_cnt].layer->id() << ' ' << layers::opName((layers::OperationType::Type)(data_[layer_cnt].layer->op())) << " "
                         << data_[layer_cnt].layer->name() << std::endl);
    ok &= data_[layer_cnt].layer->init(data_[layer_cnt].inputs);
    if (!ok) {
      std::cerr << "[ERROR] init layer " << data_[layer_cnt].layer->id() << " " << data_[layer_cnt].layer->name() << std::endl;
      break;
    }
  }
  SADL_DBG(std::cout << "[INFO] end model init\n" << std::endl);

  if (!ok) return false;

  return true;
}

template <typename T>
bool Model<T>::apply(std::vector<Tensor<T>> &in) {
  assert(!data_.empty());
  assert((int)in.size() == nb_inputs_);
  // should be ok in order (take care of that on python side)
  bool ok = true;
  int placeholders_cnt = 0;
  for (int layer_cnt = 0; layer_cnt < (int)data_.size() && ok; ++layer_cnt) {
    if (data_[layer_cnt].layer->op() == layers::OperationType::Placeholder) {
      std::vector<Tensor<T> *> v = {&in[placeholders_cnt]};
      ++placeholders_cnt;
      ok &= data_[layer_cnt].layer->apply(v);
#if DEBUG_VALUES
      std::cout << "[INFO] " << data_[layer_cnt].layer->id() << " " << data_[layer_cnt].layer->name() << " [PlaceHolder]: q=" << data_[layer_cnt].layer->out_.quantizer << " [";
      float Q = (1 << data_[layer_cnt].layer->out_.quantizer);
      for (int k = 0; k < 8 && k < (int)data_[layer_cnt].layer->out_.size(); ++k) std::cout << data_[layer_cnt].layer->out_[k] / Q << ' ';
      std::cout << "]" << std::endl;
#endif
#if DEBUG_MODEL
      data_[layer_cnt].layer->computed_ = true;
      SADL_DBG(std::cout << "[INFO] layer " << layer_cnt << ": " << (ok ? "ok" : "failed") << std::endl;)
#endif
#if DEBUG_KEEP_OUTPUT
      data_[layer_cnt].layer->outcopy_=data_[layer_cnt].layer->out_;
#endif
    }
  }

  for (int layer_cnt = 0; layer_cnt < (int)data_.size() && ok; ++layer_cnt) {
    if (data_[layer_cnt].layer->op() == layers::OperationType::Placeholder) continue;
#if DEBUG_MODEL
    for (int kk = 0; kk < (int)data_[layer_cnt].inputs.size(); ++kk) {
      const int id = data_[layer_cnt].layer->inputsId()[kk];
      const auto &L = getLayer(id);
      (void)L;
      assert(L.layer->computed_);
    }
#endif
#if DEBUG_VALUES
    std::cout << "[INFO] " << data_[layer_cnt].layer->id() << " " << data_[layer_cnt].layer->name() << " " << opName(data_[layer_cnt].layer->op()) << "]: inputs=[";
    for (int kk = 0; kk < (int)data_[layer_cnt].inputs.size(); ++kk) {
      const int id = data_[layer_cnt].layer->inputs_id_[kk];
      std::cout << id << " (q=" << data_[layer_cnt].inputs[kk]->quantizer << ") ";
    }
    std::cout << "] ";
#endif

    ok &= data_[layer_cnt].layer->apply(data_[layer_cnt].inputs);
#if DEBUG_VALUES
    std::cout << "q=" << data_[layer_cnt].layer->out_.quantizer << " [";
    float Q = (1 << data_[layer_cnt].layer->out_.quantizer);
    for (int k = 0; k < 8 && k < (int)data_[layer_cnt].layer->out_.size(); ++k) std::cout << data_[layer_cnt].layer->out_[k] / Q << ' ';
    std::cout << "]" << std::endl;
#endif
#if DEBUG_MODEL
    data_[layer_cnt].layer->computed_ = true;
    SADL_DBG(std::cout << "[INFO] layer " << layer_cnt << " (" << layers::opName((layers::OperationType::Type)(data_[layer_cnt].layer->op())) << "): " << (ok ? "ok" : "failed")<< std::endl;)
#endif
#if DEBUG_KEEP_OUTPUT
    data_[layer_cnt].layer->outcopy_=data_[layer_cnt].layer->out_;
#endif
  }
  return ok;
}

#if DEBUG_COUNTERS

template <typename T>
typename Model<T>::Stat Model<T>::printOverflow(bool printinfo) const {
  Stat stat;
  for (int layer_cnt = 0; layer_cnt < (int)data_.size(); ++layer_cnt) {
    stat.overflow += data_[layer_cnt].layer->cpt_overflow;
    stat.op += data_[layer_cnt].layer->cpt_op;
    stat.mac += data_[layer_cnt].layer->cpt_mac;
    stat.mac_nz += data_[layer_cnt].layer->cpt_mac_nz;
    if (data_[layer_cnt].layer->cpt_overflow > 0) {
      std::cout << "[WARN] layer " << data_[layer_cnt].layer->id() << ' ' << data_[layer_cnt].layer->name() << " [" << opName(data_[layer_cnt].layer->op())
                << "]: overflow: " << data_[layer_cnt].layer->cpt_overflow << '/' << data_[layer_cnt].layer->cpt_op
                << " ("<<data_[layer_cnt].layer->cpt_overflow*100./data_[layer_cnt].layer->cpt_op << "%)"<< std::endl;
    } else if (printinfo && data_[layer_cnt].layer->cpt_op > 0) {
      std::cout << "[INFO] layer " << data_[layer_cnt].layer->id() << ' ' << data_[layer_cnt].layer->name() << " [" << opName(data_[layer_cnt].layer->op())
                << "]: "<< data_[layer_cnt].layer->cpt_op << " op"<<std::endl;
    }
  }
  return stat;
}

template <typename T>
void Model<T>::resetCounters() {
  for (int layer_cnt = 0; layer_cnt < (int)data_.size(); ++layer_cnt) {
    if (data_[layer_cnt].layer->op() == layers::OperationType::Placeholder) continue;
    data_[layer_cnt].layer->cpt_overflow = 0;
    data_[layer_cnt].layer->cpt_op = 0;
    data_[layer_cnt].layer->cpt_mac = 0;
    data_[layer_cnt].layer->cpt_mac_nz = 0;
  }
}
#endif

template <typename T>
std::vector<typename layers::Layer<T>::Id> Model<T>::getLayerIdsWithInput(typename layers::Layer<T>::Id id) const {
  std::vector<typename layers::Layer<T>::Id> v;
  for (auto &L : data_) {
    const auto &ids = L.layer->inputsId();
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) v.push_back(L.layer->id());
  }
  return v;
}

template <typename T>
typename Model<T>::LayerData &Model<T>::getLayer(const typename layers::Layer<T>::Id &id) {
  auto it = std::find_if(data_.begin(), data_.end(), [&, id](const LayerData &d) { return d.layer->id() == id; });
  if (it == data_.end()) {
    std::cerr << "[ERROR] cannot find input " << id << std::endl;
    assert(false);
    exit(-1);
  }
  return *it;
}

template <typename T>
const typename Model<T>::LayerData &Model<T>::getLayer(const typename layers::Layer<T>::Id &id) const {
  auto it = std::find_if(data_.begin(), data_.end(), [&, id](const LayerData &d) { return d.layer->id() == id; });
  if (it == data_.end()) {
    std::cerr << "[ERROR] cannot find input " << id << std::endl;
    assert(false);
    exit(-1);
  }
  return *it;
}

template<typename T>
std::vector<typename layers::Layer<T>::Id> Model<T>::getLayersId() const {
  std::vector<typename layers::Layer<T>::Id> ids;
  for(const auto &L: data_) ids.push_back(L.layer->id());
  return ids;
}

// insert copy layer before some layers inputs to deal with mutability of inputs
template <typename T>
void Model<T>::insertCopyLayers() {
  typename layers::Layer<T>::Id cnt_id = -1;  // copy layers have negative id
  // create addtionnal copy layer if needed
  for (int k = 0; k < (int)data_.size(); ++k) {
    auto &current_layer = *data_[k].layer;
    auto layer_with_current_as_input = getLayerIdsWithInput(current_layer.id());
    std::vector<typename layers::Layer<T>::Id> layer_with_current_as_mutable_input;
    // remove layers which does not modify their input
    for (auto id : layer_with_current_as_input) {
      const auto &L = getLayer(id);
      if (L.layer->mutateInput()) layer_with_current_as_mutable_input.push_back(id);
    }
    if (layer_with_current_as_mutable_input.size() > 1) {      // need copy layer
      assert(layer_with_current_as_mutable_input.size() < 3);  // for now. can be removed ?
      // for current layer L, insert copy layers C just after: x x x L C C xxxx
      std::vector<typename layers::Layer<T>::Id> id_copy_layers;
      for (int n = 0; n < (int)layer_with_current_as_mutable_input.size() - 1; ++n) {
        LayerData copy_layer;
        id_copy_layers.push_back(cnt_id);
        copy_layer.layer = createLayer<T>(cnt_id, layers::OperationType::Copy);
        dynamic_cast<layers::Copy<T> &>(*copy_layer.layer).setInputLayer(current_layer.id());

        SADL_DBG(std::cout << "[INFO] insert copy id=" << cnt_id << " of id=" << current_layer.id() << std::endl);
        --cnt_id;
        data_.insert(data_.begin() + k + 1, std::move(copy_layer));
      }
      // now change inputs of the layers to a copy of the output of the current layer (except the first one which keep the output of the current layer)
      for (int n = 1; n < (int)layer_with_current_as_mutable_input.size(); ++n) {
        auto &L = getLayer(layer_with_current_as_mutable_input[n]);
        SADL_DBG(std::cout << "[INFO] replace id=" << current_layer.id() << " by id=" << id_copy_layers[n - 1] << " in layer " << L.layer->id() << std::endl);
        L.layer->replaceInputId(current_layer.id(), id_copy_layers[n - 1]);
      }
    }
  }
  SADL_DBG(std::cout << "[INFO] inserted " << (abs(cnt_id) - 1) << " copy layers" << std::endl);
}

template <typename T>
void Model<T>::reshapeMatrix() {
  for (auto &v : data_) {
    if (v.layer->op() == layers::OperationType::MatMul) {
      if (v.layer->inputsId().size() != 2) {
        std::cerr << "[ERROR] cannot find input 2 for MatMul in reshapeMatrix()" << std::endl;
        assert(false);
        exit(-1);
      }
      auto &L = getLayer(v.layer->inputsId()[1]);
      auto &R = L.layer->output();
      // invert k and l dimensions
      Dimensions d = R.dims();
      if (d.size() == 2) {  // only transpose dim 2 for now
        // do not swap dim, just data
        SADL_DBG(std::cout << "[INFO] transpose data " << L.layer->id() << ' ' << L.layer->name() << " " << R.dims() << std::endl);
        Tensor<T> T2(d);
        T2.quantizer = R.quantizer;
        for (int i = 0; i < d[0]; ++i)
          for (int j = 0; j < d[1]; ++j) T2[j * d[0] + i] = R(i, j);
        swap(R, T2);
      }
    }
  }
}

template <typename T>
void Model<T>::reshapeConv2DFilters() {
  for (auto &v : data_) {
    if (v.layer->op() == layers::OperationType::Conv2D) {
      if (v.layer->inputsId().size() != 2) {
        std::cerr << "[ERROR] cannot find input 2 for reshapeConv2DFilters" << std::endl;
        assert(false);
        exit(-1);
      }
      auto &L = getLayer(v.layer->inputsId()[1]);
      auto &W = L.layer->output();
      // invert k and l dimensions
      Dimensions d = W.dims();
      if (d.size() != 4) {
        std::cerr << "[ERROR] invalid dim in reshapeConv2DFilters" << std::endl;
        assert(false);
        exit(-1);
      }
      auto tmp = d[2];
      d[2] = d[3];
      d[3] = tmp;
      SADL_DBG(std::cout << "[INFO] reshape " << L.layer->id() << ' ' << L.layer->name() << " " << W.dims() << " => " << d << std::endl);
      Tensor<T> T2(d);
      T2.quantizer = W.quantizer;
      for (int i = 0; i < d[0]; ++i)
        for (int j = 0; j < d[1]; ++j)
          for (int k = 0; k < d[2]; ++k)
            for (int l = 0; l < d[3]; ++l) T2(i, j, k, l) = W(i, j, l, k);
      swap(W, T2);
    }
  }
}


template <typename T>
std::vector<Tensor<T>> Model<T>::getInputsTemplate() const {
  assert(!data_.empty());
  std::vector<Tensor<T>> v;
  if (version_ < Version::sadl01) {
    std::cerr<<"[ERROR] not available"<<std::endl;
    return v;
  }

  for (auto &id_input: ids_input) {
      auto &L_tmp = getLayer(id_input);
      if (L_tmp.layer->op() == layers::OperationType::Placeholder) {
          const auto &L = dynamic_cast<const layers::Placeholder<T> &>(*L_tmp.layer);
      Tensor<T> t;
      t.resize(L.dims());
      t.quantizer=L.quantizer();
      v.push_back(t);
    }
  }
  return v;
}

}  // namespace sadl
