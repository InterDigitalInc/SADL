#include <sadl/model.h>
#include <iostream>
#include <fstream>

using namespace std;

template <typename T>
bool copy(const sadl::layers::Layer<float> &layer, sadl::layers::Layer<T> &layerQ) {
  // from loadPrefix
  layerQ.name_=layer.name_;
  layerQ.inputs_id_=layer.inputs_id_;
  // WARNING: SHOULD BE SYNC BY HAND WITH NEW LAYERS
  // IF LOADINTERNAL IMPLEMENTED FOR A LAYER
  switch(layerQ.op()) {
    case sadl::layers::OperationType::Add: break;
    case sadl::layers::OperationType::BiasAdd: break;
    case sadl::layers::OperationType::Concat: break;
    case sadl::layers::OperationType::Const: layerQ.out_.resize(layer.out_.dims()); for(int k=0;k<layer.out_.size();++k) layerQ.out_[k]=layer.out_[k]; break;
    case sadl::layers::OperationType::Conv2D:
      dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).strides_=dynamic_cast<const sadl::layers::Conv2D<float> &>(layer).strides_;
      dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).pads_=dynamic_cast<const sadl::layers::Conv2D<float> &>(layer).pads_;
      break;
    case sadl::layers::OperationType::Conv2DTranspose:
      dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).strides_=dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).strides_;
      dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).pads_=dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).pads_;
      dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).out_pads_=dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).out_pads_;
      break;
    case sadl::layers::OperationType::Copy: break;
    case sadl::layers::OperationType::Identity: break;
    case sadl::layers::OperationType::LeakyRelu: break;
    case sadl::layers::OperationType::MatMul: break;
    case sadl::layers::OperationType::MaxPool:
      dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).kernel_=dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).kernel_;
      dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).strides_=dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).strides_;
      dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).pads_=dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).pads_;
      break;
    case sadl::layers::OperationType::Maximum: break;
    case sadl::layers::OperationType::Mul: break;
    case sadl::layers::OperationType::Placeholder: /* do not copy q */; break;
    case sadl::layers::OperationType::Relu: break;
    case sadl::layers::OperationType::Reshape: break;
    case sadl::layers::OperationType::OperationTypeCount: break;
    case sadl::layers::OperationType::Transpose:
      dynamic_cast<sadl::layers::Transpose<T> &>(layerQ).perm_=dynamic_cast<const sadl::layers::Transpose<float> &>(layer).perm_;
      break;
    case sadl::layers::OperationType::Flatten:
      dynamic_cast<sadl::layers::Flatten<T> &>(layerQ).axis_=dynamic_cast<const sadl::layers::Flatten<float> &>(layer).axis_;
      dynamic_cast<sadl::layers::Flatten<T> &>(layerQ).dim_=dynamic_cast<const sadl::layers::Flatten<float> &>(layer).dim_;
      break;
    case sadl::layers::OperationType::Shape: break;
    case sadl::layers::OperationType::Expand: break;
      // no default to get warning
  }

  return true;
}

template <typename T>
bool copy(const sadl::Model<float> &model, sadl::Model<T> &modelQ) {
  modelQ.version_ = model.version_;
  modelQ.data_.clear();
  modelQ.data_.resize(model.data_.size());
  modelQ.ids_input = model.ids_input;
  modelQ.ids_output = model.ids_output;
  int nb_layers = modelQ.data_.size();
  for (int k = 0; k < nb_layers; ++k) {
    modelQ.data_[k].layer = sadl::createLayer<T>(model.data_[k].layer->id(), model.data_[k].layer->op());
    modelQ.data_[k].inputs.clear();
    copy(*model.data_[k].layer, *modelQ.data_[k].layer);
  }
  return true;
}

