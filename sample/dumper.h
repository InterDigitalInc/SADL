#include <sadl/model.h>

template <typename T>
bool sadl::layers::Conv2D<T>::dump(std::ostream &file) {
  int32_t x = strides_.size();
  file.write((const char *)&x, sizeof(int32_t));
  file.write((const char *)strides_.begin(), strides_.size() * sizeof(int32_t));
  x=pads_.size();
  file.write((const char *)&x, sizeof(int32_t));
  file.write((const char *)pads_.begin(), pads_.size() * sizeof(int32_t));
  file.write((const char *)&q_, sizeof(q_));
  return true;
}

template <typename T>
bool sadl::layers::MatMul<T>::dump(std::ostream &file) {
  file.write((const char *)&q_, sizeof(q_));
  return true;
}

template <typename T>
bool sadl::layers::Mul<T>::dump(std::ostream &file) {
  file.write((const char *)&q_, sizeof(q_));
  return true;
}

template <typename T>
bool sadl::layers::Placeholder<T>::dump(std::ostream &file) {
  int32_t x = dims_.size();
  file.write((const char*)&x, sizeof(x));
  file.write((const char*)dims_.begin(), sizeof(int)*x);
  file.write((const char *)&q_, sizeof(q_));
  return true;
}

template <typename T>
bool sadl::layers::MaxPool<T>::dump(std::ostream &file) {
  int32_t x = strides_.size();
  file.write((const char *)&x, sizeof(int32_t));
  file.write((const char *)strides_.begin(), strides_.size() * sizeof(int32_t));
  x = kernel_.size();
  file.write((const char *)&x, sizeof(int32_t));
  file.write((const char *)kernel_.begin(), kernel_.size() * sizeof(int32_t));
  x=pads_.size();
  file.write((const char *)&x, sizeof(int32_t));
  file.write((const char *)pads_.begin(), pads_.size() * sizeof(int32_t));
  return true;
}

template <typename T>
bool sadl::layers::Flatten<T>::dump(std::ostream &file) {
  int32_t x = axis_;
  file.write((const char *)&x, sizeof(int32_t));
  return true;
}



template <typename T>
bool sadl::layers::Const<T>::dump(std::ostream &file) {
  // load values
  int32_t x = out_.dims().size();
  file.write((const char *)&x, sizeof(x));
  file.write((const char *)out_.dims().begin(), x * sizeof(int));
  if (std::is_same<T, int16_t>::value) {
    x = TensorInternalType::Int16;
  } else if (std::is_same<T, int32_t>::value) {
    x = TensorInternalType::Int32;
  } else if (std::is_same<T, float>::value) {
    x = TensorInternalType::Float;
  } else {
    std::cerr << "[ERROR] to do" << std::endl;
    exit(-1);
  }
  file.write((const char *)&x, sizeof(x));

  if (!std::is_same<T,float>::value) file.write((const char *)&out_.quantizer, sizeof(out_.quantizer));
  file.write((const char *)out_.data(), out_.size() * sizeof(T));
  return true;
}

template <typename T>
bool sadl::layers::Layer<T>::dump(std::ostream &file) {
 // std::cout<<"todo? "<<opName(op_)<<std::endl;
  return true;
}

template <typename T>
bool sadl::Model<T>::dump(std::ostream &file) {
  char magic[9] = "SADL0002";
  file.write(magic, 8);
  int32_t x = 0;
  if (std::is_same<T, float>::value)
    x = layers::TensorInternalType::Float;
  else if (std::is_same<T, int32_t>::value)
    x = layers::TensorInternalType::Int32;
  else if (std::is_same<T, int16_t>::value)
    x = layers::TensorInternalType::Int16;
  else {
    std::cerr << "[ERROR] to do Model::dump" << std::endl;
    exit(-1);
  }
  file.write((const char *)&x, sizeof(int32_t));

  int32_t nb_layers = data_.size();
  file.write((const char *)&nb_layers, sizeof(int32_t));
  int32_t nb = ids_input.size();
  file.write((const char *)&nb, sizeof(int32_t));
  file.write((const char *)ids_input.data(), sizeof(int32_t) * nb);
  nb = ids_output.size();
  file.write((const char *)&nb, sizeof(int32_t));
  file.write((const char *)ids_output.data(), sizeof(int32_t) * nb);


  for (int k = 0; k < nb_layers; ++k) {
    // save header
    int32_t x = data_[k].layer->id();
    file.write((const char *)&x, sizeof(int32_t));
    x = data_[k].layer->op();
    file.write((const char *)&x, sizeof(int32_t));
    // savePrefix
    int32_t L = data_[k].layer->name_.size();
    file.write((const char *)&L, sizeof(int32_t));
    file.write((const char *)data_[k].layer->name_.c_str(), data_[k].layer->name_.size());
    L = data_[k].layer->inputs_id_.size();
    file.write((const char *)&L, sizeof(int32_t));
    file.write((const char *)data_[k].layer->inputs_id_.data(), data_[k].layer->inputs_id_.size() * sizeof(int32_t));
    data_[k].layer->dump(file);
  }
  return true;
}

