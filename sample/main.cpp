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

int verbose = 0;
#include <sadl/model.h>
#include <cmath>
#include <fstream>

using namespace std;
#define MAGICNUMBER "SADL0001" 

namespace {
sadl::layers::TensorInternalType::Type getModelType(const std::string &filename) {
  ifstream file(filename, ios::binary);
  char magic[9];
  file.read(magic, 8);
  magic[8] = '\0';
  std::string magic_s = magic;
  if (magic_s != MAGICNUMBER) {
    std::cerr << "[ERROR] Pb reading model: wrong magic " << magic_s << std::endl;
    exit(-1);
  }

  int8_t x = 0;
  file.read((char *)&x, sizeof(int8_t));
  return (sadl::layers::TensorInternalType::Type)x;
}

// results file formnat:
// nb_inputs [int]
// for k in [0..nb_inputs[:
//   input_dim_size[k] [int]                  # in [1-4]
//   dim_input[k]      [int[input_dim_size]]
//   input[k]          [float[nb_elts]]       # nb_elts=product(dim_input[k])
// nb_output [int] # currently == 1
// for k in [0..nb_output[:
//   output_dim_size [int] # [1-4]
//   dim_output[k]   [int[output_dim_size]]
//   output[k]       [float[nb_elts]]         # nb_elts=product(dim_output[k])
template<typename T>
bool readResults(const string &filename, std::vector<sadl::Tensor<T>> &inputs, std::vector<float> &output, sadl::Dimensions &dim_out) {
  ifstream file(filename, ios::binary);
  int nb;
  file >> nb;
  if (nb!=(int)inputs.size()) {
    cerr << "[ERROR] invalid nb tensors" << endl;
    return false;
  }
  for (auto &t : inputs) {
    sadl::Dimensions d;
    file >> nb;
    if (nb < 1 || nb > 4) {
      cerr << "[ERROR] invalid dim in" << endl;
      return false;
    }
    d.resize(nb);
    for(auto &x: d) file>>x;
    if (!(d==t.dims())) {
      cerr << "[ERROR] invalid dimension tensor" << d<<" " <<t.dims()<<endl;
      return false;
    }
    const int Q = (1 << t.quantizer);
    for (auto &x : t) {
      float z;
      file >> z;
      if (!std::is_same<T,float>::value) {
        z = round(z * Q); // no half
      }
      x= (T)z;
    }
  }

  // outputs
  file >> nb;
  if (nb != 1) {
    cerr << "[ERROR] invalid nb output " << nb << endl;
    return false;
  }
  {
    file >> nb;
    if (nb < 1 || nb > 4) {
      cerr << "[ERROR] invalid dim out" << endl;
      return false;
    }
    dim_out.resize(nb);
    for (auto &x : dim_out) file >> x;
    output.resize(dim_out.nbElements());
    for (auto &x : output) file >> x;
  }
  return !file.fail();
}

template<typename T>
bool checkResults(const std::vector<float> &gt, const sadl::Tensor<T> &test, double abs_tol,int border_to_skip) {
  double max_a = 0.;
  int nb_e = 0;
  double max_fabs = 0.;
  double mae = 0.;
  float Q = (1 << test.quantizer);
  auto check_value = [&](auto x_test, auto x_gt) {
    float x = (float)x_test / Q;
    double a = fabs(x - x_gt);
    double fb = fabs(x_gt);
    max_fabs = max(max_fabs, fb);
    mae += fb;
    max_a = max(a, max_a);
    if (a > abs_tol) {
      ++nb_e;
    }
  };

  int nb_tested=0;
  if (border_to_skip) {
    if (test.dims().size()!=4) {
      cerr<<"[ERROR] need a tensor of dim 4 to skip border"<<endl;
      return false;
    }
    sadl::Tensor<float> gtt(test.dims());
    copy(gt.begin(),gt.end(),gtt.begin());
    for(int k=0;k<test.dims()[0];++k)
      for(int i=border_to_skip;i<test.dims()[1]-border_to_skip;++i)
        for(int j=border_to_skip;j<test.dims()[2]-border_to_skip;++j)
          for(int c=0;c<test.dims()[3];++c,++nb_tested)
            check_value(test(k,i,j,c), gtt(k,i,j,c));
  } else {
   for (int cpt = 0; cpt < (int)gt.size(); ++cpt,++nb_tested) check_value(test[cpt], gt[cpt]);
  }


  if (nb_e > 0) {
    cout << "[ERROR] test FAILED " << nb_e << "/" << nb_tested << " ";
  } else {
    cout << "[INFO] test OK ";
  }
  cout << " Qout=" << test.quantizer;
  cout << " max: max_error=" << max_a << " (th=" << abs_tol << "), output: max |x|=" << max_fabs << " av|x|=" << mae / nb_tested << endl;
  return nb_e == 0;
}

template <typename T>
void infer(const std::string &filename_results, const std::string &filename, double max_e,int border_to_skip) {
  sadl::Model<T> model;
  std::ifstream file(filename, ios::binary);
  std::cout<<"[INFO] Model loading"<<std::endl;
  if (!model.load(file)) {
    cerr << "[ERROR] Unable to read model " << filename << endl;
    exit(-1);
  }

  if (border_to_skip>0) sadl::Tensor<T>::skip_border=true;
  std::vector<sadl::Tensor<T>> inputs=model.getInputsTemplate();
  if (inputs.size()==0) {
    cerr<<"[ERROR] missing inputs information (model or prm string)"<<endl;
    exit(-1);
  }
  std::vector<float> output;  // not a tensor to be generic with integer network

  sadl::Dimensions dim_out;
  if (!readResults<T>(filename_results, inputs, output, dim_out)) {
    cerr << "[ERROR] reading result file " << filename_results << endl;
    exit(-1);
  }

  std::cout<<"[INFO] Model initilization"<<std::endl;
  if (!model.init(inputs)) {
    cerr << "[ERROR] issue during initialization" << endl;
    exit(-1);
  }

#if DEBUG_COUNTERS
  model.resetCounters();
#endif
  if (!model.apply(inputs)) {
    cerr << "[ERROR] issue during inference" << endl;
    exit(-1);
  }
  if (border_to_skip) std::cout<<"[INFO] discard border size="<<model.result().border_skip<<endl;
#if DEBUG_COUNTERS
  std::cout<<"\n[INFO] Complexity assessment"<<std::endl;
  auto stat=model.printOverflow(true);
  std::cout<<"[INFO] Total number of operations: "<<stat.second<<std::endl;
  std::cout<<"[INFO] ---------------------------------"<<std::endl;
#endif
  if (!(dim_out.nbElements() == model.result().dims().nbElements())) { // to fix: should be same dimensions
    cerr << "[ERROR] invalid output dimension" << dim_out<< " vs " << model.result().dims()<<endl;
    return;
  }
  
  std::cout<<"\n[INFO] Error assessment"<<std::endl;
  if (!checkResults<T>(output, model.result(), max_e,border_to_skip)) {
    exit(-1);
  }
}
}  // namespace

int main(int argc, char **argv) {
  if (argc!=3) {
    std::cout<<"[ERROR] sample filename_model filename_results"<<std::endl;
    return 1;
  }

  const string filename_model = argv[1];
  const string filename_results = argv[2];
  const double e_max = 0.001;
  const int border_to_skip=0;
  verbose=0;
  

  sadl::layers::TensorInternalType::Type type_model = getModelType(filename_model);
  switch (type_model) {
    case sadl::layers::TensorInternalType::Float:
      infer<float>(filename_results, filename_model, e_max,border_to_skip);
      break;
    case sadl::layers::TensorInternalType::Int32:
      infer<int32_t>(filename_results, filename_model, e_max,border_to_skip);
      break;
    case sadl::layers::TensorInternalType::Int16:
      infer<int16_t>(filename_results, filename_model, e_max,border_to_skip);
      break;
    default:
      cerr << "[ERROR] unsupported type" << endl;
      exit(-1);
  }

  return 0;
}
