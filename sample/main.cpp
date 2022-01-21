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

#include <sadl/model.h>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;

namespace {
sadl::layers::TensorInternalType::Type getModelType(const string &filename) {
  const string MAGICNUMBER="SADL0001";
  ifstream file(filename, ios::binary);
  char magic[9];
  file.read(magic, 8);
  magic[8] = '\0';
  string magic_s = magic;
  if (magic_s != MAGICNUMBER) {
    cerr << "[ERROR] Pb reading model: wrong magic " << magic_s << endl;
    exit(-1);
  }

  int8_t x = 0;
  file.read((char *)&x, sizeof(int8_t));
  return (sadl::layers::TensorInternalType::Type)x;
}
//   output_dim_size [int] # [1-4]
//   dim_output[k]   [int[output_dim_size]]
//   output[k]       [float[nb_elts]]         # nb_elts=product(dim_output[k])
template<typename T>
void infer(const string &filename) {
  const int border_to_skip=0;

  sadl::Model<T> model;
  ifstream file(filename, ios::binary);
  cout<<"[INFO] Model loading"<<endl;
  if (!model.load(file)) {
    cerr << "[ERROR] Unable to read model " << filename << endl;
    exit(-1);
  }

  if (border_to_skip>0) sadl::Tensor<T>::skip_border=true;
  vector<sadl::Tensor<T>> inputs=model.getInputsTemplate();
  cout<<"[INFO] Model initilization"<<endl;

  if (!model.init(inputs)) {
    cerr << "[ERROR] issue during initialization" << endl;
    exit(-1);
  }

#if DEBUG_COUNTERS
  model.resetCounters();
#endif

  // fill input with values from -1 to 1
  double step=(1.+1.)/(inputs[0].size()-1);
  double x0=-1.;
  for(auto &t: inputs)
    for(auto &x: t) { x=x0; x0+=step; }
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  if (!model.apply(inputs)) {
    cerr << "[ERROR] issue during inference" << endl;
    exit(-1);
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> dt = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "[INFO] "<< dt.count() * 1000. << " ms"<<endl;

  if (border_to_skip) cout<<"[INFO] discard border size="<<model.result().border_skip<<endl;

#if DEBUG_COUNTERS
  cout<<"\n[INFO] Complexity assessment"<<endl;
  auto stat=model.printOverflow(true);
  cout << "[INFO] " << stat.overflow << " overflow" << endl;
  cout << "[INFO] " << stat.op << " OPs" << endl;
  cout << "[INFO] " << stat.mac << " MACs" << endl;
  cout << "[INFO] " << stat.mac_nz << " MACs non 0" << endl;
  cout<<"[INFO] ---------------------------------"<<endl;
#endif

#if DEBUG_PRINT
  const int N=model.getIdsOutput().size();
  for(int i=0;i<N;++i) cout<<"[INFO] output "<<i<<'\n'<<model.result(i)<<endl;
#endif
  }
  
}  // namespace

int main(int argc, char **argv) {
  if (argc!=2) {
    cout<<"[ERROR] sample filename_model"<<endl;
    return 1;
  }

  const string filename_model = argv[1];
  

  sadl::layers::TensorInternalType::Type type_model = getModelType(filename_model);
  switch (type_model) {
    case sadl::layers::TensorInternalType::Float:
      infer<float>(filename_model);
      break;
    case sadl::layers::TensorInternalType::Int32:
      infer<int32_t>(filename_model);
      break;
    case sadl::layers::TensorInternalType::Int16:
      infer<int16_t>(filename_model);
      break;
    default:
      cerr << "[ERROR] unsupported type" << endl;
      exit(-1);
  }

  return 0;
}
