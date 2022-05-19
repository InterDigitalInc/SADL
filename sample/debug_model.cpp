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

#define DEBUG_VALUES        1 // show values
#define DEBUG_MODEL         1 // show pb with model
#define DEBUG_PRINT         1 // print model info
#define DEBUG_SIMD          1 // tell about non simd version

#include <sadl/model.h>
#include <cmath>
#include <fstream>
#include <chrono>
#include "helper.h"

using namespace std;

namespace
{

template<typename T> void infer(const string &filename)
{
  sadl::Model<T> model;
  ifstream       file(filename, ios::binary);
  cout << "[INFO] Model loading" << endl;
  if (!model.load(file))
  {
    cerr << "[ERROR] Unable to read model " << filename << endl;
    exit(-1);
  }

  //  sadl::Tensor<T>::skip_border = true;
  vector<sadl::Tensor<T>> inputs = model.getInputsTemplate();

  cout << "[INFO] Model initilization" << endl;

  if (!model.init(inputs))
  {
    cerr << "[ERROR] issue during initialization" << endl;
    exit(-1);
  }

  if (!model.apply(inputs))
  {
    cerr << "[ERROR] issue during inference" << endl;
    exit(-1);
  }

  if (sadl::Tensor<T>::skip_border)
    cout << "[INFO] discard border size=" << model.result().border_skip << endl;

  const int N = model.getIdsOutput().size();
  for (int i = 0; i < N; ++i)
    cout << "[INFO] output " << i << '\n' << model.result(i) << endl;
}

}   // namespace

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cout << "[ERROR] sample filename_model" << endl;
    return 1;
  }

  const string filename_model = argv[1];

  sadl::layers::TensorInternalType::Type type_model = getModelType(filename_model);
  switch (type_model)
  {
  case sadl::layers::TensorInternalType::Float: infer<float>(filename_model); break;
  case sadl::layers::TensorInternalType::Int32: infer<int32_t>(filename_model); break;
  case sadl::layers::TensorInternalType::Int16: infer<int16_t>(filename_model); break;
  default: cerr << "[ERROR] unsupported type" << endl; exit(-1);
  }

  return 0;
}
