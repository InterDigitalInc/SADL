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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <list>

#define DUMP_MODEL_EXT virtual bool dump(std::ostream &file)
#define DEBUG_KEEP_OUTPUT 1
// trick to access inner data
#define private public
#define protected public
#include <sadl/model.h>
#undef private
#undef protected
#define private private
#define protected public

#include "helper.h"
#include "dumper.h"
#include "copy.h"

using namespace std;

namespace
{
constexpr int kNoQValue=-1000;

bool toQuantize(sadl::layers::OperationType::Type type) {
    // negative logic
    return type  !=  sadl::layers::OperationType::Add &&
            type  !=  sadl::layers::OperationType::BiasAdd &&
            type  !=  sadl::layers::OperationType::Concat &&
            type !=  sadl::layers::OperationType::Copy &&
            type !=  sadl::layers::OperationType::Expand &&
            type !=  sadl::layers::OperationType::Flatten &&
            type !=  sadl::layers::OperationType::Identity &&
            type !=  sadl::layers::OperationType::MaxPool &&
            type !=  sadl::layers::OperationType::Relu &&
            type !=  sadl::layers::OperationType::Reshape &&
            type !=  sadl::layers::OperationType::Shape &&
            type !=  sadl::layers::OperationType::Transpose;
}


template <typename T>
void quantizeTensor(const sadl::Tensor<float> &B, sadl::Tensor<T> &Bq) {
    double Q = (1 << Bq.quantizer);
    for (int k = 0; k < B.size(); ++k) {
        double z = round(B[k] * Q);
        if (z <= -numeric_limits<T>::max()) {
            z = -numeric_limits<T>::max() + 1;
        }
        if (z >= numeric_limits<T>::max()) {
            z = numeric_limits<T>::max() - 1;
        }
        Bq[k] = (T)z;
    }
}

template <typename T>
void quantize(sadl::layers::Layer<T> &layerQ, const sadl::layers::Layer<float> &layer_float, int quantizer) {

    // layers with internal quantizer
    if (layerQ.op() == sadl::layers::OperationType::Conv2D) dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).q_ = quantizer;
    else if (layerQ.op() == sadl::layers::OperationType::MatMul) dynamic_cast<sadl::layers::MatMul<T> &>(layerQ).q_ = quantizer;
    else if (layerQ.op() == sadl::layers::OperationType::Mul) dynamic_cast<sadl::layers::Mul<T> &>(layerQ).q_ = quantizer;
    else if (layerQ.op() == sadl::layers::OperationType::Placeholder) dynamic_cast<sadl::layers::Placeholder<T> &>(layerQ).q_ = quantizer;
    else if (layerQ.op() == sadl::layers::OperationType::Const) {
        layerQ.out_.quantizer = quantizer;
        quantizeTensor(layer_float.out_, layerQ.out_);
    } else {
        cerr << "[ERROR] unsupported layer " << sadl::layers::opName(layerQ.op()) << endl;
        exit(-1);
    }
}


template<typename T> void quantize(const string &filename,const string &filename_out,const std::vector<int> &quantizers)
{
    // load float model
    sadl::layers::TensorInternalType::Type type_model = getModelType(filename);
    if (type_model!=sadl::layers::TensorInternalType::Float) {
        std::cerr<<"[ERROR] please input a float model"<<std::endl;
        exit(-1);
    }

    sadl::Model<float> model;
    ifstream       file(filename, ios::binary);
    cout << "[INFO] Model loading" << endl;
    if (!model.load(file))
    {
        cerr << "[ERROR] Unable to read model " << filename << endl;
        exit(-1);
    }

    // init quantize model
    sadl::Model<T> modelQ;
    if (!copy(model, modelQ)) {
        cerr << "[ERROR] Unable to copy model " << endl;
        exit(-1);
    }

    // we need to set the placeholders layers (input layers) size because init is not done
    auto inputs=model.getInputsTemplate();
    std::vector<sadl::Tensor<T>> inputsQ{inputs.size()};
    for (int s = 0; s < (int)inputsQ.size(); ++s) {
        inputsQ[s].resize(inputs[s].dims());
    }
    int cpt = 0;
    for (auto &id_input: modelQ.ids_input) {
        auto &L = modelQ.getLayer(id_input);
        if (L.layer->op() == sadl::layers::OperationType::Placeholder) {
            assert(cpt<(int)inputs.size());
            std::vector<sadl::Tensor<T> *> v = {&inputsQ[cpt]};
            ++cpt;
            L.layer->init(v);
        }
    }
    // quantize each layer + set quantizer
    for (int k=0;k<(int)modelQ.data_.size();++k) {  //
        auto &layer=*modelQ.data_[k].layer;
        if (toQuantize(layer.op())) {
            if (layer.id()>=(int)quantizers.size()||quantizers[layer.id()]==kNoQValue) {
                std::cerr << "[ERROR] need a quantizer for layer " << layer.id() <<" op="<<sadl::layers::opName(layer.op())<<" name="<<layer.name()<<std::endl;
                exit(-1);
            }
            int q=quantizers[layer.id()];
            quantize<T>(layer, *model.getLayer(layer.id()).layer, q);
        }
    }

    // dump to file
    ofstream file_out(filename_out, ios::binary);
    modelQ.dump(file_out);
    cout << "[INFO] quantize model in " << filename_out << endl;

}

}   // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "[ERROR] quantize filename_model_float filename_model_int16" << endl;
        return 1;
    }

    const string filename_model = argv[1];
    const string filename_model_out = argv[2];
    // get the list of quantizers
    std::cout<<"For layers needing a quantization, the original value x (in float) is replaced by X=round(x*2^N)\n"
               "Enter the list of pairs: layer id and N (EOF to finish)"<<std::endl;
    int id,N;
    int max_id=0;
    std::list<pair<int,int>> id_q;
    while(std::cin>>id>>N) {
        id_q.push_back({id,N});
        max_id=max(max_id,id);
    }
    std::vector<int> quantizers;
    quantizers.resize(max_id+1);
    fill(quantizers.begin(),quantizers.end(),kNoQValue);
    for(auto x: id_q) {
        quantizers[x.first]=x.second;
    }
    quantize<int16_t>(filename_model,filename_model_out,quantizers);

    return 0;
}
