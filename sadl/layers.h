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

#include "layer_placeholder.h"
#include "layer_reshape.h"
#include "layer_const.h"
#include "layer_identity.h"
#include "layer_conv2d.h" // before matmul to get def  of sum8_float
#include "layer_matmul.h"
#include "layer_biasadd.h"
#include "layer_add.h"
#include "layer_relu.h"
#include "layer_copy.h"
#include "layer_maxpool.h"
#include "layer_mul.h"
#include "layer_concat.h"
#include "layer_maximum.h"
#include "layer_leakyrelu.h"
#include "layer_transpose.h"
#include "layer_flatten.h"

namespace sadl {

namespace layers {

#if DEBUG_PRINT
inline std::string opName(const OperationType::Type op) {

#define DIRTYCASEPRINT(X) case OperationType::X: oss<< #X; break
    std::ostringstream oss;
    switch(op) {
      DIRTYCASEPRINT(Copy);
      DIRTYCASEPRINT(Const);
      DIRTYCASEPRINT(Placeholder);
      DIRTYCASEPRINT(Identity);
      DIRTYCASEPRINT(BiasAdd);
      DIRTYCASEPRINT(MaxPool);
      DIRTYCASEPRINT(MatMul);
      DIRTYCASEPRINT(Reshape);
      DIRTYCASEPRINT(Relu);
      DIRTYCASEPRINT(Conv2D);
      DIRTYCASEPRINT(Add);
      DIRTYCASEPRINT(Mul);
      DIRTYCASEPRINT(Concat);
      DIRTYCASEPRINT(Maximum);
      DIRTYCASEPRINT(LeakyRelu);
      DIRTYCASEPRINT(Transpose);
      DIRTYCASEPRINT(Flatten);
    default: oss<<"??"; break;
    }
    return oss.str();
#undef DIRTYCASEPRINT
}
#endif

}

}

