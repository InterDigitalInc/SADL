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

// build options
// behavior
#ifndef SATURATE_RESULT
#define SATURATE_RESULT     1 // avoid overflow in int NN
#endif
// optimization
// nothing/-msse42: no simd
// -mavx2:  avx2
// -mavx2 -mfma: avx2 + fuse multiply/add
// -mavx512bw -mavx512f: avx512
// #define NDEBUG        1 // remove sanity tests

// debug
// #define DEBUG_VALUES        1 // show values
// #define DEBUG_MODEL         1 // show pb with model
// #define DEBUG_COUNTERS      1 // print overflow etc.
// #define DEBUG_PRINT         1 // print model info
// #define DEBUG_SIMD          1 // tell about non simd version
// #define DEBUG_KEEP_OUTPUT   1 // keep a copy of the output tensor
#if SATURATE_RESULT
#define SATURATE(X) if (!std::is_same<T,float>::value) X = (X>ComputationType<T>::max)?ComputationType<T>::max:(X<-ComputationType<T>::max?-ComputationType<T>::max:X)
#else
#define SATURATE(X)
#endif

#if DEBUG_COUNTERS
template<typename T> T my_abs(T x) { return x<T{}?-x:x; }
#define COUNTERS(X)  ++this->cpt_op; if (my_abs(X) > ComputationType<T>::max) ++this->cpt_overflow
#define COUNTERS_MAC(X)  ++this->cpt_mac; if (X!=0) ++this->cpt_mac_nz
#else
#define COUNTERS(X) (void)X
#define COUNTERS_MAC(X)  (void)X
#endif


#if DEBUG_MODEL || DEBUG_VALUES || DEBUG_COUNTERS || !NDEBUG
#ifndef DEBUG_PRINT
#define DEBUG_PRINT 1
#endif
#endif


#ifndef DUMP_MODEL_EXT
#define DUMP_MODEL_EXT
#endif
namespace sadl {
enum class Version { unknown=-1, sadl01=2 };
}
