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
#pragma once
#include <vector>
#include <cassert>
#include <initializer_list>
#include <functional>

namespace sadl
{
struct Dimensions
{
  static constexpr int MaxDim = 6;
  using iterator              = int *;
  using const_iterator        = const int *;

  Dimensions() = default;
  Dimensions(std::initializer_list<int> L)
  {
    assert((int) L.size() <= MaxDim);
    s_ = (int) L.size();
    std::copy(L.begin(), L.end(), v_);
  }

  void resize(int s)
  {
    assert(s <= MaxDim);
    s_ = s;
  }
  int            size() const { return s_; }
  int64_t        nbElements() const { return std::accumulate(v_, v_ + s_, (int64_t)1, [](int64_t a,int64_t b) { return a*b; }); }
  int            operator[](int k) const { return v_[k]; }
  int &          operator[](int k) { return v_[k]; }
  iterator       begin() { return v_; }
  iterator       end() { return v_ + s_; }
  const_iterator begin() const { return v_; }
  const_iterator end() const { return v_ + s_; }
  bool           operator==(const Dimensions &d) const { return d.s_ == s_ && std::equal(v_, v_ + s_, d.v_); }
  int            back() const { return v_[s_ - 1]; }

private:
  int v_[MaxDim] = {};
  int s_         = 0;
};

}   // namespace sadl

//#if !NDEBUG
#include <iostream>
namespace sadl
{
inline std::ostream &operator<<(std::ostream &out, const Dimensions &d)
{
  out << "( ";
  for (int k = 0; k < (int) d.size(); ++k)
    out << d[k] << ' ';
  out << ')';
  return out;
}
}   // namespace sadl
