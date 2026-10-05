// Copyright (c) 2026 Robotiq
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

// gtest helpers for comparing Eigen dense objects coefficient-wise.
//
//   expectEigenFloatEq(actual, Eigen::Vector3f(0.2f, 0.0f, 0.0f));
//
// Each coefficient is compared with EXPECT_FLOAT_EQ; a failure names the
// offending coefficient. Failures are reported at this header's line — wrap
// the call in SCOPED_TRACE if a test has several calls to tell apart.

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace robotiq_tsf::test {

template <typename DerivedA, typename DerivedB>
void expectEigenFloatEq(const Eigen::DenseBase<DerivedA>& actual, const Eigen::DenseBase<DerivedB>& expected)
{
   ASSERT_EQ(actual.rows(), expected.rows());
   ASSERT_EQ(actual.cols(), expected.cols());
   for(Eigen::Index r = 0; r < actual.rows(); ++r)
   {
      for(Eigen::Index c = 0; c < actual.cols(); ++c)
      {
         EXPECT_FLOAT_EQ(actual.derived().coeff(r, c), expected.derived().coeff(r, c))
            << "coefficient (" << r << ", " << c << ") of\n"
            << actual.derived() << "\nvs expected\n"
            << expected.derived();
      }
   }
}

} // namespace robotiq_tsf::test
