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

// Unit tests for the AHRS fusion helpers (robotiq_tsf/fusion.hpp): per-finger
// MCU-timestamp dt derivation (with its [lo, hi] clamp), still-detection and
// bias-EMA trim. These regressed silently once (the SDK node integrated on an
// unbounded host-clock dt with a frozen gyro bias), so they're pinned here.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "eigen_test_utils.hpp"
#include "robotiq_tsf/fusion.hpp"

using Eigen::Vector3f;
using robotiq_tsf::deriveDt;
using robotiq_tsf::sampleIsStill;
using robotiq_tsf::trimBias;

namespace {
constexpr float kLo = 1e-4f; // 0.1 ms
constexpr float kHi = 0.1f; // 100 ms
} // namespace

TEST(DeriveDt, NormalDeltaIsMsToSeconds)
{
   // 10 ms apart -> 0.01 s, within [lo, hi].
   EXPECT_FLOAT_EQ(deriveDt(1000, 1010, kLo, kHi), 0.010f);
}

TEST(DeriveDt, FirstSampleAfterSeedReturnsZero)
{
   // prev == 0 means "not yet seeded" -> skip integration.
   EXPECT_FLOAT_EQ(deriveDt(0, 1234, kLo, kHi), 0.0f);
}

TEST(DeriveDt, DuplicateTimestampReturnsZero)
{
   EXPECT_FLOAT_EQ(deriveDt(1000, 1000, kLo, kHi), 0.0f);
}

TEST(DeriveDt, BackwardsTimestampReturnsZero)
{
   EXPECT_FLOAT_EQ(deriveDt(2000, 1000, kLo, kHi), 0.0f);
}

TEST(DeriveDt, StalledDeltaIsClampedToHi)
{
   // 500 ms stall -> capped at 100 ms so a delayed sample can't inject an
   // outsized integration step.
   EXPECT_FLOAT_EQ(deriveDt(1000, 1500, kLo, kHi), kHi);
}

TEST(SampleIsStill, TrueWhenGyroSmallAndAccelNearOneG)
{
   EXPECT_TRUE(sampleIsStill({0.1f, -0.1f, 0.05f}, {0.0f, 0.0f, 1.0f}, 0.8f, 0.05f));
}

TEST(SampleIsStill, FalseWhenRotating)
{
   EXPECT_FALSE(sampleIsStill({5.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.8f, 0.05f));
}

TEST(SampleIsStill, FalseUnderLinearAcceleration)
{
   // |a| well away from 1 g -> not stationary even with zero angular rate.
   EXPECT_FALSE(sampleIsStill({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.5f}, 0.8f, 0.05f));
}

TEST(TrimBias, NudgesTowardResidualByAlpha)
{
   // bias 0.10, residual 0.20, alpha 0.5 -> 0.10 + 0.5 * 0.20 = 0.20,
   // independently per axis.
   const Vector3f out = trimBias({0.10f, 0.0f, -0.10f}, {0.20f, 0.0f, 0.20f}, 0.5f);
   robotiq_tsf::test::expectEigenFloatEq(out, Vector3f(0.20f, 0.0f, 0.0f));
}

TEST(TrimBias, ZeroResidualLeavesBiasUnchanged)
{
   const Vector3f bias(0.42f, -0.17f, 0.03f);
   robotiq_tsf::test::expectEigenFloatEq(trimBias(bias, Vector3f::Zero(), 0.0005f), bias);
}
