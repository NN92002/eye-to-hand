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

// All math is done in float, matching the filter under test: the Madgwick
// reference implementation is float32 (embedded origin), and the IMU samples
// are 16-bit (±2 g / ±250 deg/s), so sensor noise and bias sit far above
// float32 rounding — double would add no accuracy. The quaternion is
// renormalized every update, so float error does not accumulate either.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

#include "robotiq_tsf/MadgwickAHRS.h"

using Eigen::AngleAxisf;
using Eigen::Quaternionf;
using Eigen::Vector3f;

namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kDegToRad = kPi / 180.0f;

// Euler angles with an analytically-known expected value: the actual value
// goes through a handful of float32 trig ops, so the real error is ~1e-5 deg;
// 1e-3 deg keeps margin while still catching any sign/axis/order mistake.
constexpr float kExactTolDeg = 1e-3f;

// Individual components of a unit quaternion: a few float32 multiply-adds
// away from exact, so near machine epsilon.
constexpr float kQuatTol = 1e-6f;

// Filter attitude after integrating 1 s of motion: dominated by algorithmic
// error (discrete integration + gradient-descent correction), not rounding.
constexpr float kTrackingTolDeg = 3.0f;

// Attitude change while accel is gated and gyro is zero: the update must be
// a no-op, so anything beyond float dust is a gating failure.
constexpr float kGateLeakTolDeg = 0.1f;

struct EulerAngles
{
   float roll;
   float pitch;
   float yaw;
};

EulerAngles eulerDegOf(const Quaternionf& q)
{
   EulerAngles e;
   quatToEulerDeg(q, e.roll, e.pitch, e.yaw);
   return e;
}

EulerAngles eulerDegOf(const MadgwickFilter& f)
{
   EulerAngles e;
   f.getEulerDeg(e.roll, e.pitch, e.yaw);
   return e;
}

float maxAbsDeg(const EulerAngles& e)
{
   return std::max(std::fabs(e.roll), std::max(std::fabs(e.pitch), std::fabs(e.yaw)));
}

::testing::AssertionResult eulerNear(const EulerAngles& actual, const EulerAngles& expected, float tol_deg)
{
   if(std::fabs(actual.roll - expected.roll) <= tol_deg && std::fabs(actual.pitch - expected.pitch) <= tol_deg
      && std::fabs(actual.yaw - expected.yaw) <= tol_deg)
   {
      return ::testing::AssertionSuccess();
   }
   return ::testing::AssertionFailure() << "(roll, pitch, yaw) = (" << actual.roll << ", " << actual.pitch << ", "
                                        << actual.yaw << ") deg, expected (" << expected.roll << ", " << expected.pitch
                                        << ", " << expected.yaw << ") deg +/- " << tol_deg;
}

TEST(QuatHelpers, EulerFromSingleAxisQuaternions)
{
   constexpr float kRollDeg = 30.0f;
   constexpr float kPitchDeg = 45.0f;
   constexpr float kYawDeg = -60.0f;

   Quaternionf q(AngleAxisf(kRollDeg * kDegToRad, Vector3f::UnitX()));
   EXPECT_TRUE(eulerNear(eulerDegOf(q), {kRollDeg, 0.0f, 0.0f}, kExactTolDeg));

   q = AngleAxisf(kPitchDeg * kDegToRad, Vector3f::UnitY());
   EXPECT_TRUE(eulerNear(eulerDegOf(q), {0.0f, kPitchDeg, 0.0f}, kExactTolDeg));

   q = AngleAxisf(kYawDeg * kDegToRad, Vector3f::UnitZ());
   EXPECT_TRUE(eulerNear(eulerDegOf(q), {0.0f, 0.0f, kYawDeg}, kExactTolDeg));
}

TEST(QuatHelpers, EulerFromComposedZyxRotation)
{
   // ZYX Tait-Bryan: q = qz(yaw) * qy(pitch) * qx(roll) must decompose back
   // into the same three angles.
   constexpr float kRollDeg = 10.0f;
   constexpr float kPitchDeg = 20.0f;
   constexpr float kYawDeg = 30.0f;

   const Quaternionf q = AngleAxisf(kYawDeg * kDegToRad, Vector3f::UnitZ())
                       * AngleAxisf(kPitchDeg * kDegToRad, Vector3f::UnitY())
                       * AngleAxisf(kRollDeg * kDegToRad, Vector3f::UnitX());

   EXPECT_TRUE(eulerNear(eulerDegOf(q), {kRollDeg, kPitchDeg, kYawDeg}, kExactTolDeg));
}

TEST(QuatHelpers, MulByConjugateGivesRelativeRotation)
{
   // Zeroing as done for relative orientation: q_rel = conj(q_ref) * q.
   constexpr float kRefRollDeg = 30.0f;
   constexpr float kCurRollDeg = 50.0f;
   constexpr float kRelRollDeg = kCurRollDeg - kRefRollDeg;

   const Quaternionf q_ref(AngleAxisf(kRefRollDeg * kDegToRad, Vector3f::UnitX()));
   const Quaternionf q_cur(AngleAxisf(kCurRollDeg * kDegToRad, Vector3f::UnitX()));
   Quaternionf q_rel = q_ref.conjugate() * q_cur;

   EXPECT_TRUE(eulerNear(eulerDegOf(q_rel), {kRelRollDeg, 0.0f, 0.0f}, kExactTolDeg));

   // Self-relative must be identity.
   q_rel = q_cur.conjugate() * q_cur;
   EXPECT_NEAR(q_rel.w(), 1.0f, kQuatTol);
   EXPECT_NEAR(q_rel.x(), 0.0f, kQuatTol);
   EXPECT_NEAR(q_rel.y(), 0.0f, kQuatTol);
   EXPECT_NEAR(q_rel.z(), 0.0f, kQuatTol);
}

TEST(QuatHelpers, HamiltonConventionGoldenComponents)
{
   // Pins the Hamilton product order and right-handed rotation signs of the
   // Eigen types the filter is built on, with exact component values, so a
   // convention regression cannot slip in silently.
   constexpr float kHalfSqrt2 = 0.70710678f;

   const Quaternionf qx(AngleAxisf(0.5f * kPi, Vector3f::UnitX()));
   EXPECT_NEAR(qx.w(), kHalfSqrt2, kQuatTol);
   EXPECT_NEAR(qx.x(), kHalfSqrt2, kQuatTol);
   EXPECT_NEAR(qx.y(), 0.0f, kQuatTol);
   EXPECT_NEAR(qx.z(), 0.0f, kQuatTol);

   const Quaternionf qy(AngleAxisf(0.5f * kPi, Vector3f::UnitY()));
   EXPECT_NEAR(qy.w(), kHalfSqrt2, kQuatTol);
   EXPECT_NEAR(qy.y(), kHalfSqrt2, kQuatTol);

   const Quaternionf qz(AngleAxisf(0.5f * kPi, Vector3f::UnitZ()));
   EXPECT_NEAR(qz.w(), kHalfSqrt2, kQuatTol);
   EXPECT_NEAR(qz.z(), kHalfSqrt2, kQuatTol);

   // Hamilton product: qx(90) * qy(90) = (0.5, 0.5, 0.5, 0.5) exactly.
   const Quaternionf q = qx * qy;
   EXPECT_NEAR(q.w(), 0.5f, kQuatTol);
   EXPECT_NEAR(q.x(), 0.5f, kQuatTol);
   EXPECT_NEAR(q.y(), 0.5f, kQuatTol);
   EXPECT_NEAR(q.z(), 0.5f, kQuatTol);
}

TEST(QuatHelpers, EulerExtractionClampsAtGimbalLock)
{
   // A slightly super-unit quaternion pushes |sin(pitch)| past 1; the
   // extraction must clamp to exactly +/-90 deg and stay finite (no NaN from
   // asin out of domain).
   float roll, pitch, yaw;

   const Quaternionf q_up(0.7071f, 0.0f, 0.7080f, 0.0f); // sinp < -1 -> +90
   quatToEulerDeg(q_up, roll, pitch, yaw);
   EXPECT_NEAR(pitch, 90.0f, kExactTolDeg);
   EXPECT_TRUE(std::isfinite(roll) && std::isfinite(yaw));

   const Quaternionf q_down(0.7071f, 0.0f, -0.7080f, 0.0f); // sinp > 1 -> -90
   quatToEulerDeg(q_down, roll, pitch, yaw);
   EXPECT_NEAR(pitch, -90.0f, kExactTolDeg);
   EXPECT_TRUE(std::isfinite(roll) && std::isfinite(yaw));
}

TEST(MadgwickFilter, InitFromAccelMatchesTilt)
{
   constexpr float kRollDeg = 30.0f;
   constexpr float kPitchDeg = 20.0f;
   constexpr float g = 9.81f; // m/s^2

   MadgwickFilter f;

   // Flat: gravity along body +Z.
   f.initFromAccel(0.0f, 0.0f, 1.0f);
   EXPECT_TRUE(eulerNear(eulerDegOf(f), {0.0f, 0.0f, 0.0f}, kExactTolDeg));

   // Pure roll: gravity_body = (0, sin r, cos r). Units must not matter,
   // so feed m/s^2 rather than g.
   f.initFromAccel(0.0f, g * std::sin(kRollDeg * kDegToRad), g * std::cos(kRollDeg * kDegToRad));
   EXPECT_TRUE(eulerNear(eulerDegOf(f), {kRollDeg, 0.0f, 0.0f}, kExactTolDeg));

   // Pure pitch: gravity_body = (-sin p, 0, cos p).
   f.initFromAccel(-std::sin(kPitchDeg * kDegToRad), 0.0f, std::cos(kPitchDeg * kDegToRad));
   EXPECT_TRUE(eulerNear(eulerDegOf(f), {0.0f, kPitchDeg, 0.0f}, kExactTolDeg));
}

TEST(MadgwickFilter, InitFromAccelCombinedRollPitch)
{
   // Gravity in body frame for roll r, pitch p (ZYX, yaw-free):
   // g_body = (-sin p, sin r * cos p, cos r * cos p).
   constexpr float kRollDeg = 30.0f;
   constexpr float kPitchDeg = 20.0f;
   const float sr = std::sin(kRollDeg * kDegToRad);
   const float cr = std::cos(kRollDeg * kDegToRad);
   const float sp = std::sin(kPitchDeg * kDegToRad);
   const float cp = std::cos(kPitchDeg * kDegToRad);

   MadgwickFilter f;
   f.initFromAccel(-sp, sr * cp, cr * cp);
   EXPECT_TRUE(eulerNear(eulerDegOf(f), {kRollDeg, kPitchDeg, 0.0f}, kExactTolDeg));
}

TEST(MadgwickFilter, InitFromZeroAccelResetsToIdentity)
{
   MadgwickFilter f;
   f.initFromAccel(0.0f, 1.0f, 0.0f); // some non-identity state first
   f.initFromAccel(0.0f, 0.0f, 0.0f);
   float q0, q1, q2, q3;
   f.getQuaternion(q0, q1, q2, q3);
   EXPECT_NEAR(q0, 1.0f, kQuatTol);
   EXPECT_NEAR(q1, 0.0f, kQuatTol);
   EXPECT_NEAR(q2, 0.0f, kQuatTol);
   EXPECT_NEAR(q3, 0.0f, kQuatTol);
}

TEST(MadgwickFilter, NonPositiveDtIsIgnored)
{
   MadgwickFilter f;
   f.initFromAccel(0.0f, 0.0f, 1.0f);
   float before[4], after[4];
   f.getQuaternion(before[0], before[1], before[2], before[3]);
   f.updateIMU(1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 1.0f, 0.0f);
   f.updateIMU(1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 1.0f, -0.01f);
   f.getQuaternion(after[0], after[1], after[2], after[3]);
   for(int i = 0; i < 4; ++i)
   {
      EXPECT_EQ(before[i], after[i]);
   }
}

TEST(MadgwickFilter, TracksRotationWithMeasuredDt)
{
   // 90 deg/s roll for 1 s, sampled at 200 Hz. Accel stays consistent with
   // the true attitude (pure tilt, |a| = 1 g), gyro is exact. The legacy
   // filter integrated every sample as 1/1000 s regardless of the real rate,
   // so it tracked this rotation ~5x too slow — the "drift" in PR #5.
   constexpr float kRateDegS = 90.0f;
   constexpr float kDt = 0.005f;
   constexpr int kSteps = 200;
   constexpr float kExpectedRollDeg = kRateDegS * kDt * kSteps;

   MadgwickFilter f;
   f.initFromAccel(0.0f, 0.0f, 1.0f);
   for(int i = 0; i < kSteps; ++i)
   {
      const float rollTrue = kRateDegS * kDt * static_cast<float>(i) * kDegToRad;
      f.updateIMU(kRateDegS * kDegToRad, 0.0f, 0.0f, 0.0f, std::sin(rollTrue), std::cos(rollTrue), kDt);
   }
   EXPECT_TRUE(eulerNear(eulerDegOf(f), {kExpectedRollDeg, 0.0f, 0.0f}, kTrackingTolDeg));
}

TEST(MadgwickFilter, AccelGateRejectsMotionBursts)
{
   // Stationary sensor, zero rotation, but a sustained linear-acceleration
   // burst (|a| ~ 1.35 g). The legacy filter treated the burst as a tilted
   // gravity vector and walked the attitude away — the "noisy in transition"
   // in PR #5. Accel feedback outside [0.85, 1.15] g must be gated, so the
   // attitude must not move at all.
   constexpr float kDt = 0.005f;
   constexpr int kBurstSteps = 1000; // 5 s of burst at 200 Hz
   // gravity (1 g on Z) + 0.9 g lateral -> |a| ~ 1.35 g, outside the accel gate
   constexpr float kBurstLateralG = 0.9f;

   MadgwickFilter f;
   f.initFromAccel(0.0f, 0.0f, 1.0f);

   float peakDeg = 0.0f;
   for(int i = 0; i < kBurstSteps; ++i)
   {
      f.updateIMU(0.0f, 0.0f, 0.0f, kBurstLateralG, 0.0f, 1.0f, kDt);
      peakDeg = std::max(peakDeg, maxAbsDeg(eulerDegOf(f)));
   }

   EXPECT_LT(peakDeg, kGateLeakTolDeg);
}

TEST(MadgwickFilter, GoldenTrajectoryCheckpoints)
{
   // End-to-end numeric regression: a fixed synthetic IMU stream must keep
   // reproducing the quaternion checkpoints recorded from the pre-Eigen
   // implementation. Guards the library migration against any behavior
   // change beyond float-op reordering: 1e-5 (~80 float32 ulps at unit
   // scale) absorbs reordering drift accumulated over the 350 updates while
   // staying orders of magnitude below a real algorithmic change.
   constexpr float kGoldenTol = 1e-5f;
   constexpr float kDt = 0.005f;
   struct Checkpoint
   {
      const char* tag;
      float q[4];
   };

   MadgwickFilter f;
   f.initFromAccel(0.0f, 0.0f, 1.0f);

   auto expectCheckpoint = [&f](const Checkpoint& c) {
      float q[4];
      f.getQuaternion(q[0], q[1], q[2], q[3]);
      for(int i = 0; i < 4; ++i)
      {
         EXPECT_NEAR(q[i], c.q[i], kGoldenTol) << "checkpoint " << c.tag << " component " << i;
      }
   };

   // Segment A: 100 steps stationary (accel gate open, gradient active).
   for(int i = 0; i < 100; ++i)
   {
      f.updateIMU(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, kDt);
   }
   expectCheckpoint({"A", {1.00000000f, 0.00000000f, 0.00000000f, 0.00000000f}});

   // Segment B: 150 steps rolling at 60 deg/s with attitude-consistent accel.
   for(int i = 0; i < 150; ++i)
   {
      const float roll = 60.0f * kDt * static_cast<float>(i) * kDegToRad;
      f.updateIMU(60.0f * kDegToRad, 0.0f, 0.0f, 0.0f, std::sin(roll), std::cos(roll), kDt);
   }
   expectCheckpoint({"B", {0.92394269f, 0.38253102f, 0.00000000f, 0.00000000f}});

   // Segment C: 100 steps yawing at 45 deg/s while accel is gated
   // (|a| = 1.4 g): pure gyro integration path.
   for(int i = 0; i < 100; ++i)
   {
      f.updateIMU(0.0f, 0.0f, 45.0f * kDegToRad, 0.98f, 0.0f, 1.0f, kDt);
   }
   expectCheckpoint({"C", {0.90618920f, 0.37518126f, -0.07462808f, 0.18025191f}});
}

} // namespace
