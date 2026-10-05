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

// Unit tests for sdk_bridge: SDK Fingers sample -> aggregate Sensor message
// field mapping (taxels, dynamic, accelerometer, gyroscope, timestamps).

#include <gtest/gtest.h>

#include <cstdint>

#include "robotiq_tsf/sdk_bridge.hpp"

namespace {

Fingers makeSample()
{
   Fingers fingers{};
   for(int f = 0; f < FINGER_COUNT; ++f)
   {
      FingerData& d = fingers.finger[f];
      for(int i = 0; i < FINGER_STATIC_TACTILE_COUNT; ++i)
      {
         d.staticTactile[i] = static_cast<uint16_t>(1000 * (f + 1) + i);
      }
      d.dynamicTactile[0] = static_cast<int16_t>(f == 0 ? -123 : 456);
      for(int i = 0; i < 3; ++i)
      {
         d.accelerometer[i] = static_cast<int16_t>(-100 * (f + 1) - i);
         d.gyroscope[i] = static_cast<int16_t>(200 * (f + 1) + i);
      }
      d.timestamp = 40000 + f;
   }
   return fingers;
}

TEST(SdkBridge, MapsAllFields)
{
   const Fingers fingers = makeSample();
   robotiq_tsf::msg::Sensor out;
   robotiq_tsf::fillSensorMessages(fingers, out);

   for(int f = 0; f < FINGER_COUNT; ++f)
   {
      for(int i = 0; i < FINGER_STATIC_TACTILE_COUNT; ++i)
      {
         EXPECT_EQ(out.staticdata.taxels[f].values[i], 1000 * (f + 1) + i) << "finger " << f << " taxel " << i;
      }
      EXPECT_EQ(out.dynamic.data[f].value, f == 0 ? -123 : 456);
      for(int i = 0; i < 3; ++i)
      {
         EXPECT_EQ(out.accelerometer.data[f].values[i], -100 * (f + 1) - i);
         EXPECT_EQ(out.gyroscope.data[f].values[i], 200 * (f + 1) + i);
      }
      EXPECT_EQ(out.timestamp.values[f], 40000 + f);
   }
}

TEST(SdkBridge, OverwritesPreviousSample)
{
   robotiq_tsf::msg::Sensor out;
   robotiq_tsf::fillSensorMessages(makeSample(), out);

   Fingers zeroed{};
   robotiq_tsf::fillSensorMessages(zeroed, out);
   for(int f = 0; f < FINGER_COUNT; ++f)
   {
      for(int i = 0; i < FINGER_STATIC_TACTILE_COUNT; ++i)
      {
         EXPECT_EQ(out.staticdata.taxels[f].values[i], 0);
      }
      EXPECT_EQ(out.dynamic.data[f].value, 0);
      EXPECT_EQ(out.timestamp.values[f], 0);
   }
}

TEST(SdkBridge, NegativeImuValuesSurvive)
{
   Fingers fingers{};
   fingers.finger[0].accelerometer[0] = -32768;
   fingers.finger[0].gyroscope[2] = -1;
   fingers.finger[1].dynamicTactile[0] = -32768;

   robotiq_tsf::msg::Sensor out;
   robotiq_tsf::fillSensorMessages(fingers, out);
   EXPECT_EQ(out.accelerometer.data[0].values[0], -32768);
   EXPECT_EQ(out.gyroscope.data[0].values[2], -1);
   EXPECT_EQ(out.dynamic.data[1].value, -32768);
}

} // namespace
