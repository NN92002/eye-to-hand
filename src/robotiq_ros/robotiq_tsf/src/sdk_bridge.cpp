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

#include "robotiq_tsf/sdk_bridge.hpp"

#include <cstring>

namespace robotiq_tsf {

void fillSensorMessages(const Fingers& fingers, msg::Sensor& out)
{
   // The memcpy below is only safe while the ROS message arrays and the SDK
   // finger arrays are the same size. Fail the build if either side changes.
   static_assert(sizeof(out.staticdata.taxels[0].values) == sizeof(fingers.finger[0].staticTactile),
                 "StaticData taxel array size != SDK staticTactile array");
   static_assert(sizeof(out.accelerometer.data[0].values) == sizeof(fingers.finger[0].accelerometer),
                 "Accelerometer array size != SDK accelerometer array");
   static_assert(sizeof(out.gyroscope.data[0].values) == sizeof(fingers.finger[0].gyroscope),
                 "Gyroscope array size != SDK gyroscope array");

   for(int f = 0; f < FINGER_COUNT; ++f)
   {
      const FingerData& finger = fingers.finger[f];

      std::memcpy(out.staticdata.taxels[f].values.data(), finger.staticTactile, sizeof(finger.staticTactile));

      out.dynamic.data[f].value = finger.dynamicTactile[0];

      std::memcpy(out.accelerometer.data[f].values.data(), finger.accelerometer, sizeof(finger.accelerometer));

      std::memcpy(out.gyroscope.data[f].values.data(), finger.gyroscope, sizeof(finger.gyroscope));

      out.timestamp.values[f] = finger.timestamp;
   }
}

} // namespace robotiq_tsf
