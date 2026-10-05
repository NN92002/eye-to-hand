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

// Unit tests for device_autodetect: USB-descriptor matching and the
// /dev + /sys/class/tty scan, against fixture directory trees.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "robotiq_tsf/device_autodetect.hpp"

namespace fs = std::filesystem;
using robotiq_tsf::autoDetectSensorDevice;
using robotiq_tsf::deviceExists;
using robotiq_tsf::detail::deviceMatchesKnownSensor;
using robotiq_tsf::detail::matchesKnownUsbString;
using robotiq_tsf::detail::trimWhitespace;

namespace {

void writeFile(const fs::path& path, const std::string& content)
{
   fs::create_directories(path.parent_path());
   std::ofstream f(path);
   f << content;
}

// Fixture building a fake /dev and /sys/class/tty under a temp dir.
class DeviceAutodetectTest : public ::testing::Test
{
protected:
   void SetUp() override
   {
      root_ = fs::temp_directory_path()
            / ("tsf_autodetect_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
      fs::remove_all(root_);
      dev_ = root_ / "dev";
      sysfs_ = root_ / "sys";
      fs::create_directories(dev_);
      fs::create_directories(sysfs_);
   }

   void TearDown() override { fs::remove_all(root_); }

   void addTty(const std::string& name)
   {
      writeFile(dev_ / name, "");
      fs::create_directories(sysfs_ / name / "device");
   }

   // Descriptor read via the "device/../product" candidate.
   void setProduct(const std::string& name, const std::string& product)
   {
      writeFile(sysfs_ / name / "product", product);
   }

   void setManufacturer(const std::string& name, const std::string& m) { writeFile(sysfs_ / name / "manufacturer", m); }

   fs::path root_, dev_, sysfs_;
};

TEST(TrimWhitespace, Behavior)
{
   EXPECT_EQ(trimWhitespace("  Robotiq Tactile Sensor \r\n"), "Robotiq Tactile Sensor");
   EXPECT_EQ(trimWhitespace("no_trim"), "no_trim");
   EXPECT_EQ(trimWhitespace(" \t\r\n"), "");
   EXPECT_EQ(trimWhitespace(""), "");
}

TEST(MatchesKnownUsbString, KnownStrings)
{
   // The TSF-85 enumerates as product "Robotiq Tactile Sensor",
   // manufacturer "Robotiq" (VID:PID 16d0:14cc).
   EXPECT_TRUE(matchesKnownUsbString("Robotiq Tactile Sensor"));
   EXPECT_TRUE(matchesKnownUsbString("Robotiq"));
   // Older units / dev boards.
   EXPECT_TRUE(matchesKnownUsbString("CoRo Tactile Sensor"));
   EXPECT_TRUE(matchesKnownUsbString("Cypress USB UART"));
   // Match is case-insensitive (strcasecmp).
   EXPECT_TRUE(matchesKnownUsbString("robotiq tactile sensor"));
}

TEST(MatchesKnownUsbString, Rejections)
{
   EXPECT_FALSE(matchesKnownUsbString(""));
   EXPECT_FALSE(matchesKnownUsbString("Arduino Uno"));
   EXPECT_FALSE(matchesKnownUsbString("Robotiq Tactile")); // no substring match
   EXPECT_FALSE(matchesKnownUsbString("Robotiq Tactile Sensor X"));
}

TEST_F(DeviceAutodetectTest, MatchesByProduct)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Robotiq Tactile Sensor\n");
   EXPECT_TRUE(deviceMatchesKnownSensor((dev_ / "ttyACM0").string(), sysfs_.string()));
}

TEST_F(DeviceAutodetectTest, MatchesByManufacturerFallback)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Some CDC Gadget");
   setManufacturer("ttyACM0", "Robotiq");
   EXPECT_TRUE(deviceMatchesKnownSensor((dev_ / "ttyACM0").string(), sysfs_.string()));
}

TEST_F(DeviceAutodetectTest, MatchesByDeviceDirProduct)
{
   addTty("ttyACM0");
   // The "device/product" candidate (no ".." hop).
   writeFile(sysfs_ / "ttyACM0" / "device" / "product", "Cypress USB UART");
   EXPECT_TRUE(deviceMatchesKnownSensor((dev_ / "ttyACM0").string(), sysfs_.string()));
}

TEST_F(DeviceAutodetectTest, RejectsUnknownDevice)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Arduino Uno");
   setManufacturer("ttyACM0", "Arduino LLC");
   EXPECT_FALSE(deviceMatchesKnownSensor((dev_ / "ttyACM0").string(), sysfs_.string()));
}

TEST_F(DeviceAutodetectTest, RejectsWhenNoSysfsEntries)
{
   addTty("ttyACM0");
   EXPECT_FALSE(deviceMatchesKnownSensor((dev_ / "ttyACM0").string(), sysfs_.string()));
}

TEST_F(DeviceAutodetectTest, PrefersUdevSymlink)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Robotiq Tactile Sensor");
   writeFile(dev_ / "rq_tsf85_0", "");
   EXPECT_EQ(autoDetectSensorDevice(dev_.string(), sysfs_.string()), (dev_ / "rq_tsf85_0").string());
}

TEST_F(DeviceAutodetectTest, FindsSensorAmongOtherTtys)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Some CDC Gadget");
   addTty("ttyACM1");
   setProduct("ttyACM1", "Robotiq Tactile Sensor");
   addTty("ttyUSB0");
   setProduct("ttyUSB0", "FTDI Bridge"); // e.g. the gripper RS-485 adapter
   EXPECT_EQ(autoDetectSensorDevice(dev_.string(), sysfs_.string()), (dev_ / "ttyACM1").string());
}

TEST_F(DeviceAutodetectTest, FallsBackToTtyUsb)
{
   addTty("ttyUSB0");
   setProduct("ttyUSB0", "Cypress USB UART");
   EXPECT_EQ(autoDetectSensorDevice(dev_.string(), sysfs_.string()), (dev_ / "ttyUSB0").string());
}

TEST_F(DeviceAutodetectTest, ReturnsEmptyWhenNothingMatches)
{
   addTty("ttyACM0");
   setProduct("ttyACM0", "Arduino Uno");
   EXPECT_EQ(autoDetectSensorDevice(dev_.string(), sysfs_.string()), "");
   EXPECT_EQ(autoDetectSensorDevice((root_ / "empty").string(), sysfs_.string()), "");
}

TEST_F(DeviceAutodetectTest, DeviceExists)
{
   writeFile(dev_ / "ttyACM0", "");
   EXPECT_TRUE(deviceExists((dev_ / "ttyACM0").string()));
   EXPECT_FALSE(deviceExists((dev_ / "ttyACM9").string()));
}

} // namespace
