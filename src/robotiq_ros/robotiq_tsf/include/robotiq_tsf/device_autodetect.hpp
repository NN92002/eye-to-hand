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

#include <string>

namespace robotiq_tsf {

// Default udev-symlink path used by sensor_install.sh.
constexpr const char* kDefaultSensorDevice = "/dev/rq_tsf85_0";

// Search the system for a connected Robotiq tactile sensor. Tries, in order:
//   1. /dev/rq_tsf85_* udev symlinks
//   2. /dev/ttyACM* and /dev/ttyUSB* nodes whose USB descriptor strings
//      match a known sensor manufacturer/product.
// Returns the absolute device path on success, or an empty string if nothing
// matched.
std::string autoDetectSensorDevice();

// Testable core of the above: scan `dev_root` for the candidate device nodes
// and match their USB descriptor strings under `sysfs_tty_root` (the
// /sys/class/tty equivalent).
std::string autoDetectSensorDevice(const std::string& dev_root, const std::string& sysfs_tty_root);

// True if `device` refers to a path that currently exists on the filesystem.
bool deviceExists(const std::string& device);

// Internals exposed for unit tests.
namespace detail {
std::string trimWhitespace(const std::string& value);
bool matchesKnownUsbString(const std::string& value);
bool deviceMatchesKnownSensor(const std::string& device_path, const std::string& sysfs_tty_root);
} // namespace detail

} // namespace robotiq_tsf
