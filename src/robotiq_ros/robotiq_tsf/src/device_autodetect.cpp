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

#include "robotiq_tsf/device_autodetect.hpp"

#include <strings.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

namespace robotiq_tsf {
namespace {

bool readSysfsEntry(const fs::path& path, std::string& value)
{
   std::ifstream file(path);
   if(!file.is_open())
   {
      return false;
   }
   std::string line;
   std::getline(file, line);
   value = detail::trimWhitespace(line);
   return !value.empty();
}

// Absolute paths of entries in `dir` whose filename starts with `prefix`,
// sorted for deterministic selection. Empty if `dir` doesn't exist.
std::vector<std::string> entriesWithPrefix(const fs::path& dir, const std::string& prefix)
{
   std::vector<std::string> matches;
   std::error_code ec;
   for(const auto& entry : fs::directory_iterator(dir, ec))
   {
      const std::string name = entry.path().filename().string();
      if(name.compare(0, prefix.size(), prefix) == 0)
      {
         matches.push_back(entry.path().string());
      }
   }
   std::sort(matches.begin(), matches.end());
   return matches;
}

} // namespace

namespace detail {

std::string trimWhitespace(const std::string& value)
{
   const auto start = value.find_first_not_of(" \t\r\n");
   if(start == std::string::npos)
   {
      return std::string();
   }
   const auto end = value.find_last_not_of(" \t\r\n");
   return value.substr(start, end - start + 1);
}

bool matchesKnownUsbString(const std::string& value)
{
   if(value.empty())
   {
      return false;
   }

   static const char* kKnownStrings[] = {"Robotiq Tactile Sensor",
                                         "Robotiq",
                                         "CoRo Tactile Sensor",
                                         "Cypress USB UART"};
   for(const char* expected : kKnownStrings)
   {
      if(strcasecmp(value.c_str(), expected) == 0)
      {
         return true;
      }
   }
   return false;
}

bool deviceMatchesKnownSensor(const std::string& device_path, const std::string& sysfs_tty_root)
{
   const std::string dev_name = fs::path(device_path).filename().string();
   const fs::path device_dir = fs::path(sysfs_tty_root) / dev_name / "device";

   const fs::path candidate_files[] = {device_dir / ".." / "product",
                                       device_dir / "product",
                                       device_dir / ".." / "manufacturer",
                                       device_dir / "manufacturer"};
   for(const auto& file : candidate_files)
   {
      std::string entry;
      if(readSysfsEntry(file, entry) && matchesKnownUsbString(entry))
      {
         return true;
      }
   }
   return false;
}

} // namespace detail

bool deviceExists(const std::string& device)
{
   std::error_code ec;
   return fs::exists(device, ec);
}

std::string autoDetectSensorDevice(const std::string& dev_root, const std::string& sysfs_tty_root)
{
   // Prefer udev symlinks created by sensor_install.sh.
   const auto symlinks = entriesWithPrefix(dev_root, "rq_tsf85_");
   if(!symlinks.empty())
   {
      return symlinks.front();
   }

   // Otherwise scan the raw ttys and match by USB descriptor string.
   for(const std::string& prefix : {"ttyACM", "ttyUSB"})
   {
      for(const std::string& candidate : entriesWithPrefix(dev_root, prefix))
      {
         if(detail::deviceMatchesKnownSensor(candidate, sysfs_tty_root))
         {
            return candidate;
         }
      }
   }
   return std::string();
}

std::string autoDetectSensorDevice()
{
   return autoDetectSensorDevice("/dev", "/sys/class/tty");
}

} // namespace robotiq_tsf
