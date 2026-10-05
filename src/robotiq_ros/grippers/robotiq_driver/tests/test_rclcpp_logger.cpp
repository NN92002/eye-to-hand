// Copyright (c) 2026 Robotiq, Inc.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rcutils/logging.h>

#include <cstdarg>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include <robotiq_driver/rclcpp_logger.hpp>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

namespace robotiq_driver::test {
namespace {
struct CapturedLine
{
   int severity = 0;
   std::string message;
};

std::vector<CapturedLine>& captured()
{
   static std::vector<CapturedLine> lines;
   return lines;
}

// rcutils hands the formatted line to a C callback with no user data, hence
// the file-scope sink above.
void capture(const rcutils_log_location_t* /*location*/,
             int severity,
             const char* /*name*/,
             rcutils_time_point_value_t /*timestamp*/,
             const char* format,
             va_list* args)
{
   char buffer[1024];
   va_list copy;
   va_copy(copy, *args);
   vsnprintf(buffer, sizeof(buffer), format, copy);
   va_end(copy);
   captured().push_back({severity, buffer});
}

//! Installs the capturing sink for the duration of a test and restores the
//! previous sink afterwards.
class LogCapture
{
public:
   LogCapture()
      : previous_{rcutils_logging_get_output_handler()}
   {
      captured().clear();
      // Debug lines are filtered out before the sink unless the logger asks
      // for them, which would make the Debug case untestable.
      EXPECT_EQ(RCUTILS_RET_OK, rcutils_logging_set_logger_level(kLoggerName, RCUTILS_LOG_SEVERITY_DEBUG));
      rcutils_logging_set_output_handler(&capture);
   }

   ~LogCapture() { rcutils_logging_set_output_handler(previous_); }

   LogCapture(const LogCapture&) = delete;
   LogCapture& operator=(const LogCapture&) = delete;

   static constexpr const char* kLoggerName = "test_rclcpp_logger";

private:
   rcutils_logging_output_handler_t previous_;
};

RclcppLogger make_logger()
{
   return RclcppLogger{rclcpp::get_logger(LogCapture::kLoggerName)};
}
} // namespace

TEST(RclcppLoggerTest, ForwardsEachLevelAtTheMatchingSeverity)
{
   const LogCapture capture;
   RclcppLogger logger = make_logger();

   logger.log(Robotiq::Logger::Level::Debug, "a debug line");
   logger.log(Robotiq::Logger::Level::Info, "an info line");
   logger.log(Robotiq::Logger::Level::Warn, "a warn line");
   logger.log(Robotiq::Logger::Level::Error, "an error line");

   ASSERT_THAT(captured(), testing::SizeIs(4));
   EXPECT_EQ(RCUTILS_LOG_SEVERITY_DEBUG, captured()[0].severity);
   EXPECT_EQ(RCUTILS_LOG_SEVERITY_INFO, captured()[1].severity);
   EXPECT_EQ(RCUTILS_LOG_SEVERITY_WARN, captured()[2].severity);
   EXPECT_EQ(RCUTILS_LOG_SEVERITY_ERROR, captured()[3].severity);
}

TEST(RclcppLoggerTest, ForwardsTheMessageVerbatim)
{
   const LogCapture capture;
   RclcppLogger logger = make_logger();
   const std::string msg = "exchange cycle failed: timed out";

   logger.log(Robotiq::Logger::Level::Info, msg);

   ASSERT_THAT(captured(), testing::SizeIs(1));
   EXPECT_EQ(msg, captured()[0].message);
}

TEST(RclcppLoggerTest, TreatsTheMessageAsTextRatherThanAFormatString)
{
   // SDK messages quote wire data and error strings; a stray %s in one must
   // not be interpreted as a conversion and read off the end of the varargs.
   const LogCapture capture;
   RclcppLogger logger = make_logger();

   logger.log(Robotiq::Logger::Level::Warn, "unexpected response: %s %d %n");

   ASSERT_THAT(captured(), testing::SizeIs(1));
   EXPECT_EQ("unexpected response: %s %d %n", captured()[0].message);
}

TEST(RclcppLoggerTest, HonoursTheExtentOfANonTerminatedView)
{
   // Robotiq::Logger takes a string_view, which need not be null-terminated;
   // only its extent may be printed.
   const LogCapture capture;
   RclcppLogger logger = make_logger();

   const std::string backing = "activated and holding";
   logger.log(Robotiq::Logger::Level::Info, std::string_view{backing}.substr(0, 9));

   ASSERT_THAT(captured(), testing::SizeIs(1));
   EXPECT_EQ("activated", captured()[0].message);
}
} // namespace robotiq_driver::test
