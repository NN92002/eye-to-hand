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

// Regression test pinning the ROS surface of poll_data_sdk_node: the set of
// advertised topics and the service. This is exactly what silently regressed
// once (the TactileSensor/Quaternion topic was dropped in the SDK-node port),
// and the sdk_bridge/device_autodetect gtests don't cover the node's own graph.
//
// The node's publishers/service are created in its constructor, independently
// of startSensor()/waitForStreaming(), so this needs no hardware sensor.

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "robotiq_tsf/poll_data_sdk_node.hpp"

namespace {
// Wait (spinning to let intra-process graph discovery settle) until `node`
// reports all of `expected`, or the timeout elapses. Returns the last snapshot.
template <typename QueryFn>
std::map<std::string, std::vector<std::string>> waitForNames(const rclcpp::Node::SharedPtr& node,
                                                             const std::vector<std::string>& expected,
                                                             QueryFn query)
{
   using namespace std::chrono; // NOLINT(build/namespaces)
   const auto deadline = steady_clock::now() + seconds(10);
   // Executor rather than the free rclcpp::spin_some(node), which is deprecated
   // from Lyrical on.
   rclcpp::executors::SingleThreadedExecutor executor;
   executor.add_node(node);
   std::map<std::string, std::vector<std::string>> names;
   while(steady_clock::now() < deadline)
   {
      executor.spin_some();
      names = query();
      bool all = true;
      for(const auto& e : expected)
      {
         if(names.find(e) == names.end())
         {
            all = false;
            break;
         }
      }
      if(all)
      {
         break;
      }
      std::this_thread::sleep_for(milliseconds(50));
   }
   return names;
}
} // namespace

TEST(PollDataSdkNodeSurface, AdvertisesExpectedTopicsAndService)
{
   rclcpp::init(0, nullptr);
   {
      auto node = std::make_shared<PollDataSdkNode>();

      const std::vector<std::string> expected_topics = {
         "/TactileSensor/StaticData",
         "/TactileSensor/Dynamic",
         "/TactileSensor/Accelerometer",
         "/TactileSensor/Gyroscope",
         "/TactileSensor/EulerAngle",
         "/TactileSensor/Quaternion",
         "/TactileSensor/Timestamp",
      };

      const auto topics = waitForNames(node, expected_topics, [&] { return node->get_topic_names_and_types(); });
      for(const auto& t : expected_topics)
      {
         EXPECT_TRUE(topics.find(t) != topics.end()) << "missing advertised topic: " << t;
      }

      const std::vector<std::string> expected_services = {
         "/tactile_sensors_service",
      };
      const auto services = waitForNames(node, expected_services, [&] { return node->get_service_names_and_types(); });
      EXPECT_TRUE(services.find("/tactile_sensors_service") != services.end()) << "missing tactile_sensors_service";
   }
   rclcpp::shutdown();
}
