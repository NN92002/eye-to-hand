import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('ur5e_camera_workspace')
    ur_description_share = get_package_share_directory('ur_description')
    config = os.path.join(share, 'config', 'ur5e_camera_workspace.yaml')
    rviz = os.path.join(share, 'rviz', 'ur5e_camera_workspace.rviz')
    urdf_file = LaunchConfiguration('urdf_file')
    ur_type = LaunchConfiguration('ur_type')
    default_urdf_file = os.path.join(ur_description_share, 'urdf', 'ur.urdf.xacro')
    return LaunchDescription([
        DeclareLaunchArgument('urdf_file', default_value=default_urdf_file, description='URDF/Xacro file from the installed ur_description package.'),
        DeclareLaunchArgument('ur_type', default_value='ur5e', description='Universal Robots type passed to ur.urdf.xacro.'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_realsense', default_value='false', description='Start the connected D435 driver and publish camera images.'),
        LogInfo(msg=['Using UR5e description: ', urdf_file]),
        Node(package='ur5e_camera_workspace', executable='workspace_visualizer', name='workspace_visualizer', output='screen', parameters=[config]),
        Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher', output='screen', parameters=[{'robot_description': Command(['xacro ', urdf_file, ' name:=ur5e ur_type:=', ur_type])}]),
        Node(package='joint_state_publisher', executable='joint_state_publisher', name='joint_state_publisher', output='screen', parameters=[{'robot_description': Command(['xacro ', urdf_file, ' name:=ur5e ur_type:=', ur_type])}]),
        Node(package='realsense2_camera', executable='realsense2_camera_node', name='camera', output='screen', parameters=[{'enable_color': True, 'enable_depth': True}], condition=IfCondition(LaunchConfiguration('use_realsense'))),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen', arguments=['-d', rviz], condition=IfCondition(LaunchConfiguration('use_rviz'))),
    ])