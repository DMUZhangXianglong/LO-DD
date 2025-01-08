'''
Author: DMUZhangXianglong 347913076@qq.com
Date: 2024-11-18 21:27:36
LastEditors: DMUZhangXianglong 347913076@qq.com
LastEditTime: 2024-12-11 16:03:38
* @FilePath: /LO-DD/launch/run.launch.py
Description: ROS2 节点启动
'''
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():

    share_dir = get_package_share_directory('lo_dd')
    parameter_file = LaunchConfiguration('params_file')
    
    xacro_path = os.path.join(share_dir, 'config', 'robot.urdf.xacro')
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')


    return LaunchDescription([
    params_declare,
   
    Node(
        package='lo_dd',
        executable='lo_dd_lidarOdometry',
        name='lo_dd_lidarOdometry',
        parameters=[parameter_file],
        output='screen'
    ),

    Node(
    package='lo_dd',
    executable='lo_dd_featureExtraction',
    name='lo_dd_featureExtraction',
    parameters=[parameter_file],
    output='screen'
    ),

    # Node(
    # package='rviz2',
    # executable='rviz2',
    # name='rviz2',
    # arguments=['-d', rviz_config_file],
    # output='screen'
    # )
    ])

    