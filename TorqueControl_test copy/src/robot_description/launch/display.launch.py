from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'rr_position_control.urdf')
    sdf_file = os.path.join(pkg_share, 'sdf', 'building_robot.sdf')

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='true',
            description='Whether to launch Gazebo'
        ),

        Node(
            package='robot_description',
            executable='joint_angle_plotter.py',
            name='joint_angle_plotter',
            output='screen',
            prefix=['python3 '],  # 明確指定 python3
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': 30.0,
                'use_sim_time': True,
                'tf_buffer_duration': 20.0
            }]
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            parameters=[{'use_sim_time': True}],
            condition=UnlessCondition(LaunchConfiguration('use_gazebo'))
        ),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', sdf_file],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_gazebo'))
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/tingyi/rviz_config.rviz'],
            parameters=[{'use_sim_time': True}]
        )
    ])