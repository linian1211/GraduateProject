from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'rr_position_control.urdf')
    sdf_file = '/home/tingyi/git/GraduateProject/Rviz_AngleDisplay_test/building_robot.sdf'

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    return LaunchDescription([
        # 宣告 use_gazebo 參數
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='true',
            description='Whether to launch Gazebo'
        ),
        # 啟動 robot_state_publisher（用於 RViz2）
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description, 'publish_frequency': 10.0, 'use_sim_time': True}]
        ),
        # 啟動 joint_state_publisher_gui（僅在不使用 Gazebo 時）
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            condition=UnlessCondition(LaunchConfiguration('use_gazebo'))
        ),
        # 啟動 Gazebo（使用 building_robot.sdf）
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', sdf_file],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_gazebo'))
        ),
        # 啟動 RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/tingyi/rviz_config.rviz'],
            parameters=[{'use_sim_time': True}]
        )
    ])