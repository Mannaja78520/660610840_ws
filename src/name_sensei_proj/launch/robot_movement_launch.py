from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    monitor = Node(
        package='mor_luam',
        executable='robot_movement.py',
        name='robot_movement_monitor',
        output='screen',
    )

    return LaunchDescription([monitor])
