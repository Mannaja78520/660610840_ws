from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    default_params = os.path.join(
        FindPackageShare('mor_luam').perform(None),
        'config', 'joystick.yaml'
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the joystick parameter file'
    )

    node = Node(
        package='mor_luam',
        executable='joystick_control.py',   # ตรงกับที่คุณ install(PROGRAMS ...) ใน CMakeLists
        name='joystick_control_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
        # สามารถใส่ remappings=[('cmd_vel','/your/other/cmd_vel')] ได้ถ้าต้องการ
    )

    return LaunchDescription([params_arg, node])
