from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():

    keyboard = Node(
        package='name_sensei_proj',
        executable='keyboard_control_own_msg.py',
        name='keyboard_control_node',
        output='screen',
        # parameters=[LaunchConfiguration('params_file')]
        # สามารถใส่ remappings=[('cmd_vel','/your/other/cmd_vel')] ได้ถ้าต้องการ
    )
    
    mediapipe = Node(
        package='name_sensei_proj',
        executable='mediapipe_control_own_msg.py',
        name='mediapipe_control_node',
        output='screen',
        # parameters=[LaunchConfiguration('params_file')]
        # สามารถใส่ remappings=[('cmd_vel','/your/other/cmd_vel')] ได้ถ้าต้องการ
    )
    
    main_controller_srv_server = Node(
        package='name_sensei_proj',
        executable='main_controller.py',
        name='main_controller_node',
        output='screen',
        # parameters=[LaunchConfiguration('params_file')]
        # สามารถใส่ remappings=[('cmd_vel','/your/other/cmd_vel')] ได้ถ้าต้องการ
    )
    
    obj_avoid = Node(
        package='name_sensei_proj',
        executable='obj_nearest_alert.py',
        name='obj_avoid_node',
        output='screen',
        # parameters=[LaunchConfiguration('params_file')]
        # สามารถใส่ remappings=[('cmd_vel','/your/other/cmd_vel')] ได้ถ้าต้องการ
    )

    return LaunchDescription([
                                mediapipe, 
                                main_controller_srv_server,
                                obj_avoid
                            ])