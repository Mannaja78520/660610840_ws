#!/bin/bash

source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash
cd ~/660610840_ws && python test.py
roslaunch myagv_odometry myagv_active.launch
