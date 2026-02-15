#!/bin/bash

SESSION=bridge_session

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

tmux send-keys -t $SESSION "source /opt/ros/noetic/setup.bash" C-m
tmux send-keys -t $SESSION "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION "export ROS_DOMAIN_ID=0" C-m
tmux send-keys -t $SESSION "ros2 run ros1_bridge dynamic_bridge --bridge-all-topics" C-m

tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION "export ROS_DOMAIN_ID=1" C-m
tmux send-keys -t $SESSION "python ~/660610840_ws/udp_sender_sub_from_ros2.py" C-m

tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/galactic/setup.bash" C-m
tmux send-keys -t $SESSION "export ROS_DOMAIN_ID=0" C-m
tmux send-keys -t $SESSION "python ~/660610840_ws/udp_receiver_pub_ros2.py" C-m

tmux select-layout -t $SESSION even-horizontal

tmux attach -t $SESSION
