#!/usr/bin/env python3

import sys
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


class KeyboardControl(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')

        # Publishers
        self.keyboard_pub = self.create_publisher(
            String, '/keyboard_input', 10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel_command', qos_profile_sensor_data
        )

        # ===== Parameters =====
        self.max_linear_speed = 2.0  # m/s

        self.declare_parameter('max_speed', self.max_linear_speed)
        self.declare_parameter('plus_move_speed', 0.25)
        self.declare_parameter('plus_slide_speed', 0.25)
        self.declare_parameter('plus_turn_speed', 0.5)
        self.declare_parameter('plus_speed_size', 0.01)

        self.maxSpeed = self.get_parameter('max_speed').value
        self.plusMoveSpeed = self.get_parameter('plus_move_speed').value
        self.plusSlideSpeed = self.get_parameter('plus_slide_speed').value
        self.plusTurnSpeed = self.get_parameter('plus_turn_speed').value
        self.plusSpeedSize = self.get_parameter('plus_speed_size').value

        # Current velocities
        self.moveSpeed = 0.0
        self.slideSpeed = 0.0
        self.turnSpeed = 0.0

        # ===== Timer 10 Hz =====
        self.timer = self.create_timer(0.1, self.timer_publish_cmd)

        self.show_log()
        self.get_logger().info("Keyboard control started (10 Hz cmd_vel). Press 'p' to quit.")

        # Start keyboard thread
        self.key_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.key_thread.start()

    # --------------------------------------------------

    def clip(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def show_log(self):
        log_message = (
            "\n=== Keyboard Control (m/s mode) ===\n"
            "w / s : Move forward / backward\n"
            "a / d : Slide left / right\n"
            "q / e : Turn left / right\n"
            "W / S : Increase / decrease move increment\n"
            "A / D : Increase / decrease slide increment\n"
            "Q / E : Increase / decrease turn increment\n"
            "SPACE : Brake\n"
            "p     : Quit\n"
            f"Max speed      : {self.maxSpeed:.2f} m/s\n"
            f"Current speed  : x={self.moveSpeed:.2f}, y={self.slideSpeed:.2f}, z={self.turnSpeed:.2f}\n"
            f"Increments     : move={self.plusMoveSpeed:.2f}, slide={self.plusSlideSpeed:.2f}, turn={self.plusTurnSpeed:.2f}\n"
        )
        self.get_logger().info(log_message)

    # --------------------------------------------------

    def update_speeds(self, key):
        if key == 'w':
            self.moveSpeed += self.plusMoveSpeed
        elif key == 's':
            self.moveSpeed -= self.plusMoveSpeed
        elif key == 'a':
            self.slideSpeed += self.plusSlideSpeed
        elif key == 'd':
            self.slideSpeed -= self.plusSlideSpeed
        elif key == 'q':
            self.turnSpeed += self.plusTurnSpeed
        elif key == 'e':
            self.turnSpeed -= self.plusTurnSpeed

        elif key == 'W':
            self.plusMoveSpeed += self.plusSpeedSize
        elif key == 'S':
            self.plusMoveSpeed -= self.plusSpeedSize
        elif key == 'A':
            self.plusSlideSpeed += self.plusSpeedSize
        elif key == 'D':
            self.plusSlideSpeed -= self.plusSpeedSize
        elif key == 'Q':
            self.plusTurnSpeed += self.plusSpeedSize
        elif key == 'E':
            self.plusTurnSpeed -= self.plusSpeedSize

        elif key == ' ':
            self.moveSpeed = 0.0
            self.slideSpeed = 0.0
            self.turnSpeed = 0.0

        self.moveSpeed = self.clip(self.moveSpeed, -self.maxSpeed, self.maxSpeed)
        self.slideSpeed = self.clip(self.slideSpeed, -self.maxSpeed, self.maxSpeed)
        self.turnSpeed = self.clip(self.turnSpeed, -self.maxSpeed, self.maxSpeed)
        self.show_log()

    # --------------------------------------------------

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def keyboard_loop(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'p':
                rclpy.shutdown()
                return

            self.update_speeds(key)

            msg = String()
            msg.data = key
            self.keyboard_pub.publish(msg)

    # --------------------------------------------------

    def timer_publish_cmd(self):
        cmd = Twist()
        cmd.linear.x = float(self.moveSpeed)
        cmd.linear.y = float(self.slideSpeed)
        cmd.angular.z = float(self.turnSpeed)

        self.cmd_vel_pub.publish(cmd)


# ======================================================

def main():
    rclpy.init()
    node = KeyboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

