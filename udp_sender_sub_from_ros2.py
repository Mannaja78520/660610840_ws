import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket


UDP_IP = '127.0.0.1'   # IP or robot IP
UDP_PORT = 15000


class CmdVelToUDP(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_udp')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel_command',
            self.cb,
            10
        )
        self.get_logger().info('Forwarding /cmd_vel_command to UDP')

    def cb(self, msg):
        data = f"{msg.linear.x} {msg.linear.y} {msg.angular.z}"
        self.sock.sendto(data.encode(), (UDP_IP, UDP_PORT))


def main():
    rclpy.init()
    rclpy.spin(CmdVelToUDP())
    rclpy.shutdown()


if __name__ == '__main__':
    main()