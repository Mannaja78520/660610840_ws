import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy

import socket, threading, time


class UdpCmdVelRelay(Node):
    def __init__(self):
        super().__init__('udp_cmd_vel_relay')

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub = self.create_publisher(Twist, '/cmd_vel', qos)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 15000))

        self.last_rx = time.time()
        self.timeout = 0.5

        threading.Thread(target=self.udp_loop, daemon=True).start()
        self.create_timer(0.1, self.watchdog)

        self.get_logger().info('UDP → /cmd_vel relay running')

    def udp_loop(self):
        while rclpy.ok():
            data, _ = self.sock.recvfrom(1024)
            try:
                x, y, z = map(float, data.decode().split())
                msg = Twist()
                msg.linear.x = x
                msg.linear.y = y
                msg.angular.z = z
                self.pub.publish(msg)
                self.last_rx = time.time()
            except Exception:
                pass

    def watchdog(self):
        if time.time() - self.last_rx > self.timeout:
            self.pub.publish(Twist())


def main():
    rclpy.init()
    rclpy.spin(UdpCmdVelRelay())
    rclpy.shutdown()


if __name__ == '__main__':
    main()