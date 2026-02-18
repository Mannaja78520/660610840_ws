#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import socket
import threading

# ตั้งค่าให้ตรงกับระบบจริง
ROBOT_IP = 'er.local' 
CMD_PORT = 15000      # พอร์ตส่งคำสั่งไปหุ่น
SCAN_PORT = 15001     # พอร์ตรับข้อมูล Scan จากหุ่น

class UdpGatewayNode(Node):
    def __init__(self):
        super().__init__('udp_gateway_node')
        
        # --- [ฝั่งส่งคำสั่ง: ROS -> UDP] ---
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_command', self.cmd_cb, qos_profile_sensor_data)

        # --- [ฝั่งรับ Scan: UDP -> ROS] ---
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', SCAN_PORT))
        
        # Thread สำหรับรอรับข้อมูล Scan ตลอดเวลา
        threading.Thread(target=self.receive_scan_loop, daemon=True).start()
        self.get_logger().info(f'Gateway Online: CMD to {ROBOT_IP}:{CMD_PORT}, SCAN on port {SCAN_PORT}')

    def cmd_cb(self, msg):
        """ รับค่าจาก Main Server และส่งไปที่หุ่น """
        data = f"{msg.linear.x:.3f} {msg.linear.y:.3f} {msg.angular.z:.3f}"
        self.sock_send.sendto(data.encode(), (ROBOT_IP, CMD_PORT))

    def receive_scan_loop(self):
        """ รับข้อมูลดิบจากหุ่น แล้วแปลงเป็น LaserScan Message """
        while rclpy.ok():
            try:
                data, _ = self.sock_recv.recvfrom(65535)
                # สมมติหุ่นส่งข้อมูลเป็น string คั่นด้วยคอมม่า "1.2,0.5,3.0,..."
                raw_ranges = [float(x) for x in data.decode().split(',')]
                
                scan_msg = LaserScan()
                scan_msg.header.stamp = self.get_clock().now().to_msg()
                scan_msg.header.frame_id = 'laser_frame'
                scan_msg.angle_min = 0.0
                scan_msg.angle_max = 2.0 * 3.14159
                scan_msg.angle_increment = scan_msg.angle_max / len(raw_ranges)
                scan_msg.range_min = 0.12
                scan_msg.range_max = 8.0
                scan_msg.ranges = raw_ranges
                
                self.scan_pub.publish(scan_msg)
            except Exception:
                pass

def main():
    rclpy.init()
    rclpy.spin(UdpGatewayNode())
    rclpy.shutdown()

if __name__ == '__main__': main()