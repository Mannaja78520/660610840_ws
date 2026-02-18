#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import socket
import threading

# ตั้งค่า IP คอมพิวเตอร์ของคุณ
PC_IP = 'er.local' # <--- สำคัญ: ใส่ IP เครื่องที่รัน ROS Main Server
CMD_PORT = 15000
SCAN_PORT = 15001

class RobotHardwareNode(Node):
    def __init__(self):
        super().__init__('robot_hardware_node')
        
        # 1. รับคำสั่งจากคอมพิวเตอร์ (UDP) มา Publish ลง Topic ในหุ่น
        self.local_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', CMD_PORT))
        
        # 2. รับ Scan จาก Driver ในหุ่น (Topic /scan) แล้วส่งไปคอมพิวเตอร์ (UDP)
        self.sock_send_scan = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10)
        
        threading.Thread(target=self.udp_cmd_loop, daemon=True).start()
        self.get_logger().info(f"Robot Node Started. PC IP: {PC_IP}")

    def udp_cmd_loop(self):
        """ รับคำสั่ง x y z จาก UDP แล้วพ่นออกเป็น /cmd_vel ในหุ่น """
        while rclpy.ok():
            try:
                data, _ = self.sock_recv.recvfrom(1024)
                x, y, z = map(float, data.decode().split())
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.angular.z = x, y, z
                self.local_cmd_pub.publish(msg)
            except:
                pass

    def scan_cb(self, msg):
        """ รับข้อมูลจาก Topic /scan บนหุ่น แล้วส่งออก UDP ไปหาคอมพิวเตอร์ """
        try:
            # แปลงข้อมูล ranges เป็น string "1.2,0.5,..." เพื่อส่งผ่าน UDP
            data_str = ",".join([f"{r:.3f}" for r in msg.ranges])
            self.sock_send_scan.sendto(data_str.encode(), (PC_IP, SCAN_PORT))
        except Exception as e:
            self.get_logger().error(f"Send Scan Error: {e}")

def main():
    rclpy.init()
    rclpy.spin(RobotHardwareNode())
    rclpy.shutdown()

if __name__ == '__main__': main()