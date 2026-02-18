#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from name_sensei_proj.msg import MecanumCmd
from std_msgs.msg import String
import math

class Obj_nearest_alert(Node):
    def __init__(self):
        super().__init__('obj_nearest_alert')
        
        self.yaw_deg = 180.0 
        self.target_dist = 0.12  # ระยะห่างที่ต้องการรักษา (นับจากขอบหุ่น)
        self.alert_dist = 0.15   # ระยะแจ้งเตือน (นับจากขอบหุ่น)
        self.kp = 1.5            # เพิ่ม Gain ให้ตอบสนองไวขึ้น

        # --- ขนาดตัวหุ่น myAGV (หน่วยเมตร) ---
        self.robot_half_length = 0.311 / 2  # 0.1555 m
        self.robot_half_width = 0.190 / 2   # 0.095 m

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(MecanumCmd, '/cmd_collision', 10)
        self.alert_pub = self.create_publisher(String, '/obstacle_alert', 10)

    def rotate_ranges(self, msg, yaw_deg):
        ranges = list(msg.ranges)
        yaw_rad = math.radians(yaw_deg)
        shift = int(yaw_rad / msg.angle_increment)
        N = len(ranges)
        shift = shift % N
        rotated_ranges = ranges[-shift:] + ranges[:-shift]
        return rotated_ranges

    def scan_callback(self, msg):
        rotated_ranges = self.rotate_ranges(msg, self.yaw_deg)
        num_readings = len(rotated_ranges)
        
        vel_x = 0.0
        vel_y = 0.0
        
        for i in range(num_readings):
            raw_dist = rotated_ranges[i]
            
            if not math.isfinite(raw_dist) or raw_dist <= 0.01:
                continue

            angle_rad = msg.angle_min + (i * msg.angle_increment)
            
            # --- ส่วนที่เพิ่ม: คำนวณระยะทางจากขอบหุ่น (Effective Distance) ---
            # ใช้ Geometric Offset เพื่อดูว่า LiDAR อยู่ห่างจากขอบหุ่นในมุมนั้นๆ เท่าไหร่
            offset_x = abs(math.cos(angle_rad) * self.robot_half_length)
            offset_y = abs(math.sin(angle_rad) * self.robot_half_width)
            # ระยะห่างจากจุดกลาง LiDAR ถึงขอบหุ่นในทิศทางนั้น
            robot_edge_offset = math.sqrt(offset_x**2 + offset_y**2)
            
            # ระยะที่เหลือจริงๆ จากขอบหุ่นถึงวัตถุ
            effective_dist = raw_dist - robot_edge_offset

            # --- ใช้ effective_dist ในการตัดสินใจแทน raw_dist ---
            if effective_dist < self.alert_dist:
                cw_deg = (-math.degrees(angle_rad) + 360) % 360
                alert_msg = String()
                alert_msg.data = f"WARNING! Object {effective_dist*1000:.0f}mm from edge at {cw_deg:.1f}°"
                self.alert_pub.publish(alert_msg)

            if effective_dist < self.target_dist:
                # ถ้าวัตถุชิดขอบหุ่นเกินไป (เช่น < 120mm จากขอบ) ให้สร้างแรงผลัก
                error = self.target_dist - effective_dist
                vel_x -= math.cos(angle_rad) * error * self.kp
                vel_y -= math.sin(angle_rad) * error * self.kp

        self.control_mecanum(vel_x, vel_y)

    def control_mecanum(self, vx, vy):
        move_cmd = MecanumCmd()
        limit = 0.3  # ปรับ speed limit ขึ้นเล็กน้อยเพื่อให้หนีทัน
        move_cmd.x = max(min(vx, limit), -limit)
        move_cmd.y = max(min(vy, limit), -limit)
        self.cmd_vel_pub.publish(move_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = Obj_nearest_alert()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()