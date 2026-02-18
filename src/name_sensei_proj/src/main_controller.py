#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

# เปลี่ยนจาก SetBool มาใช้ SetMode ที่คุณสร้างเอง
from name_sensei_proj.srv import SetMode

class MainDecisionServer(Node):
    def __init__(self):
        super().__init__('main_decision_server')
        
        # Subscribe รับคำสั่งจาก Hand และ LIDAR
        self.create_subscription(Twist, '/cmd_vel_control', self.hand_callback, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_collision', self.collision_callback, 10)
        
        # Publish คำสั่งจริงไปที่หุ่นยนต์
        self.real_cmd_pub = self.create_publisher(Twist, '/cmd_vel_command', 10)
        self.status_pub = self.create_publisher(String, '/system_mode', 10)

        # --- แก้ไขตรงนี้: เปลี่ยนเป็น SetMode ---
        self.srv = self.create_service(SetMode, '/set_master_lock', self.handle_lock_service)

        self.master_lock = False
        self.collision_active = False
        self.last_hand_cmd = Twist()
        self.last_collision_cmd = Twist()
        
        self.create_timer(0.05, self.decision_loop)
        self.get_logger().info("Main Decision Server with Custom SetMode SRV started.")

    def handle_lock_service(self, request, response):
        # เปลี่ยนจาก request.data เป็น request.master_lock ตามไฟล์ .srv ของคุณ
        self.master_lock = request.master_lock
        
        response.success = True
        if self.master_lock:
            response.message = "SYSTEM LOCKED: Hand control disabled by admin."
            self.get_logger().warn("Master Lock Engaged!")
        else:
            response.message = "SYSTEM UNLOCKED: Hand control enabled."
            self.get_logger().info("Master Lock Disengaged.")
            
        return response

    def hand_callback(self, msg):
        self.last_hand_cmd = msg

    def collision_callback(self, msg):
        self.last_collision_cmd = msg
        if abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01:
            self.collision_active = True
        else:
            self.collision_active = False

    def decision_loop(self):
        final_cmd = Twist()
        mode_str = ""

        if self.master_lock:
            final_cmd = Twist() # หยุดนิ่ง
            mode_str = "LOCKED: SERVICE OVERRIDE"
        elif self.collision_active:
            final_cmd = self.last_collision_cmd
            mode_str = "AUTO: OBSTACLE AVOIDANCE"
        else:
            final_cmd = self.last_hand_cmd
            mode_str = "MANUAL: HAND GESTURE"

        self.real_cmd_pub.publish(final_cmd)
        
        status_msg = String()
        status_msg.data = mode_str
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MainDecisionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()