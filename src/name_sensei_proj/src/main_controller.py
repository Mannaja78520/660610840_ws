#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist # ใช้สำหรับ Output
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

# นำเข้า Message และ Service ที่คุณสร้างเอง
from name_sensei_proj.srv import SetMode
from name_sensei_proj.msg import MecanumCmd 

class MainDecisionServer(Node):
    def __init__(self):
        super().__init__('main_decision_server')
        
        # 1. Subscribe รับข้อมูลขาเข้าแบบ MecanumCmd
        self.create_subscription(MecanumCmd, '/cmd_vel_control', self.hand_callback, qos_profile_sensor_data)
        self.create_subscription(MecanumCmd, '/cmd_collision', self.collision_callback, 10)
        
        # 2. Publish ข้อมูลขาออกเป็น Twist ตามที่คุณต้องการ
        self.real_cmd_pub = self.create_publisher(Twist, '/cmd_vel_command', 10)
        self.status_pub = self.create_publisher(String, '/system_mode', 10)

        self.srv = self.create_service(SetMode, '/set_master_lock', self.handle_lock_service)

        self.master_lock = False
        self.collision_active = False
        self.last_hand_cmd = MecanumCmd()
        self.last_collision_cmd = MecanumCmd()
        
        self.create_timer(0.05, self.decision_loop)
        # self.get_logger().info("Decision Server: Receiving MecanumCmd -> Publishing Twist")

    def handle_lock_service(self, request, response):
        self.master_lock = request.master_lock
        response.success = True
        response.message = f"Master Lock Status: {self.master_lock}"
        return response

    def hand_callback(self, msg):
        self.last_hand_cmd = msg

    def collision_callback(self, msg):
        self.last_collision_cmd = msg
        # ตรวจสอบว่าระบบหลบหลีกทำงานหรือไม่
        if abs(msg.x) > 0.01 or abs(msg.y) > 0.01:
            self.collision_active = True
        else:
            self.collision_active = False

    def decision_loop(self):
        # สร้าง Message วัตถุ Twist สำหรับส่งออก
        final_msg = Twist() 
        mode_str = ""

        # --- ส่วนการตัดสินใจ (Priority Logic) ---
        if self.master_lock:
            # หยุดนิ่ง (Twist เริ่มต้นเป็น 0 อยู่แล้ว)
            mode_str = "LOCKED: SERVICE OVERRIDE"
        
        elif self.collision_active:
            # แปลง MecanumCmd จาก LIDAR เป็น Twist
            final_msg.linear.x = self.last_collision_cmd.x
            final_msg.linear.y = self.last_collision_cmd.y
            final_msg.angular.z = self.last_collision_cmd.omega_z
            mode_str = "AUTO: OBSTACLE AVOIDANCE"
            
        else:
            # แปลง MecanumCmd จากมือคน (MediaPipe) เป็น Twist
            final_msg.linear.x = self.last_hand_cmd.x
            final_msg.linear.y = self.last_hand_cmd.y
            final_msg.angular.z = self.last_hand_cmd.omega_z
            mode_str = "MANUAL: HAND GESTURE"

        # ส่งคำสั่ง Twist ออกไป
        self.real_cmd_pub.publish(final_msg)
        
        # ส่งสถานะระบบ
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