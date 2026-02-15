#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np
import mediapipe as mp
import time

from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DualHandSmoothControl(Node):

    def __init__(self):
        super().__init__('dual_hand_smooth_control')

        # ===============================
        # ROS Publisher
        # ===============================
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel_command',
            qos_profile_sensor_data
        )

        # ===============================
        # Speed Limits
        # ===============================
        self.min_linear = 0.2
        self.max_linear = 1.5
        self.linear_speed = 0.8

        # ===============================
        # Smoothing
        # ===============================
        self.alpha = 0.25
        self.filtered_x = 0.0
        self.filtered_y = 0.0

        # ===============================
        # Joystick Settings (Right Hand)
        # ===============================
        self.joy_radius = 0.12
        self.joy_center = None
        self.deadzone = 0.15

        # ===============================
        # Speed Adjust (Left Hand)
        # ===============================
        self.pinch_threshold = 0.06
        self.pinch_start_time = None
        self.speed_adjust_enabled = False
        self.last_pinch_y = None
        
        # ===============================
        # Rotation Settings (Left Hand)
        # ===============================
        self.claw_threshold = 0.18
        self.min_angular = 0.2
        self.max_angular = 2.0
        self.angular_speed = 1.0 
        self.filtered_z = 0.0     
        self.claw_start_time = None
        self.ang_speed_adjust_enabled = False
        self.last_claw_y = None

        # ===============================
        # MediaPipe Setup
        # ===============================
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.03, self.timer_callback)
        self.get_logger().info("Dual Hand Control Started")

    def smooth(self, target, current):
        return self.alpha * target + (1 - self.alpha) * current

    def is_pinch(self, hand):
        dist = np.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        return dist < self.pinch_threshold
    
    def is_claw(self, hand):
        # ระยะห่างระหว่างนิ้วโป้ง (4) และนิ้วก้อย (20)
        dist_stretch = np.sqrt((hand[4].x - hand[20].x)**2 + (hand[4].y - hand[20].y)**2)
        # ตรวจสอบว่านิ้วชี้ (8) กางออก (อยู่สูงกว่าข้อต่อโคนนิ้วที่ 5)
        is_extended = hand[8].y < hand[5].y
        # ใช้ Threshold 0.18 (ปรับเพิ่ม/ลดได้ตามระยะห่างจากกล้อง)
        return dist_stretch > self.claw_threshold and is_extended

    def is_fist(self, hand):
        # เช็คว่าปลายนิ้วชี้ (8) อยู่ต่ำกว่าข้อต่อกลางนิ้ว (6) หรือไม่ (การงอนิ้ว)
        return hand[8].y > hand[6].y and hand[12].y > hand[10].y

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        h, w, _ = frame.shape

        target_x, target_y, target_z = 0.0, 0.0, 0.0
        right_hand, left_hand = None, None

        if result.hand_landmarks:
            for idx, hand in enumerate(result.hand_landmarks):
                handedness = result.handedness[idx][0].category_name
                if handedness == "Right": right_hand = hand
                elif handedness == "Left": left_hand = hand

        # ==========================================================
        # RIGHT HAND → JOYSTICK & EMERGENCY STOP
        # ==========================================================
        is_emergency_stop = False
        if right_hand:
            if self.is_fist(right_hand):
                is_emergency_stop = True
                # บังคับค่าให้เป็น 0 ทันที ไม่ผ่านฟังก์ชัน smooth
                self.filtered_x, self.filtered_y, self.filtered_z = 0.0, 0.0, 0.0
                target_x, target_y, target_z = 0.0, 0.0, 0.0
            else:
                idx_tip_r = right_hand[8]
                if self.joy_center is None:
                    self.joy_center = (idx_tip_r.x, idx_tip_r.y)
                
                dx = idx_tip_r.x - self.joy_center[0]
                dy = idx_tip_r.y - self.joy_center[1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > self.joy_radius:
                    scale = self.joy_radius / dist
                    dx *= scale
                    dy *= scale

                norm_x, norm_y = dx/self.joy_radius, dy/self.joy_radius
                if abs(norm_x) < self.deadzone: norm_x = 0
                if abs(norm_y) < self.deadzone: norm_y = 0

                target_x = -norm_y * self.linear_speed
                target_y = -norm_x * self.linear_speed

                # Draw Right UI
                cv2.circle(frame, (int(self.joy_center[0]*w), int(self.joy_center[1]*h)), 
                           int(self.joy_radius*w), (255, 255, 0), 2)
                cv2.circle(frame, (int(idx_tip_r.x*w), int(idx_tip_r.y*h)), 8, (0, 255, 0), -1)
        else:
            self.joy_center = None

        # ==========================================================
        # LEFT HAND → ROTATION & SPEED ADJUST
        # ==========================================================
        if left_hand and not is_emergency_stop:
            idx_tip_l = left_hand[8]
            palm_center_l = left_hand[0] # ใช้จุดข้อมือเป็นศูนย์กลางอุ้งมือ
            px_palm, py_palm = int(palm_center_l.x * w), int(palm_center_l.y * h)
            current_time = time.time()

            # 1. ระบบหมุน (ปัดซ้าย-ขวา)
            side_offset = palm_center_l.x - 0.3
            if abs(side_offset) > 0.05:
                target_z = -side_offset * self.angular_speed * 5.0

            # 2. ระบบปรับความเร็วหมุน (มือเสือ - CLAW) -> ใช้สีฟ้า (Cyan)
            if self.is_claw(left_hand):
                if self.claw_start_time is None:
                    self.claw_start_time = current_time
                    self.last_claw_y = palm_center_l.y
                elif current_time - self.claw_start_time > 1.5:
                    self.ang_speed_adjust_enabled = True
                    dy_ang = self.last_claw_y - palm_center_l.y
                    self.angular_speed = np.clip(self.angular_speed + dy_ang * 0.05, self.min_angular, self.max_angular)
                
                # วาด UI ของ Claw (สีฟ้า)
                cv2.circle(frame, (px_palm, py_palm), 12, (255, 255, 0), -1) # จุดกลางฝ่ามือ
                if self.ang_speed_adjust_enabled:
                    ref_y_claw = int(self.last_claw_y * h)
                    # เส้นอ้างอิงสีน้ำเงินหนา
                    cv2.line(frame, (px_palm - 60, ref_y_claw), (px_palm + 60, ref_y_claw), (255, 100, 0), 2)
                    # เส้นลากเชื่อมโยง
                    cv2.line(frame, (px_palm, py_palm), (px_palm, ref_y_claw), (255, 255, 0), 1)
                    cv2.putText(frame, "ANG SPEED ADJ", (px_palm - 60, py_palm - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    # Progress Bar สีฟ้า
                    prog_claw = (current_time - self.claw_start_time) / 1.5
                    cv2.ellipse(frame, (px_palm, py_palm), (22, 22), 0, 0, int(prog_claw * 360), (255, 255, 0), 3)
            else:
                self.claw_start_time = None
                self.ang_speed_adjust_enabled = False

            # 3. ระบบปรับความเร็วเดิน (จีบ - PINCH) -> ใช้สีขาว/เหลือง
            # เช็คว่าถ้าทำ Claw อยู่ จะไม่ทำ Pinch ซ้อนกันเพื่อความเสถียร
            if self.is_pinch(left_hand) and not self.is_claw(left_hand):
                px_tip, py_tip = int(idx_tip_l.x * w), int(idx_tip_l.y * h)
                if self.pinch_start_time is None:
                    self.pinch_start_time = current_time
                    self.last_pinch_y = idx_tip_l.y
                elif current_time - self.pinch_start_time > 1.5:
                    self.speed_adjust_enabled = True
                    dy_lin = self.last_pinch_y - idx_tip_l.y
                    self.linear_speed = np.clip(self.linear_speed + dy_lin * 0.05, self.min_linear, self.max_linear)
                
                # วาด UI ของ Pinch (สีขาว/เขียว)
                cv2.circle(frame, (px_tip, py_tip), 10, (255, 255, 255), -1)
                if self.speed_adjust_enabled:
                    ref_y_pinch = int(self.last_pinch_y * h)
                    cv2.line(frame, (px_tip - 40, ref_y_pinch), (px_tip + 40, ref_y_pinch), (0, 255, 0), 2)
                    cv2.line(frame, (px_tip, py_tip), (px_tip, ref_y_pinch), (255, 255, 255), 1)
                    cv2.putText(frame, "LIN SPEED ADJ", (px_tip - 60, py_tip - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    prog_pinch = (current_time - self.pinch_start_time) / 1.5
                    cv2.ellipse(frame, (px_tip, py_tip), (18, 18), 0, 0, int(prog_pinch * 360), (255, 255, 255), 2)
            else:
                self.pinch_start_time = None
                self.speed_adjust_enabled = False

        # ==========================================================
        # FINAL COMMANDS
        # ==========================================================
        self.filtered_x = self.smooth(target_x, self.filtered_x)
        self.filtered_y = self.smooth(target_y, self.filtered_y)
        self.filtered_z = self.smooth(target_z, self.filtered_z)

        cmd = Twist()
        cmd.linear.x = float(self.filtered_x)
        cmd.linear.y = float(self.filtered_y)
        cmd.angular.z = float(self.filtered_z)
        self.cmd_pub.publish(cmd)

        # Dashboard UI
        cv2.putText(frame, f"Lin Speed: {self.linear_speed:.2f}", (20, 40), 0, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"Ang Speed: {self.angular_speed:.2f}", (20, 70), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow("Dual Hand Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): rclpy.shutdown()

def main():
    rclpy.init()
    node = DualHandSmoothControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()