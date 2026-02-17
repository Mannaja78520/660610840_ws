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
        self.min_linear = 0.1
        self.max_linear = 1.0
        self.linear_speed = 0.15
        self.angular_speed = 1.0 
        self.min_angular = 0.2
        self.max_angular = 2.0

        # ===============================
        # Smoothing
        # ===============================
        self.alpha = 0.25
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.filtered_z = 0.0 

        # ===============================
        # Joystick Settings (Right Hand)
        # ===============================
        self.joy_radius = 0.12
        self.joy_center = None
        self.deadzone = 0.15

        # ===============================
        # Speed Adjust (Left Hand)
        # ===============================
        self.pinch_threshold = 0.1
        self.pinch_start_time = None
        self.speed_adjust_enabled = False
        self.last_pinch_y = None
        self.pinch_start_time_cooldown = 1.0
        self.draw_pinch_pos = None
        
        # ===============================
        # Rotation Settings (Left Hand)
        # ===============================
        self.claw_threshold = 0.15
        self.claw_start_time_cooldown = 1.0
        self.claw_start_time = None
        self.ang_speed_adjust_enabled = False
        self.last_claw_y = None
        self.draw_claw_ref = None
        self.draw_claw_curr = None
        
        # =============================
        # Latching Settings
        # =============================
        self.latch_start_time = None
        self.latch_duration = 0.15  # ระยะเวลาที่จะให้ดูดเข้าหาศูนย์กลาง (วินาที)

        # ===============================
        # MediaPipe Setup
        # ===============================
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.get_logger().info("Dual Hand Control - Frame-driven Mode Started")

    def smooth(self, target, current):
        return self.alpha * target + (1 - self.alpha) * current

    def is_pinch(self, hand):
        dist_tips = np.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        dist_joints = np.sqrt((hand[3].x - hand[6].x)**2 + (hand[3].y - hand[6].y)**2)
        dist_middle = np.sqrt((hand[8].x - hand[12].x)**2 + (hand[8].y - hand[12].y)**2)
        current_threshold = self.pinch_threshold
        if self.speed_adjust_enabled:
            current_threshold = self.pinch_threshold * 1.6 
        return dist_tips < current_threshold and dist_joints < 0.16 and dist_middle > 0.05
    
    def is_claw(self, hand):
        current_claw_thresh = self.claw_threshold
        if self.ang_speed_adjust_enabled:
            current_claw_thresh = self.claw_threshold * 0.5
        dist_stretch = np.sqrt((hand[4].x - hand[20].x)**2 + (hand[4].y - hand[20].y)**2)
        dist_index_middle = np.sqrt((hand[8].x - hand[12].x)**2 + (hand[8].y - hand[12].y)**2)
        dist_thumb_index = np.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        return dist_stretch > current_claw_thresh and dist_index_middle > 0.08 and dist_thumb_index > 0.10

    def is_fist(self, hand):
        return (hand[8].y > hand[6].y and 
                hand[12].y > hand[10].y and 
                hand[16].y > hand[14].y)

    def run(self):
        """ลูปหลักรันตามเฟรมกล้อง"""
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            self.process_frame(frame)
            
            # ยอมให้ ROS ทำงานเบื้องหลัง (เช่น รับ Subs)
            rclpy.spin_once(self, timeout_sec=0)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def process_frame(self, frame):
        timestamp_ms = int(time.time() * 1000)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        target_x, target_y, target_z = 0.0, 0.0, 0.0
        right_hand, left_hand = None, None

        if result.hand_landmarks:
            for idx, hand in enumerate(result.hand_landmarks):
                handedness = result.handedness[idx][0].category_name
                if handedness == "Right": right_hand = hand
                elif handedness == "Left": left_hand = hand

        # ==========================================================
        # RIGHT HAND → HYBRID CONTROL (WITH TIMED LATCHING)
        # ==========================================================
        is_emergency_stop = False
        if right_hand:
            if self.is_fist(right_hand):
                is_emergency_stop = True
                target_x = target_y = target_z = 0.0
                self.filtered_x = self.filtered_y = self.filtered_z = 0.0
                self.joy_center = None 
                self.latch_start_time = None # Reset latch timer
                cv2.putText(frame, "EMERGENCY STOP", (int(w/2)-180, h-50), 0, 1.2, (0, 0, 255), 3)
            else:
                idx_tip_r = right_hand[8]
                current_time = time.time()
                if self.joy_center is None: 
                    self.joy_center = (idx_tip_r.x, idx_tip_r.y)
                
                dx_raw = idx_tip_r.x - self.joy_center[0]
                dy_raw = idx_tip_r.y - self.joy_center[1]
                dist_raw = np.sqrt(dx_raw**2 + dy_raw**2)

                # --- 1. คำนวณ Angular (เลี้ยว) ---
                mid_tip, mid_base = right_hand[12], right_hand[9]
                ring_tip, ring_base = right_hand[16], right_hand[13]
                fold_l, fold_r = mid_tip.y - mid_base.y, ring_tip.y - ring_base.y

                target_z = 0.0
                is_turning = False
                if fold_l > 0.025: 
                    target_z = fold_l * self.angular_speed * 12.0
                    is_turning = True
                elif fold_r > 0.025: 
                    target_z = -fold_r * self.angular_speed * 12.0
                    is_turning = True

                # --- 2. Smart Timed Latching Logic ---
                if is_turning:
                    if self.latch_start_time is None:
                        self.latch_start_time = current_time # เริ่มนับเวลาถอยหลังการดูด
                    
                    # จะดูดเข้าศูนย์กลางเฉพาะในช่วงเวลา latch_duration แรกที่เริ่มเลี้ยว
                    elapsed_latch = current_time - self.latch_start_time
                    if elapsed_latch < self.latch_duration and dist_raw < (self.joy_radius * 0.5):
                        latch_speed = 0.2 
                        self.joy_center = (
                            self.joy_center[0] * (1 - latch_speed) + idx_tip_r.x * latch_speed,
                            self.joy_center[1] * (1 - latch_speed) + idx_tip_r.y * latch_speed
                        )
                        cv2.putText(frame, "LATCHING...", (int(idx_tip_r.x*w)+20, int(idx_tip_r.y*h)), 0, 0.5, (0, 255, 255), 1)
                else:
                    self.latch_start_time = None # รีเซ็ตเมื่อเลิกเลี้ยว

                # --- 3. คำนวณ Linear (Joystick) ---
                # ส่วนนี้จะทำงานปกติ ทำให้ถ้าพ้น 0.5 วินาทีไปแล้ว คุณดันนิ้วชี้หุ่นจะเดินไปด้วยเลี้ยวไปด้วยได้
                dx = idx_tip_r.x - self.joy_center[0]
                dy = idx_tip_r.y - self.joy_center[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > self.joy_radius:
                    scale = self.joy_radius / dist
                    dx *= scale; dy *= scale

                norm_x, norm_y = dx/self.joy_radius, dy/self.joy_radius
                if abs(norm_x) < self.deadzone: norm_x = 0
                if abs(norm_y) < self.deadzone: norm_y = 0

                target_x = -norm_y * self.linear_speed
                target_y = -norm_x * self.linear_speed

                # --- 4. แสดงผล UI การเลี้ยว (Rotation Indicator) ---
                cx, cy = int(self.joy_center[0]*w), int(self.joy_center[1]*h)
                joy_px_radius = int(self.joy_radius * w)
                
                # วาดขอบจอยปกติ
                cv2.circle(frame, (cx, cy), joy_px_radius, (255, 255, 0), 1)
                
                if is_turning:
                    # วาดเส้นโค้งบอกทิศทาง (Arc)
                    # ถ้า target_z > 0 (เลี้ยวซ้าย) -> วาดทวนเข็ม / target_z < 0 (เลี้ยวขวา) -> วาดตามเข็ม
                    start_angle = -90 
                    end_angle = start_angle - (target_z * 20) # ยืดความยาวตามความเร็วหมุน
                    
                    color = (255, 0, 255) if target_z > 0 else (0, 255, 255) # ม่วง=ซ้าย, เหลือง=ขวา
                    thickness = 5
                    cv2.ellipse(frame, (cx, cy), (joy_px_radius + 10, joy_px_radius + 10), 
                                0, start_angle, end_angle, color, thickness)
                    
                    direction_text = "LEFT" if target_z > 0 else "RIGHT"
                    cv2.putText(frame, direction_text, (cx - 30, cy - joy_px_radius - 20), 
                                0, 0.7, color, 2)

                # วาดจุดนิ้วชี้
                tip_px = (int(idx_tip_r.x*w), int(idx_tip_r.y*h))
                cv2.line(frame, (cx, cy), tip_px, (0, 255, 0), 2)
                cv2.circle(frame, tip_px, 10, (0, 255, 0), -1)
        else:
            self.joy_center = None

        # ==========================================================
        # LEFT HAND → SPEED ADJUST
        # ==========================================================
        if left_hand:
            current_time = time.time()
            
            # 1. ระบบปรับความเร็วเดิน (จีบ - PINCH)
            if self.is_pinch(left_hand) and not self.is_claw(left_hand):
                px_t, py_t = int(left_hand[8].x * w), int(left_hand[8].y * h)
                
                if self.pinch_start_time is None:
                    self.pinch_start_time = current_time
                    # ล็อคจุดอ้างอิง (Anchor) ทันทีที่เริ่มจีบ
                    self.last_pinch_y = left_hand[8].y
                
                elapsed_pinch = current_time - self.pinch_start_time
                self.draw_pinch_ref = (px_t, int(self.last_pinch_y * h))
                self.draw_pinch_curr = (px_t, py_t)

                if elapsed_pinch < self.pinch_start_time_cooldown:
                    # จังหวะ HOLD: วาดจุด Anchor และวงกลม Cooldown
                    prog_p = elapsed_pinch / self.pinch_start_time_cooldown
                    cv2.circle(frame, self.draw_pinch_ref, 5, (255, 255, 255), -1) # จุด Anchor ขณะ Hold
                    cv2.ellipse(frame, (px_t, py_t), (18, 18), 0, 0, int(prog_p * 360), (255, 255, 255), 2)
                    cv2.putText(frame, "HOLD PINCH", (px_t - 30, py_t - 35), 0, 0.5, (255, 255, 255), 1)
                else:
                    # พ้น Cooldown: เริ่มโหมดปรับค่าแบบ Relative
                    self.speed_adjust_enabled = True
                    dy_lin = self.last_pinch_y - left_hand[8].y
                    # ปรับความเร็วตามระยะห่างจากจุด Anchor
                    self.linear_speed = np.clip(self.linear_speed + dy_lin * 0.1, self.min_linear, self.max_linear)
            else:
                self.pinch_start_time = None
                self.speed_adjust_enabled = False
                
            # 2. ระบบปรับความเร็วหมุน (มือเสือ - CLAW)
            if self.is_claw(left_hand):
                px_p, py_p = int(left_hand[0].x * w), int(left_hand[0].y * h)
                
                if self.claw_start_time is None:
                    self.claw_start_time = time.time()
                    # ล็อคจุดอ้างอิง (เส้นกลาง) ทันทีที่เริ่มท่าทาง
                    self.last_claw_y = left_hand[0].y
                
                current_time = time.time()
                elapsed_claw = current_time - self.claw_start_time
                
                # เก็บพิกัดปัจจุบันและจุดอ้างอิงไว้สำหรับวาด UI
                self.draw_claw_ref = (px_p, int(self.last_claw_y * h))
                self.draw_claw_curr = (px_p, py_p)

                if elapsed_claw < self.claw_start_time_cooldown:
                    # จังหวะ HOLD: วาดจุดกึ่งกลางอ้างอิง และวงกลม Cooldown
                    prog = elapsed_claw / self.claw_start_time_cooldown
                    cv2.circle(frame, self.draw_claw_ref, 5, (255, 255, 0), -1) # จุดกลางขณะ Hold
                    cv2.ellipse(frame, (px_p, py_p), (22, 22), 0, 0, int(prog * 360), (255, 255, 0), 2)
                    cv2.putText(frame, "HOLD CLAW", (px_p - 30, py_p - 35), 0, 0.5, (255, 255, 0), 1)
                else:
                    # พ้น Cooldown: เข้าสู่โหมดปรับค่า (Relative Adjust)
                    self.ang_speed_adjust_enabled = True
                    dy_total = self.last_claw_y - left_hand[0].y
                    self.angular_speed = np.clip(self.angular_speed + dy_total * 0.2, self.min_angular, self.max_angular)
            else:
                self.claw_start_time = None
                self.ang_speed_adjust_enabled = False

        # Final Cmd
        self.filtered_x = self.smooth(target_x, self.filtered_x)
        self.filtered_y = self.smooth(target_y, self.filtered_y)
        self.filtered_z = self.smooth(target_z, self.filtered_z)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(self.filtered_x), float(self.filtered_y), float(self.filtered_z)
        self.cmd_pub.publish(cmd)

        # ==========================================================
        # MODERN UI RENDERING
        # ==========================================================
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (40, 40, 40), -1)
        cv2.rectangle(overlay, (w - 260, 10), (w - 10, 180), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # UI Text & Bars
        cv2.putText(frame, "SPEED LIMITS", (25, 35), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"LIN: {self.linear_speed:.2f}", (25, 70), 0, 0.5, (0, 255, 255), 1)
        l_bar = int((self.linear_speed / self.max_linear) * 150)
        cv2.rectangle(frame, (25, 80), (25 + l_bar, 90), (0, 255, 255), -1)
        cv2.putText(frame, f"ANG: {self.angular_speed:.2f}", (25, 120), 0, 0.5, (255, 255, 0), 1)
        a_bar = int((self.angular_speed / self.max_angular) * 150)
        cv2.rectangle(frame, (25, 130), (25 + a_bar, 140), (255, 255, 0), -1)

        st_x = w - 245
        cv2.putText(frame, "LIVE TELEMETRY", (st_x, 35), 0, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"X: {self.filtered_x:.2f}", (st_x, 70), 0, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, f"Y: {self.filtered_y:.2f}", (st_x, 105), 0, 0.6, (100, 255, 255), 2)
        cv2.putText(frame, f"Z: {self.filtered_z:.2f}", (st_x, 140), 0, 0.6, (100, 100, 255), 2)

        # Draw Left UI (Last layer to avoid being covered)
        if self.speed_adjust_enabled and self.draw_pinch_pos:
            px, py = self.draw_pinch_pos
            ref_y = int(self.last_pinch_y * h)
            cv2.line(frame, (px - 50, ref_y), (px + 50, ref_y), (0, 255, 0), 3)
            cv2.line(frame, (px, py), (px, ref_y), (255, 255, 255), 1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), -1)
            
        if hasattr(self, 'draw_pinch_ref') and self.pinch_start_time is not None:
            prx, pry = self.draw_pinch_ref
            pcx, pcy = self.draw_pinch_curr
            
            if self.speed_adjust_enabled:
                # พ้น Hold: วาดเส้นกลางหนาสีเขียว และเส้นเชื่อมระยะ
                cv2.line(frame, (pcx - 50, pry), (pcx + 50, pry), (0, 255, 0), 3) # เส้นกลาง (Anchor Line)
                cv2.line(frame, (pcx, pcy), (pcx, pry), (255, 255, 255), 1) # เส้นเชื่อม
                cv2.circle(frame, (pcx, pcy), 12, (255, 255, 255), -1) # จุดปัจจุบัน
                cv2.putText(frame, "LIN ADJ", (pcx - 30, pcy - 25), 0, 0.5, (0, 255, 0), 2)
            else:
                # อยู่ระหว่าง Hold: วาดจุด Anchor สีขาวนิ่งๆ
                cv2.circle(frame, (prx, pry), 4, (255, 255, 255), -1)

        if hasattr(self, 'draw_claw_ref') and self.claw_start_time is not None:
            rx, ry = self.draw_claw_ref
            cx, cy = self.draw_claw_curr
            
            if self.ang_speed_adjust_enabled:
                # พ้น Hold แล้ว: วาดเส้นกลางหนา และเส้นเชื่อม
                cv2.line(frame, (rx - 50, ry), (rx + 50, ry), (255, 150, 0), 3) # เส้นกลาง
                cv2.line(frame, (rx, ry), (cx, cy), (255, 255, 255), 1) # เส้นเชื่อมระยะ
                cv2.circle(frame, (cx, cy), 12, (255, 255, 0), -1) # จุดปัจจุบัน
            else:
                # ยังอยู่ในช่วง Hold: วาดจุด Anchor เล็กๆ ให้เห็นว่าล็อคตรงนี้
                cv2.circle(frame, (rx, ry), 4, (0, 255, 255), -1)
                
        if self.ang_speed_adjust_enabled and self.draw_claw_ref:
            rx, ry = self.draw_claw_ref
            cx, cy = self.draw_claw_curr
            cv2.line(frame, (rx - 50, ry), (rx + 50, ry), (255, 150, 0), 3)
            cv2.line(frame, (rx, ry), (cx, cy), (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 0), -1)

        cv2.imshow("Dual Hand Control - Frame-driven Mode", frame)

def main():
    rclpy.init()
    node = DualHandSmoothControl()
    try:
        node.run()
    except KeyboardInterrupt: pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()