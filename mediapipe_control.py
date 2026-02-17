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

        # ==========================================================
        # ⚙️ CONFIGURABLE VARIABLES (ปรับจูนที่นี่)
        # ==========================================================
        
        # 1. Linear Speed Limits (มือขวา + มือซ้ายปรับค่า)
        self.min_linear = 0.1
        self.max_linear = 1.0
        self.linear_speed = 0.15
        
        # 2. Angular Speed Limits (เลี้ยวมือขวา + มือซ้ายปรับค่า)
        self.min_angular = 0.2
        self.max_angular = 2.0
        self.angular_speed = 1.0 
        self.rot_gain = 12.0           # ตัวคูณความเร็วการเลี้ยวตามการพับนิ้ว
        self.rot_ui_gain = 20.0        # ความยาวของ Arc สีม่วง/เหลืองบนหน้าจอ

        # 3. Smoothing & Deadzone
        self.alpha = 0.25              # 0.1 หน่วงมาก, 0.9 ตอบสนองเร็ว
        self.deadzone = 0.15           # ระยะไข่แดงจอยที่ไม่ขยับ
        self.joy_radius = 0.13         # รัศมีจอยสติ๊กจำลอง

        # 4. Right Hand Latching (การดูดศูนย์กลาง)
        self.latch_duration = 0.135     # เวลาดูด (วินาที)
        self.latch_speed = 0.18         # ความแรงในการดูด (0-1)
        self.latch_trigger_dist = 0.4  # รัศมีที่จะเริ่มดูด (0.5 = ครึ่งหนึ่งของ joy_radius)

        # 5. Left Hand Adjustments (ความไวในการรูดนิ้ว)
        self.lin_adj_sensitivity = 0.1 # ความเร็วเปลี่ยนเท่าไหร่เมื่อรูดนิ้ว (Pinch)
        self.ang_adj_sensitivity = 0.2 # ความเร็วเปลี่ยนเท่าไหร่เมื่อรูดนิ้ว (Claw)
        self.pinch_start_time_cooldown = 1.0
        self.claw_start_time_cooldown = 1.0

        # 6. Gesture Thresholds (เงื่อนไขท่าทาง)
        self.pinch_threshold = 0.1
        self.pinch_tolerance = 1.7     # ตัวคูณให้หลุดยากขึ้นตอนกำลังรูดปรับค่า
        self.pinch_joint_dist = 0.16
        self.pinch_middle_gap = 0.05
        
        self.claw_threshold = 0.135
        self.claw_tolerance = 0.45      # ตัวคูณสำหรับ Claw Mode
        self.claw_index_mid_gap = 0.08
        self.claw_thumb_idx_gap = 0.10
        self.fold_trigger = 0.025      # ระยะพับนิ้วเพื่อเริ่มเลี้ยว

        # 7. UI Appearance
        self.ui_bg_color = (40, 40, 40)
        self.ui_telemetry_bg = (20, 20, 20)
        self.ui_lin_color = (0, 255, 255)
        self.ui_ang_color = (255, 255, 0)
        self.ui_turn_left_color = (255, 0, 255)
        self.ui_turn_right_color = (0, 255, 255)
        self.ui_joy_edge_color = (255, 255, 0)
        self.ui_anchor_color = (255, 255, 255)
        self.ui_bar_width = 150

        # ==========================================================
        # 🛠️ INTERNAL SYSTEM (ไม่ต้องแก้ไข)
        # ==========================================================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_command', qos_profile_sensor_data)
        self.filtered_x = self.filtered_y = self.filtered_z = 0.0
        self.joy_center = None
        self.pinch_start_time = self.claw_start_time = self.latch_start_time = None
        self.speed_adjust_enabled = self.ang_speed_adjust_enabled = False
        self.last_pinch_y = self.last_claw_y = None
        self.draw_pinch_pos = self.draw_pinch_ref = self.draw_pinch_curr = None
        self.draw_claw_ref = self.draw_claw_curr = None

        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.get_logger().info("Dual Hand Control - Variable Driven Mode Started")

    def smooth(self, target, current):
        return self.alpha * target + (1 - self.alpha) * current

    def is_pinch(self, hand):
        dist_tips = np.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        dist_joints = np.sqrt((hand[3].x - hand[6].x)**2 + (hand[3].y - hand[6].y)**2)
        dist_middle = np.sqrt((hand[8].x - hand[12].x)**2 + (hand[8].y - hand[12].y)**2)
        thresh = self.pinch_threshold * self.pinch_tolerance if self.speed_adjust_enabled else self.pinch_threshold
        return dist_tips < thresh and dist_joints < self.pinch_joint_dist and dist_middle > self.pinch_middle_gap
    
    def is_claw(self, hand):
        thresh = self.claw_threshold * self.claw_tolerance if self.ang_speed_adjust_enabled else self.claw_threshold
        dist_stretch = np.sqrt((hand[4].x - hand[20].x)**2 + (hand[4].y - hand[20].y)**2)
        dist_index_middle = np.sqrt((hand[8].x - hand[12].x)**2 + (hand[8].y - hand[12].y)**2)
        dist_thumb_index = np.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        return dist_stretch > thresh and dist_index_middle > self.claw_index_mid_gap and dist_thumb_index > self.claw_thumb_idx_gap

    def is_fist(self, hand):
        return (hand[8].y > hand[6].y and hand[12].y > hand[10].y and hand[16].y > hand[14].y)

    def run(self):
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            self.process_frame(frame)
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

        if right_hand:
            if self.is_fist(right_hand):
                target_x = target_y = target_z = 0.0
                self.filtered_x = self.filtered_y = self.filtered_z = 0.0
                self.joy_center = self.latch_start_time = None
                cv2.putText(frame, "EMERGENCY STOP", (int(w/2)-180, h-50), 0, 1.2, (0, 0, 255), 3)
            else:
                idx_tip_r = right_hand[8]
                current_time = time.time()
                if self.joy_center is None: self.joy_center = (idx_tip_r.x, idx_tip_r.y)
                
                dx_raw = idx_tip_r.x - self.joy_center[0]
                dy_raw = idx_tip_r.y - self.joy_center[1]
                dist_raw = np.sqrt(dx_raw**2 + dy_raw**2)

                fold_l = right_hand[12].y - right_hand[9].y
                fold_r = right_hand[16].y - right_hand[13].y

                is_turning = False
                if fold_l > self.fold_trigger: 
                    target_z = fold_l * self.angular_speed * self.rot_gain
                    is_turning = True
                elif fold_r > self.fold_trigger: 
                    target_z = -fold_r * self.angular_speed * self.rot_gain
                    is_turning = True

                if is_turning:
                    if self.latch_start_time is None: self.latch_start_time = current_time
                    elapsed_latch = current_time - self.latch_start_time
                    if elapsed_latch < self.latch_duration and dist_raw < (self.joy_radius * self.latch_trigger_dist):
                        self.joy_center = (
                            self.joy_center[0] * (1 - self.latch_speed) + idx_tip_r.x * self.latch_speed,
                            self.joy_center[1] * (1 - self.latch_speed) + idx_tip_r.y * self.latch_speed
                        )
                        cv2.putText(frame, "LATCHING...", (int(idx_tip_r.x*w)+20, int(idx_tip_r.y*h)), 0, 0.5, (0, 255, 255), 1)
                else: self.latch_start_time = None

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

                cx, cy = int(self.joy_center[0]*w), int(self.joy_center[1]*h)
                joy_px_radius = int(self.joy_radius * w)
                cv2.circle(frame, (cx, cy), joy_px_radius, self.ui_joy_edge_color, 1)
                
                if is_turning:
                    start_angle = -90 
                    end_angle = start_angle - (target_z * self.rot_ui_gain)
                    color = self.ui_turn_left_color if target_z > 0 else self.ui_turn_right_color
                    cv2.ellipse(frame, (cx, cy), (joy_px_radius + 10, joy_px_radius + 10), 0, start_angle, end_angle, color, 5)
                    cv2.putText(frame, "LEFT" if target_z > 0 else "RIGHT", (cx - 30, cy - joy_px_radius - 20), 0, 0.7, color, 2)

                tip_px = (int(idx_tip_r.x*w), int(idx_tip_r.y*h))
                cv2.line(frame, (cx, cy), tip_px, (0, 255, 0), 2)
                cv2.circle(frame, tip_px, 10, (0, 255, 0), -1)
        else: self.joy_center = None

        if left_hand:
            current_time = time.time()
            if self.is_pinch(left_hand) and not self.is_claw(left_hand):
                px_t, py_t = int(left_hand[8].x * w), int(left_hand[8].y * h)
                if self.pinch_start_time is None:
                    self.pinch_start_time = current_time
                    self.last_pinch_y = left_hand[8].y
                elapsed_pinch = current_time - self.pinch_start_time
                self.draw_pinch_ref = (px_t, int(self.last_pinch_y * h))
                self.draw_pinch_curr = (px_t, py_t)
                if elapsed_pinch < self.pinch_start_time_cooldown:
                    prog_p = elapsed_pinch / self.pinch_start_time_cooldown
                    cv2.circle(frame, self.draw_pinch_ref, 5, self.ui_anchor_color, -1)
                    cv2.ellipse(frame, (px_t, py_t), (18, 18), 0, 0, int(prog_p * 360), self.ui_anchor_color, 2)
                else:
                    self.speed_adjust_enabled = True
                    dy_lin = self.last_pinch_y - left_hand[8].y
                    self.linear_speed = np.clip(self.linear_speed + dy_lin * self.lin_adj_sensitivity, self.min_linear, self.max_linear)
            else:
                self.pinch_start_time = None
                self.speed_adjust_enabled = False
                
            if self.is_claw(left_hand):
                px_p, py_p = int(left_hand[0].x * w), int(left_hand[0].y * h)
                if self.claw_start_time is None:
                    self.claw_start_time = current_time
                    self.last_claw_y = left_hand[0].y
                elapsed_claw = current_time - self.claw_start_time
                self.draw_claw_ref = (px_p, int(self.last_claw_y * h))
                self.draw_claw_curr = (px_p, py_p)
                if elapsed_claw < self.claw_start_time_cooldown:
                    prog = elapsed_claw / self.claw_start_time_cooldown
                    cv2.circle(frame, self.draw_claw_ref, 5, self.ui_ang_color, -1)
                    cv2.ellipse(frame, (px_p, py_p), (22, 22), 0, 0, int(prog * 360), self.ui_ang_color, 2)
                else:
                    self.ang_speed_adjust_enabled = True
                    dy_total = self.last_claw_y - left_hand[0].y
                    self.angular_speed = np.clip(self.angular_speed + dy_total * self.ang_adj_sensitivity, self.min_angular, self.max_angular)
            else:
                self.claw_start_time = None
                self.ang_speed_adjust_enabled = False

        self.filtered_x = self.smooth(target_x, self.filtered_x)
        self.filtered_y = self.smooth(-target_y, self.filtered_y)
        self.filtered_z = self.smooth(target_z, self.filtered_z)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(self.filtered_x), float(self.filtered_y), float(self.filtered_z)
        self.cmd_pub.publish(cmd)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), self.ui_bg_color, -1)
        cv2.rectangle(overlay, (w - 260, 10), (w - 10, 180), self.ui_telemetry_bg, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "SPEED LIMITS", (25, 35), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"LIN: {self.linear_speed:.3f}", (25, 70), 0, 0.5, self.ui_lin_color, 1)
        l_bar = int((self.linear_speed / self.max_linear) * self.ui_bar_width)
        cv2.rectangle(frame, (25, 80), (25 + l_bar, 90), self.ui_lin_color, -1)
        cv2.putText(frame, f"ANG: {self.angular_speed:.3f}", (25, 120), 0, 0.5, self.ui_ang_color, 1)
        a_bar = int((self.angular_speed / self.max_angular) * self.ui_bar_width)
        cv2.rectangle(frame, (25, 130), (25 + a_bar, 140), self.ui_ang_color, -1)

        st_x = w - 245
        cv2.putText(frame, "LIVE TELEMETRY", (st_x, 35), 0, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"X: {self.filtered_x:.3f}", (st_x, 70), 0, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, f"Y: {self.filtered_y:.3f}", (st_x, 105), 0, 0.6, (100, 255, 255), 2)
        cv2.putText(frame, f"Z: {self.filtered_z:.3f}", (st_x, 140), 0, 0.6, (100, 100, 255), 2)

        if hasattr(self, 'draw_pinch_ref') and self.pinch_start_time is not None:
            prx, pry = self.draw_pinch_ref
            pcx, pcy = self.draw_pinch_curr
            if self.speed_adjust_enabled:
                cv2.line(frame, (pcx - 50, pry), (pcx + 50, pry), (0, 255, 0), 3)
                cv2.line(frame, (pcx, pcy), (pcx, pry), (255, 255, 255), 1)
                cv2.circle(frame, (pcx, pcy), 12, (255, 255, 255), -1)
                cv2.putText(frame, "LIN ADJ", (pcx - 30, pcy - 25), 0, 0.5, (0, 255, 0), 2)
            else: cv2.circle(frame, (prx, pry), 4, self.ui_anchor_color, -1)

        if hasattr(self, 'draw_claw_ref') and self.claw_start_time is not None:
            rx, ry = self.draw_claw_ref
            cx, cy = self.draw_claw_curr
            if self.ang_speed_adjust_enabled:
                cv2.line(frame, (rx - 50, ry), (rx + 50, ry), (255, 150, 0), 3)
                cv2.line(frame, (rx, ry), (cx, cy), (255, 255, 255), 1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 0), -1)
            else: cv2.circle(frame, (rx, ry), 4, self.ui_ang_color, -1)

        cv2.imshow("Dual Hand Control - Frame-driven Mode", frame)

def main():
    rclpy.init()
    node = DualHandSmoothControl()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()