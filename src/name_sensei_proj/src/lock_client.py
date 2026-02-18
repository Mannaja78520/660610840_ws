#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from name_sensei_proj.srv import SetMode

class MasterLockClient(Node):
    def __init__(self):
        super().__init__('master_lock_client')
        self.client = self.create_client(SetMode, '/set_master_lock')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for SetMode service...')
        self.req = SetMode.Request()

    def send_request(self, lock_state):
        # แก้ไขตรงนี้ให้ตรงกับฟิลด์ใน SetMode.srv
        self.req.master_lock = lock_state 
        self.future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run name_sensei_proj lock_client [true|false]")
        return

    rclpy.init()
    client = MasterLockClient()
    state = sys.argv[1].lower() == 'true'
    
    response = client.send_request(state)
    client.get_logger().info(f"Response: {response.message}")
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()