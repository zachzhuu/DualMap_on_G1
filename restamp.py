#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, CameraInfo
import subprocess
import sys
import select
import termios
import tty
import os
from pathlib import Path
from datetime import datetime


class RealsenseRestamper(Node):
    def __init__(self, bag_name=None):
        super().__init__('realsense_restamper')

        # ===== 录制相关 =====
        self.bag_name = bag_name
        self.bag_dir = Path('./bag')
        self.bag_dir.mkdir(exist_ok=True)
        self.recording_process = None
        self.is_recording = False
        
        # 频率控制（15 Hz）
        self.publish_rate = 15.0  # Hz
        self.min_interval = 1.0 / self.publish_rate
        self.last_publish_time = 0.0
        
        # 终端设置（用于非阻塞键盘输入）
        self.old_settings = None
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            self.get_logger().warn("Cannot set terminal to raw mode. Keyboard input may not work.")

        # ===== 1) 参数：原始话题 / 输出话题 =====
        # 原始话题（保持和 realsense2_camera 一致）
        self.declare_parameter('rgb_in', '/camera/color/image_raw')
        self.declare_parameter('depth_in', '/camera/depth/image_rect_raw')
        # 你现在 Runner 用的是 depth 的 camera_info：
        # Camera Info: /camera/depth/camera_info_restamped
        self.declare_parameter('info_in', '/camera/depth/camera_info')

        # 输出话题（runner_ros 订阅这些）
        self.declare_parameter('rgb_out', '/camera/color/image_restamped')
        self.declare_parameter('depth_out', '/camera/depth/image_restamped')
        self.declare_parameter('info_out', '/camera/depth/camera_info_restamped')

        rgb_in = self.get_parameter('rgb_in').get_parameter_value().string_value
        depth_in = self.get_parameter('depth_in').get_parameter_value().string_value
        info_in = self.get_parameter('info_in').get_parameter_value().string_value

        rgb_out = self.get_parameter('rgb_out').get_parameter_value().string_value
        depth_out = self.get_parameter('depth_out').get_parameter_value().string_value
        info_out = self.get_parameter('info_out').get_parameter_value().string_value

        # ===== 2) QoS：订阅端用 SensorData（BEST_EFFORT），发布端用 RELIABLE =====

        # 订阅 RealSense：用 sensor_data_qos（和相机保持一致）
        sub_qos = qos_profile_sensor_data  # BEST_EFFORT，深度小，低延迟

        # 发布给 runner_ros：单独建一个 RELIABLE 的 QoS
        pub_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # ===== 3) 订阅原始话题 =====
        self.rgb_sub = self.create_subscription(
            Image, rgb_in, self.rgb_cb, sub_qos)
        self.depth_sub = self.create_subscription(
            Image, depth_in, self.depth_cb, sub_qos)
        self.info_sub = self.create_subscription(
            CameraInfo, info_in, self.info_cb, 10)  # CameraInfo 不需要特别低延迟

        # ===== 4) 发布重打时间戳后的话题 =====
        self.rgb_pub = self.create_publisher(Image, rgb_out, pub_qos)
        self.depth_pub = self.create_publisher(Image, depth_out, pub_qos)
        self.info_pub = self.create_publisher(CameraInfo, info_out, pub_qos)

        self.get_logger().info(
            f"Restamping:\n"
            f"  RGB:   {rgb_in} -> {rgb_out}\n"
            f"  Depth: {depth_in} -> {depth_out}\n"
            f"  Info:  {info_in} -> {info_out}\n"
            f"  Sub QoS:  SensorData (BEST_EFFORT)\n"
            f"  Pub QoS:  depth=10, RELIABLE"
        )
        
        if self.bag_name:
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"Recording Mode Enabled\n"
                f"  Bag name: {self.bag_name}\n"
                f"  Bag dir:  {self.bag_dir.absolute()}\n"
                f"  Rate:     15 Hz\n"
                f"  Topics:   {rgb_out}, {depth_out}, {info_out}, /Odometry\n"
                f"\n"
                f"  Press 'S' to START recording\n"
                f"  Press 'Q' to STOP recording and quit\n"
                f"{'='*60}"
            )
            # 创建定时器检查键盘输入
            self.timer = self.create_timer(0.1, self.check_keyboard)
        else:
            self.get_logger().info("Recording disabled. Run with --bag-name to enable.")

    def check_keyboard(self):
        """检查键盘输入（非阻塞）"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            
            if key == 's' and not self.is_recording:
                self.start_recording()
            elif key == 'q':
                if self.is_recording:
                    self.stop_recording()
                self.get_logger().info("Quitting...")
                rclpy.shutdown()
    
    def start_recording(self):
        """开始录制 rosbag"""
        if self.is_recording:
            self.get_logger().warn("Already recording!")
            return
        
        # 添加时间戳到bag名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = self.bag_dir / f"{self.bag_name}_{timestamp}"
        
        # 获取输出话题名称
        rgb_out = self.get_parameter('rgb_out').get_parameter_value().string_value
        depth_out = self.get_parameter('depth_out').get_parameter_value().string_value
        info_out = self.get_parameter('info_out').get_parameter_value().string_value
        
        # 构建 ros2 bag record 命令
        cmd = [
            'ros2', 'bag', 'record',
            '-o', str(bag_path),
            '--max-bag-duration', '0',  # 不限制单个bag文件大小
            rgb_out,
            depth_out,
            info_out,
            '/Odometry',
        ]
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"RECORDING STARTED")
        self.get_logger().info(f"  Saving to: {bag_path}")
        self.get_logger().info(f"  Press 'Q' to stop and quit")
        self.get_logger().info(f"{'='*60}\n")
        
        try:
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.is_recording = True
        except Exception as e:
            self.get_logger().error(f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """停止录制 rosbag"""
        if not self.is_recording or self.recording_process is None:
            return
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("STOPPING RECORDING...")
        
        try:
            self.recording_process.terminate()
            self.recording_process.wait(timeout=5)
            self.get_logger().info("Recording stopped successfully")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Timeout waiting for recording to stop, forcing...")
            self.recording_process.kill()
        except Exception as e:
            self.get_logger().error(f"Error stopping recording: {e}")
        
        self.is_recording = False
        self.recording_process = None
        self.get_logger().info(f"{'='*60}\n")
    
    def destroy_node(self):
        """清理资源"""
        # 恢复终端设置
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
        
        # 停止录制
        if self.is_recording:
            self.stop_recording()
        
        super().destroy_node()

    def _now_stamp(self):
        return self.get_clock().now().to_msg()
    
    def _should_publish(self):
        """检查是否应该发布（频率限制）"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_publish_time >= self.min_interval:
            self.last_publish_time = current_time
            return True
        return False

    def rgb_cb(self, msg: Image):
        if self._should_publish():
            msg.header.stamp = self._now_stamp()
            self.rgb_pub.publish(msg)

    def depth_cb(self, msg: Image):
        if self._should_publish():
            msg.header.stamp = self._now_stamp()
            self.depth_pub.publish(msg)

    def info_cb(self, msg: CameraInfo):
        if self._should_publish():
            msg.header.stamp = self._now_stamp()
            self.info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Restamp RealSense topics and optionally record to rosbag')
    parser.add_argument('--bag-name', type=str, default=None,
                       help='Name of the rosbag (without timestamp). If provided, enables recording mode.')
    
    # 只解析我们的参数，剩余的传给 ROS2
    parsed_args, unknown = parser.parse_known_args()
    
    node = RealsenseRestamper(bag_name=parsed_args.bag_name)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
