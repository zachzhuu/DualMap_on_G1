#!/usr/bin/env python3
"""
Data collection script: Subscribe to RealSense RGB images and Livox LiDAR point clouds
Press SPACE to save a pair of data (RGB image + point cloud), press Q to quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from pathlib import Path
import json
from datetime import datetime

# Import Livox custom message type
try:
    from livox_ros_driver2.msg import CustomMsg
    LIVOX_AVAILABLE = True
except ImportError:
    LIVOX_AVAILABLE = False
    print("Warning: livox_ros_driver2 not found. LiDAR data collection will not work.")
    print("Install with: sudo apt install ros-humble-livox-ros-driver2")


# Configuration parameters
# Modify these topic names according to your robot
CONFIG = {
    # ROS2 topic names
    'camera_rgb_topic': '/camera/color/image_raw',
    'camera_info_topic': '/camera/color/camera_info',
    'lidar_topic': '/livox/lidar',
    
    # Data save directory
    'save_dir': './calibration_data',
    
    # Visualization settings
    'rgb_window_name': 'RGB Image (Press SPACE to capture, Q to quit)',
}


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        self.bridge = CvBridge()
        
        # Latest data cache
        self.latest_rgb = None
        self.latest_lidar = None
        self.camera_info = None
        
        self.capture_count = 0
        
        # Create save directory
        self.save_path = Path(CONFIG['save_dir'])
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Create CV2 window
        cv2.namedWindow(CONFIG['rgb_window_name'], cv2.WINDOW_NORMAL)
        
        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            CONFIG['camera_rgb_topic'],
            self.rgb_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            CustomMsg,
            CONFIG['lidar_topic'],
            self.lidar_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            CONFIG['camera_info_topic'],
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info("Data collector node initialized")
        self.get_logger().info(f"RGB topic: {CONFIG['camera_rgb_topic']}")
        self.get_logger().info(f"LiDAR topic: {CONFIG['lidar_topic']}")
        self.get_logger().info(f"Camera info topic: {CONFIG['camera_info_topic']}")
        self.get_logger().info("\nPress SPACE to capture data, press Q to quit")
    
    def rgb_callback(self, msg):
        """Receive RGB image"""
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.display_rgb()
    
    def lidar_callback(self, msg):
        """Receive LiDAR point cloud (Livox CustomMsg format)"""
        self.latest_lidar = msg
        if not hasattr(self, '_lidar_received'):
            self._lidar_received = True
            self.get_logger().info(f"LiDAR data received: {msg.point_num} points")
    
    def livox_to_numpy(self, livox_msg):
        """Convert Livox CustomMsg to numpy array"""
        points_list = []
        for point in livox_msg.points:
            # Extract: x, y, z, reflectivity (intensity), tag, line
            points_list.append([
                point.x,
                point.y,
                point.z,
                point.reflectivity,
                point.tag,
                point.line
            ])
        return np.array(points_list)
    
    def camera_info_callback(self, msg):
        """Receive camera intrinsics (only once)"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera intrinsics received")
    
    def display_rgb(self):
        """Display RGB image with status information"""
        if self.latest_rgb is None:
            return
        
        rgb_display = self.latest_rgb.copy()
        
        # Add status text
        status_text = f"Captures: {self.capture_count}"
        cv2.putText(rgb_display, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.camera_info is None:
            cv2.putText(rgb_display, "Waiting for camera info...", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.latest_lidar is None:
            cv2.putText(rgb_display, "Waiting for LiDAR data...", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(CONFIG['rgb_window_name'], rgb_display)
    
    def save_camera_intrinsics(self):
        """Save camera intrinsics to file"""
        if self.camera_info is None:
            self.get_logger().warn("Warning: Camera intrinsics not received")
            return
        
        # Extract intrinsic matrix
        K = np.array(self.camera_info.k).reshape(3, 3)
        D = np.array(self.camera_info.d)
        
        intrinsics = {
            'width': self.camera_info.width,
            'height': self.camera_info.height,
            'camera_matrix': K.tolist(),
            'distortion_coefficients': D.tolist(),
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2],
        }
        
        # Save to main directory
        intrinsics_file = self.save_path / "camera_intrinsics.json"
        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        
        self.get_logger().info(f"Camera intrinsics saved to: {intrinsics_file}")
    
    def capture_data(self):
        """Capture and save a pair of data"""
        # Check if all data has been received
        if self.latest_rgb is None:
            self.get_logger().warn("RGB image not yet received")
            return False
        
        if self.latest_lidar is None:
            self.get_logger().warn("LiDAR point cloud not yet received")
            return False
        
        if self.camera_info is None:
            self.get_logger().warn("Camera intrinsics not yet received")
            return False
        
        self.get_logger().info(f"\nCapturing data set {self.capture_count + 1}...")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_dir = self.save_path / f"capture_{self.capture_count:03d}_{timestamp}"
        capture_dir.mkdir(exist_ok=True)
        
        # 1. Save RGB image
        rgb_file = capture_dir / "rgb.png"
        cv2.imwrite(str(rgb_file), self.latest_rgb)
        self.get_logger().info(f"  RGB image saved: {rgb_file.name}")
        
        # 2. Save point cloud (Livox custom format)
        # Convert Livox CustomMsg to numpy array
        points_array = self.livox_to_numpy(self.latest_lidar)
        
        # Save as .npy for computation
        pointcloud_npy = capture_dir / "lidar.npy"
        np.save(str(pointcloud_npy), points_array)
        
        # Save as .ply for visualization in CloudCompare
        pointcloud_ply = capture_dir / "lidar.ply"
        self.save_pointcloud_as_ply(points_array, pointcloud_ply)
        
        self.get_logger().info(f"  Point cloud saved: {pointcloud_ply.name} ({len(points_array)} points)")
        
        # 3. Save point cloud field information
        field_names = ['x', 'y', 'z', 'reflectivity', 'tag', 'line']
        metadata = {
            'timestamp': timestamp,
            'pointcloud_fields': field_names,
            'num_points': len(points_array),
            'image_shape': [self.latest_rgb.shape[0], self.latest_rgb.shape[1]],
        }
        
        metadata_file = capture_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 4. If first capture, save camera intrinsics
        if self.capture_count == 0:
            self.save_camera_intrinsics()
        
        self.capture_count += 1
        self.get_logger().info(f"Data set {self.capture_count} captured successfully")
        self.get_logger().info(f"  Total captures: {self.capture_count}\n")
        
        return True
    
    def save_pointcloud_as_ply(self, points_array, output_file):
        """Save point cloud as PLY format for CloudCompare visualization"""
        num_points = len(points_array)
        has_intensity = points_array.shape[1] >= 4
        
        with open(output_file, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_intensity:
                f.write("property float intensity\n")
            f.write("end_header\n")
            
            # Write point data
            for point in points_array:
                if has_intensity:
                    f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")


def main():
    rclpy.init()
    
    collector = DataCollector()
    
    print("\n" + "="*60)
    print("Calibration Data Collection Tool")
    print("="*60)
    print("\nInstructions:")
    print("  1. Place the calibration board (square board with red high-")
    print("     reflectivity circle in center) where both camera and LiDAR")
    print("     can see it")
    print("  2. Check RGB window to verify camera is working")
    print("  3. Press [SPACE] to capture a data set (RGB image + point cloud)")
    print("  4. Move the board to different positions and angles")
    print("  5. Repeat step 3, recommend 5-10 data sets")
    print("  6. Press [Q] to quit")
    print("\nWaiting for sensor data...")
    print("="*60 + "\n")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(collector, timeout_sec=0.01)
            
            # Check for keyboard input from CV2 window
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar
                collector.capture_data()
                print("Press [SPACE] to continue, [Q] to quit\n")
            
            elif key == ord('q') or key == ord('Q'):  # Q key
                print(f"\nCollection complete! Total: {collector.capture_count} data sets")
                print(f"Data saved to: {collector.save_path.absolute()}")
                print("\nNext: run cali_compute.py to compute calibration\n")
                break
            
            elif key == 27:  # ESC key
                print("\nExiting program\n")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user\n")
    
    finally:
        cv2.destroyAllWindows()
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
