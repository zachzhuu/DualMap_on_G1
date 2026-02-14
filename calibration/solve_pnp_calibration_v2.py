#!/usr/bin/env python3
"""
Camera-Lidar Extrinsic Calibration using PnP Algorithm
Directly processes RGB images and lidar point clouds to extract circle centers,
then solves for extrinsic parameters
"""

import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d


# Configuration
CONFIG = {
    # Input directories
    'calibration_data_dirs': ['calibration_data/0', 'calibration_data/2'],
    
    # Output directory
    'output_dir': 'calibration_results',
    
    # RGB circle detection parameters
    'rgb_min_area': 1000,
    'rgb_max_area': 100000,
    
    # Lidar reflectivity filtering parameters
    'reflectivity_percentile': 95,
    'min_reflectivity_ratio': 0.7,
    'circle_ransac_threshold': 0.01,
    'circle_ransac_iterations': 1000,
    'min_circle_points': 10,
    
    # Outlier removal parameters
    'nb_neighbors': 20,
    'std_ratio': 2.0,
    
    # Initial guess for extrinsic parameters
    # Lidar: x forward, y right, z down
    # Camera: x right, y down, z forward
    # Initial pitch angle: -44.731 degrees around lidar y-axis
    # Initial translation (camera center relative to lidar): [0.058, -0.01, -0.034]
    'initial_pitch_deg': -44.731,
    'initial_translation': np.array([0.058, -0.01, -0.034]),
    
    # Reprojection error threshold (pixels)
    'max_reprojection_error': 3.0,
}


def detect_white_circle_from_rgb(image):
    """
    Detect white circle calibration board in RGB image
    
    Args:
        image: Input RGB image
    
    Returns:
        ellipse: Fitted ellipse parameters ((cx, cy), (major, minor), angle) or None
        center_2d: Circle center in image coordinates [x, y] or None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < CONFIG['rgb_min_area'] or area > CONFIG['rgb_max_area']:
            continue
        
        # Need at least 5 points to fit ellipse
        if len(contour) < 5:
            continue
        
        # Fit ellipse
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            
            # Check aspect ratio
            aspect_ratio = max(major, minor) / (min(major, minor) + 1e-6)
            if aspect_ratio > 3.0:
                continue
            
            # Check area matching
            ellipse_area = np.pi * major * minor / 4
            area_ratio = min(area, ellipse_area) / (max(area, ellipse_area) + 1e-6)
            if area_ratio < 0.7:
                continue
            
            # Circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            # Score
            score = area_ratio * 0.5 + circularity * 0.5
            
            if score > best_score:
                best_score = score
                best_ellipse = ellipse
                
        except Exception as e:
            continue
    
    if best_ellipse is None:
        return None, None
    
    (cx, cy), (major, minor), angle = best_ellipse
    center_2d = np.array([cx, cy])
    
    return best_ellipse, center_2d


def filter_high_reflectivity_points(points):
    """
    Filter high-reflectivity points from lidar point cloud
    
    Args:
        points: Nx6 array [x, y, z, reflectivity, tag, line]
    
    Returns:
        high_reflectivity_points: Filtered points (xyz only)
    """
    if points.shape[1] < 4:
        return None
    
    xyz = points[:, :3]
    reflectivity = points[:, 3]
    
    # Apply filtering
    reflectivity_threshold = np.percentile(reflectivity, CONFIG['reflectivity_percentile'])
    reflectivity_max = np.max(reflectivity)
    reflectivity_min_value = reflectivity_max * CONFIG['min_reflectivity_ratio']
    reflectivity_threshold = max(reflectivity_threshold, reflectivity_min_value)
    
    high_reflectivity_mask = reflectivity >= reflectivity_threshold
    high_reflectivity_points = xyz[high_reflectivity_mask]
    
    return high_reflectivity_points


def remove_spatial_outliers(points):
    """
    Remove spatial outliers using statistical outlier removal
    
    Args:
        points: Nx3 point cloud array
        
    Returns:
        inlier_points: Filtered points after outlier removal
    """
    if len(points) < CONFIG['nb_neighbors']:
        return points
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=CONFIG['nb_neighbors'],
        std_ratio=CONFIG['std_ratio']
    )
    
    inlier_points = points[ind]
    
    return inlier_points


def fit_circle_center_3d(points):
    """
    Fit circle center from 3D points
    
    Args:
        points: Nx3 point cloud array
        
    Returns:
        center_3d: Circle center coordinates [x, y, z]
    """
    # Compute centroid as circle center
    center_3d = np.mean(points, axis=0)
    return center_3d


def process_single_capture(capture_dir, parent_dir_name):
    """
    Process a single capture directory to extract 2D and 3D circle centers
    
    Args:
        capture_dir: Path to capture directory
        parent_dir_name: Name of parent directory (0 or 2)
        
    Returns:
        result: Dictionary with detection results or None
    """
    capture_name = capture_dir.name
    
    # Load RGB image
    rgb_path = capture_dir / 'rgb.png'
    if not rgb_path.exists():
        return None
    
    rgb_image = cv2.imread(str(rgb_path))
    if rgb_image is None:
        return None
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Detect circle in RGB
    ellipse, center_2d = detect_white_circle_from_rgb(rgb_image)
    if center_2d is None:
        print(f"  {capture_name}: RGB detection failed")
        return None
    
    # Load lidar data
    lidar_path = capture_dir / 'lidar.npy'
    if not lidar_path.exists():
        return None
    
    lidar_points = np.load(lidar_path)
    
    # Filter high-reflectivity points
    high_refl_points = filter_high_reflectivity_points(lidar_points)
    if high_refl_points is None or len(high_refl_points) < 10:
        print(f"  {capture_name}: Lidar filtering failed")
        return None
    
    # Remove outliers
    filtered_points = remove_spatial_outliers(high_refl_points)
    if len(filtered_points) < 3:
        print(f"  {capture_name}: Too few points after outlier removal")
        return None
    
    # Fit circle center
    center_3d = fit_circle_center_3d(filtered_points)
    
    print(f"  {capture_name}: OK - 2D center: [{center_2d[0]:.1f}, {center_2d[1]:.1f}], "
          f"3D center: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}], "
          f"Points: {len(filtered_points)}")
    
    return {
        'capture_name': capture_name,
        'parent_dir': parent_dir_name,
        'center_2d': center_2d,
        'center_3d': center_3d,
        'ellipse': {
            'center_x': float(ellipse[0][0]),
            'center_y': float(ellipse[0][1]),
            'major_axis': float(ellipse[1][0]),
            'minor_axis': float(ellipse[1][1]),
            'angle': float(ellipse[2]),
        },
        'points_count': len(filtered_points),
    }


def load_camera_intrinsics(json_file):
    """Load camera intrinsic parameters"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['distortion_coefficients'])
    
    return camera_matrix, dist_coeffs

def get_initial_extrinsics():
    """
    Compute initial guess for extrinsic parameters.

    Assumptions:
    - Lidar frame L: x forward, y right, z down
    - Camera frame C: x right, y down, z forward
    - CONFIG['initial_translation'] is an initial guess of lidar origin in camera frame (t_CL).
    - We need R_CL, t_CL such that: X_c = R_CL * X_L + t_CL
    """

    pitch_rad = np.deg2rad(CONFIG['initial_pitch_deg'])

    # 1) Rotation around lidar y-axis (in lidar frame)
    R_y = np.array([
        [np.cos(pitch_rad), 0.0,               -np.sin(pitch_rad)],
        [0.0,               1.0,               0.0             ],
        [np.sin(pitch_rad),0.0,               np.cos(pitch_rad)]
    ], dtype=np.float64)

    # 2) Axis alignment: L (forward, right, down) -> C (right, down, forward)
    # Mapping (Lx, Ly, Lz) -> (Cx, Cy, Cz) = (Ly, Lz, Lx)
    R_align = np.array([
        [0.0, 1.0, 0.0],   # Cx
        [0.0, 0.0, 1.0],   # Cy
        [1.0, 0.0, 0.0]    # Cz
    ], dtype=np.float64)

    # 3) Total rotation: lidar → camera
    R_CL = R_align @ R_y

    # 4) Translation:
    #    Treat CONFIG['initial_translation'] directly as t_CL
    #    (lidar origin in camera frame) instead of trying to
    #    convert from camera-in-lidar again.
    t_CL = CONFIG['initial_translation'].astype(np.float64).reshape(3, 1)

    # 5) Convert to rvec / tvec for OpenCV
    rvec_init, _ = cv2.Rodrigues(R_CL)
    tvec_init = t_CL.copy()

    return rvec_init, tvec_init, R_CL, t_CL.reshape(3,)


def solve_pnp_calibration(matched_pairs, camera_matrix, dist_coeffs):
    """
    Solve PnP for all matched pairs
    
    Args:
        matched_pairs: List of matched detection pairs
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    
    Returns:
        result: Dictionary with calibration results
    """
    print(f"\n{'='*60}")
    print(f"Solving PnP Calibration")
    print(f"{'='*60}")
    
    if len(matched_pairs) < 4:
        print(f"  Error: Need at least 4 points for PnP, got {len(matched_pairs)}")
        return None
    
    # Prepare data
    object_points = np.array([pair['center_3d'] for pair in matched_pairs], dtype=np.float32)
    image_points = np.array([pair['center_2d'] for pair in matched_pairs], dtype=np.float32)
    
    print(f"  Number of point pairs: {len(matched_pairs)}")
    
    # Get initial guess
    rvec_init, tvec_init, R_init, t_init = get_initial_extrinsics()
    
    # Solve PnP with iterative refinement
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("  Error: PnP solving failed")
        return None
    
    # Convert rotation vector to matrix
    R_final, _ = cv2.Rodrigues(rvec)
    t_final = tvec.flatten()
    
    # Compute reprojection error
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    
    reprojection_errors = np.linalg.norm(projected_points - image_points, axis=1)
    mean_error = np.mean(reprojection_errors)
    max_error = np.max(reprojection_errors)
    
    # Print results
    print(f"\n  === Initial Extrinsics (Input) ===")
    print(f"  Rotation matrix:")
    print(f"    [{R_init[0,0]:10.6f}, {R_init[0,1]:10.6f}, {R_init[0,2]:10.6f}]")
    print(f"    [{R_init[1,0]:10.6f}, {R_init[1,1]:10.6f}, {R_init[1,2]:10.6f}]")
    print(f"    [{R_init[2,0]:10.6f}, {R_init[2,1]:10.6f}, {R_init[2,2]:10.6f}]")
    print(f"  Translation vector:")
    print(f"    [{t_init[0]:10.6f}, {t_init[1]:10.6f}, {t_init[2]:10.6f}]")
    
    print(f"\n  === Optimized Extrinsics (Output) ===")
    print(f"  Rotation matrix:")
    print(f"    [{R_final[0,0]:10.6f}, {R_final[0,1]:10.6f}, {R_final[0,2]:10.6f}]")
    print(f"    [{R_final[1,0]:10.6f}, {R_final[1,1]:10.6f}, {R_final[1,2]:10.6f}]")
    print(f"    [{R_final[2,0]:10.6f}, {R_final[2,1]:10.6f}, {R_final[2,2]:10.6f}]")
    print(f"  Translation vector:")
    print(f"    [{t_final[0]:10.6f}, {t_final[1]:10.6f}, {t_final[2]:10.6f}]")
    
    # Convert to Euler angles
    rotation = Rotation.from_matrix(R_final)
    euler_xyz = rotation.as_euler('xyz', degrees=True)
    euler_zyx = rotation.as_euler('zyx', degrees=True)
    
    print(f"\n  Euler angles (XYZ): [{euler_xyz[0]:.2f}, {euler_xyz[1]:.2f}, {euler_xyz[2]:.2f}] degrees")
    print(f"  Euler angles (ZYX): [{euler_zyx[0]:.2f}, {euler_zyx[1]:.2f}, {euler_zyx[2]:.2f}] degrees")
    
    print(f"\n  Reprojection errors:")
    print(f"    Mean: {mean_error:.4f} pixels")
    print(f"    Max: {max_error:.4f} pixels")
    print(f"    Min: {np.min(reprojection_errors):.4f} pixels")
    print(f"    Std: {np.std(reprojection_errors):.4f} pixels")
    
    # Check outliers
    outliers = reprojection_errors > CONFIG['max_reprojection_error']
    num_outliers = np.sum(outliers)
    
    if num_outliers > 0:
        print(f"\n  Warning: {num_outliers} points have reprojection error > {CONFIG['max_reprojection_error']} pixels")
    else:
        print(f"\n  ✓ All reprojection errors < {CONFIG['max_reprojection_error']} pixels")
    
    return {
        'num_points': len(matched_pairs),
        'R_init': R_init,
        't_init': t_init,
        'R': R_final,
        't': t_final,
        'rvec': rvec,
        'tvec': tvec,
        'euler_xyz': euler_xyz,
        'euler_zyx': euler_zyx,
        'reprojection_errors': reprojection_errors,
        'mean_error': mean_error,
        'max_error': max_error,
        'pairs': matched_pairs,
        'object_points': object_points,
        'image_points': image_points,
        'projected_points': projected_points,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
    }


def visualize_reprojection_on_images(result, output_dir, calibration_data_dirs):
    """
    Visualize reprojection on each individual RGB image
    
    Args:
        result: Calibration result dictionary
        output_dir: Output directory for visualization
        calibration_data_dirs: List of calibration data directories
    """
    pairs = result['pairs']
    projected_points = result['projected_points']
    reprojection_errors = result['reprojection_errors']
    
    # Create subdirectory for individual images
    images_dir = output_dir / 'reprojection_images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Visualizing reprojection on individual images...")
    
    # Create directory mapping
    dir_map = {}
    for dir_str in calibration_data_dirs:
        dir_path = Path(dir_str)
        dir_map[dir_path.name] = dir_path
    
    for i, pair in enumerate(pairs):
        parent_dir = pair['parent_dir']
        capture_name = pair['capture_name']
        
        # Load RGB image
        rgb_path = dir_map[parent_dir] / capture_name / 'rgb.png'
        if not rgb_path.exists():
            continue
        
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get points for this image
        detected_center = pair['center_2d']
        reprojected_center = projected_points[i]
        error = reprojection_errors[i]
        ellipse = pair['ellipse']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_rgb)
        ax.set_title(f'{capture_name}\nReprojection Error: {error:.4f} pixels', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw detected ellipse
        ellipse_patch = plt.matplotlib.patches.Ellipse(
            (ellipse['center_x'], ellipse['center_y']),
            ellipse['major_axis'], ellipse['minor_axis'],
            angle=ellipse['angle'], fill=False, edgecolor='cyan', linewidth=2,
            label='Detected ellipse'
        )
        ax.add_patch(ellipse_patch)
        
        # Draw detected center
        ax.plot(detected_center[0], detected_center[1], 'bo', markersize=12, 
               label='Detected center', zorder=5)
        
        # Draw reprojected center
        ax.plot(reprojected_center[0], reprojected_center[1], 'rx', markersize=12,
               markeredgewidth=3, label='Reprojected center', zorder=5)
        
        # Draw error line
        ax.plot([detected_center[0], reprojected_center[0]],
               [detected_center[1], reprojected_center[1]],
               'g-', linewidth=2, alpha=0.7, label=f'Error: {error:.2f}px')
        
        # Add error text
        mid_x = (detected_center[0] + reprojected_center[0]) / 2
        mid_y = (detected_center[1] + reprojected_center[1]) / 2
        ax.text(mid_x, mid_y - 20, f'{error:.2f}px',
               fontsize=10, color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7),
               ha='center')
        
        ax.legend(loc='upper right', fontsize=10)
        
        # Save figure
        output_file = images_dir / f'{parent_dir}_{capture_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    Saved {len(pairs)} visualization images to: {images_dir}")


def visualize_reprojection_summary(result, output_dir):
    """
    Visualize reprojection error summary
    """
    image_points = result['image_points']
    projected_points = result['projected_points']
    reprojection_errors = result['reprojection_errors']
    pairs = result['pairs']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Image space with reprojection
    ax = axes[0]
    ax.set_title(f'Reprojection Visualization - Combined Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Plot original detections
    ax.scatter(image_points[:, 0], image_points[:, 1], 
              c='blue', s=100, marker='o', label='Detected centers', zorder=3)
    
    # Plot projected points
    ax.scatter(projected_points[:, 0], projected_points[:, 1],
              c='red', s=100, marker='x', label='Reprojected centers', zorder=3)
    
    # Draw error vectors
    for i in range(len(image_points)):
        ax.plot([image_points[i, 0], projected_points[i, 0]],
               [image_points[i, 1], projected_points[i, 1]],
               'g-', alpha=0.5, linewidth=1)
    
    # Add error annotations for outliers
    threshold = CONFIG['max_reprojection_error']
    for i, error in enumerate(reprojection_errors):
        if error > threshold:
            ax.annotate(f'{error:.2f}px',
                       xy=image_points[i],
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Error distribution
    ax = axes[1]
    ax.set_title(f'Reprojection Error Distribution - Combined Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Capture Index', fontsize=12)
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=12)
    
    indices = np.arange(len(reprojection_errors))
    colors = ['red' if e > threshold else 'green' for e in reprojection_errors]
    
    ax.bar(indices, reprojection_errors, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold} px)')
    ax.axhline(y=result['mean_error'], color='blue', linestyle='--', linewidth=2, 
              label=f'Mean ({result["mean_error"]:.2f} px)')
    
    # Rotate x-axis labels
    capture_names = [pair['capture_name'][-6:] for pair in pairs]
    ax.set_xticks(indices)
    ax.set_xticklabels(capture_names, rotation=45, ha='right', fontsize=8)
    
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add statistics
    stats_text = f"Statistics:\n"
    stats_text += f"Mean: {result['mean_error']:.4f} px\n"
    stats_text += f"Max: {result['max_error']:.4f} px\n"
    stats_text += f"Std: {np.std(reprojection_errors):.4f} px\n"
    stats_text += f"Points: {result['num_points']}"
    
    ax.text(0.98, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10,
           family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'reprojection_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def save_calibration_results(result, output_dir):
    """
    Save calibration results to JSON and text files
    """
    # Prepare data for JSON
    output_data = {
        'num_points': int(result['num_points']),
        'initial_extrinsics': {
            'rotation_matrix': result['R_init'].tolist(),
            'translation_vector': result['t_init'].tolist(),
        },
        'optimized_extrinsics': {
            'rotation_matrix': result['R'].tolist(),
            'translation_vector': result['t'].tolist(),
            'rotation_vector': result['rvec'].flatten().tolist(),
            'euler_angles_xyz_degrees': result['euler_xyz'].tolist(),
            'euler_angles_zyx_degrees': result['euler_zyx'].tolist(),
        },
        'reprojection_errors': {
            'mean': float(result['mean_error']),
            'max': float(result['max_error']),
            'min': float(np.min(result['reprojection_errors'])),
            'std': float(np.std(result['reprojection_errors'])),
            'per_point': result['reprojection_errors'].tolist(),
        },
        'camera_matrix': result['camera_matrix'].tolist(),
        'distortion_coefficients': result['dist_coeffs'].tolist(),
        'captures': [pair['capture_name'] for pair in result['pairs']],
    }
    
    # Save JSON
    json_file = output_dir / 'calibration_result.json'
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {json_file}")
    
    # Save human-readable text file
    txt_file = output_dir / 'calibration_result.txt'
    with open(txt_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Camera-Lidar Extrinsic Calibration Results\n")
        f.write(f"Combined Data from Groups 0 and 2\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of calibration points: {result['num_points']}\n\n")
        
        f.write("="*60 + "\n")
        f.write("INITIAL EXTRINSICS (Input Guess)\n")
        f.write("="*60 + "\n\n")
        
        f.write("Initial Rotation Matrix (3x3):\n")
        R_init = result['R_init']
        f.write(f"  [{R_init[0,0]:10.6f}, {R_init[0,1]:10.6f}, {R_init[0,2]:10.6f}]\n")
        f.write(f"  [{R_init[1,0]:10.6f}, {R_init[1,1]:10.6f}, {R_init[1,2]:10.6f}]\n")
        f.write(f"  [{R_init[2,0]:10.6f}, {R_init[2,1]:10.6f}, {R_init[2,2]:10.6f}]\n\n")
        
        f.write("Initial Translation Vector (3x1):\n")
        t_init = result['t_init']
        f.write(f"  [{t_init[0]:10.6f}, {t_init[1]:10.6f}, {t_init[2]:10.6f}]\n\n")
        
        f.write("="*60 + "\n")
        f.write("OPTIMIZED EXTRINSICS (PnP Output)\n")
        f.write("="*60 + "\n\n")
        
        f.write("Optimized Rotation Matrix (3x3):\n")
        R = result['R']
        f.write(f"  [{R[0,0]:10.6f}, {R[0,1]:10.6f}, {R[0,2]:10.6f}]\n")
        f.write(f"  [{R[1,0]:10.6f}, {R[1,1]:10.6f}, {R[1,2]:10.6f}]\n")
        f.write(f"  [{R[2,0]:10.6f}, {R[2,1]:10.6f}, {R[2,2]:10.6f}]\n\n")
        
        f.write("Optimized Translation Vector (3x1):\n")
        t = result['t']
        f.write(f"  [{t[0]:10.6f}, {t[1]:10.6f}, {t[2]:10.6f}]\n\n")
        
        f.write("Euler Angles (XYZ order, degrees):\n")
        euler = result['euler_xyz']
        f.write(f"  Roll:  {euler[0]:8.3f}°\n")
        f.write(f"  Pitch: {euler[1]:8.3f}°\n")
        f.write(f"  Yaw:   {euler[2]:8.3f}°\n\n")
        
        f.write("Reprojection Error Statistics (pixels):\n")
        f.write(f"  Mean:  {result['mean_error']:8.4f}\n")
        f.write(f"  Max:   {result['max_error']:8.4f}\n")
        f.write(f"  Min:   {np.min(result['reprojection_errors']):8.4f}\n")
        f.write(f"  Std:   {np.std(result['reprojection_errors']):8.4f}\n\n")
        
        f.write("Per-point Reprojection Errors:\n")
        for i, (pair, error) in enumerate(zip(result['pairs'], result['reprojection_errors'])):
            f.write(f"  {i+1:2d}. {pair['capture_name']}: {error:8.4f} pixels\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Transformation Usage:\n")
        f.write("="*60 + "\n")
        f.write("To transform a point from lidar to camera coordinates:\n")
        f.write("  P_camera = R * P_lidar + t\n\n")
        f.write("To project a lidar point to image plane:\n")
        f.write("  P_camera = R * P_lidar + t\n")
        f.write("  p_image = K * P_camera / P_camera[2]\n")
        f.write("  where K is the camera intrinsic matrix\n")
    
    print(f"  Saved: {txt_file}")


def main():
    print("="*60)
    print("Camera-Lidar Extrinsic Calibration using PnP")
    print("="*60)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all captures
    print(f"\n=== Processing Calibration Data ===")
    matched_pairs = []
    
    for dir_str in CONFIG['calibration_data_dirs']:
        dir_path = Path(dir_str)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        
        print(f"\nProcessing directory: {dir_path.name}")
        
        # Find all capture directories
        capture_dirs = sorted([d for d in dir_path.iterdir() 
                              if d.is_dir() and d.name.startswith('capture_')])
        
        for capture_dir in capture_dirs:
            result = process_single_capture(capture_dir, dir_path.name)
            if result is not None:
                matched_pairs.append(result)
    
    if len(matched_pairs) == 0:
        print("\nError: No valid capture pairs found!")
        return
    
    print(f"\n=== Summary ===")
    print(f"  Total matched pairs: {len(matched_pairs)}")
    
    # Load camera intrinsics (using group 0, same for all)
    camera_intrinsics_path = Path(CONFIG['calibration_data_dirs'][0]) / 'camera_intrinsics.json'
    camera_matrix, dist_coeffs = load_camera_intrinsics(camera_intrinsics_path)
    
    print(f"\n=== Camera Intrinsics ===")
    print(f"  Camera matrix:")
    print(f"    [{camera_matrix[0,0]:8.2f}, {camera_matrix[0,1]:8.2f}, {camera_matrix[0,2]:8.2f}]")
    print(f"    [{camera_matrix[1,0]:8.2f}, {camera_matrix[1,1]:8.2f}, {camera_matrix[1,2]:8.2f}]")
    print(f"    [{camera_matrix[2,0]:8.2f}, {camera_matrix[2,1]:8.2f}, {camera_matrix[2,2]:8.2f}]")
    
    # Solve PnP
    result = solve_pnp_calibration(matched_pairs, camera_matrix, dist_coeffs)
    
    if result is None:
        print("\nError: PnP solving failed!")
        return
    
    # Save results
    print(f"\n=== Saving Results ===")
    save_calibration_results(result, output_dir)
    
    # Visualize reprojection
    visualize_reprojection_summary(result, output_dir)
    visualize_reprojection_on_images(result, output_dir, CONFIG['calibration_data_dirs'])
    
    # Final summary
    print(f"\n{'='*60}")
    print("Calibration Complete - Summary")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nCalibration points: {result['num_points']}")
    print(f"Mean reprojection error: {result['mean_error']:.4f} pixels")
    print(f"Max reprojection error: {result['max_error']:.4f} pixels")
    
    if result['mean_error'] < CONFIG['max_reprojection_error']:
        print(f"Status: ✓ PASS (< {CONFIG['max_reprojection_error']} pixels)")
    else:
        print(f"Status: ✗ WARNING (mean error > {CONFIG['max_reprojection_error']} pixels)")
    
    print(f"\n{'='*60}")
    print("Files generated:")
    print("  - calibration_result.json: Full calibration parameters (initial + optimized)")
    print("  - calibration_result.txt: Human-readable results with comparison")
    print("  - reprojection_summary.png: Summary visualization of errors")
    print("  - reprojection_images/: Individual images with reprojection overlay")
    print("="*60)


if __name__ == '__main__':
    main()
