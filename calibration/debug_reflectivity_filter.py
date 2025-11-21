#!/usr/bin/env python3
"""
Debug script: Batch process all calibration data with reflectivity filtering
Processes all captures in calibration_data/0 and calibration_data/2
Saves filtered point clouds and circle centers for manual inspection in CloudCompare
"""

import numpy as np
from pathlib import Path
import sys
import open3d as o3d
import json

# Configuration - modify these parameters to test different settings
CONFIG = {
    # Input directories to process
    'input_dirs': ['calibration_data/0', 'calibration_data/2'],
    
    # Output directory for filtered point clouds
    'output_dir': './debug_filtered_output',
    
    # Filtering parameters (same as cali_compute.py)
    'reflectivity_percentile': 95,  # Reflectivity percentile threshold
    'min_reflectivity_ratio': 0.7,  # Minimum ratio relative to maximum reflectivity
    'circle_ransac_threshold': 0.01,  # RANSAC threshold for plane fitting (meters)
    'circle_ransac_iterations': 1000,
    'min_circle_points': 10,
    
    # Outlier removal parameters
    'nb_neighbors': 20,  # Number of neighbors for statistical outlier removal
    'std_ratio': 2.0,    # Standard deviation ratio for outlier detection
}


def load_point_cloud(npy_file):
    """Load point cloud from .npy file"""
    points = np.load(npy_file)
    print(f"\nLoaded point cloud: {npy_file}")
    print(f"  Shape: {points.shape}")
    print(f"  Number of points: {len(points)}")
    
    if points.shape[1] >= 4:
        print(f"  Fields: x, y, z, reflectivity, ...")
    else:
        print(f"  Warning: Expected at least 4 fields, got {points.shape[1]}")
    
    return points


def filter_high_reflectivity_points(points):
    """
    Filter high-reflectivity points using the same method as cali_compute.py
    
    Parameters:
        points: Nx6 array [x, y, z, reflectivity, tag, line]
    
    Returns:
        high_reflectivity_points: Filtered points (xyz only)
        high_reflectivity_mask: Boolean mask of selected points
        stats: Dictionary with filtering statistics
    """
    if points.shape[1] < 4:
        print("Error: Point cloud does not have reflectivity information")
        return None, None, None
    
    xyz = points[:, :3]
    reflectivity = points[:, 3]
    
    # Calculate statistics
    print(f"\n=== Reflectivity Statistics ===")
    print(f"  Min reflectivity: {np.min(reflectivity):.2f}")
    print(f"  Max reflectivity: {np.max(reflectivity):.2f}")
    print(f"  Mean reflectivity: {np.mean(reflectivity):.2f}")
    print(f"  Median reflectivity: {np.median(reflectivity):.2f}")
    print(f"  Std reflectivity: {np.std(reflectivity):.2f}")
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}th: {np.percentile(reflectivity, p):.2f}")
    
    # Apply filtering (same as cali_compute.py)
    reflectivity_threshold = np.percentile(reflectivity, CONFIG['reflectivity_percentile'])
    reflectivity_max = np.max(reflectivity)
    reflectivity_min_value = reflectivity_max * CONFIG['min_reflectivity_ratio']
    reflectivity_threshold = max(reflectivity_threshold, reflectivity_min_value)
    
    print(f"\n=== Filtering Parameters ===")
    print(f"  Percentile threshold ({CONFIG['reflectivity_percentile']}th): {np.percentile(reflectivity, CONFIG['reflectivity_percentile']):.2f}")
    print(f"  Max ratio threshold ({CONFIG['min_reflectivity_ratio']}): {reflectivity_min_value:.2f}")
    print(f"  Final threshold used: {reflectivity_threshold:.2f}")
    
    high_reflectivity_mask = reflectivity >= reflectivity_threshold
    high_reflectivity_points = xyz[high_reflectivity_mask]
    
    print(f"\n=== Filtering Results ===")
    print(f"  Points above threshold: {len(high_reflectivity_points)} / {len(points)} ({100*len(high_reflectivity_points)/len(points):.2f}%)")
    
    stats = {
        'total_points': len(points),
        'filtered_points': len(high_reflectivity_points),
        'percentage': 100 * len(high_reflectivity_points) / len(points),
        'threshold': reflectivity_threshold,
        'reflectivity_min': np.min(reflectivity),
        'reflectivity_max': np.max(reflectivity),
        'reflectivity_mean': np.mean(reflectivity),
    }
    
    return high_reflectivity_points, high_reflectivity_mask, stats


def apply_ransac_plane_fitting(high_reflectivity_points):
    """
    Apply RANSAC plane fitting to extract calibration board points
    Same method as cali_compute.py
    """
    print(f"\n=== RANSAC Plane Fitting ===")
    
    if len(high_reflectivity_points) < CONFIG['min_circle_points']:
        print(f"  Error: Too few points ({len(high_reflectivity_points)}) for RANSAC")
        return None
    
    best_inliers = None
    best_plane = None
    
    for iteration in range(CONFIG['circle_ransac_iterations'] // 10):
        if len(high_reflectivity_points) < 3:
            break
        
        # Randomly select 3 points to fit plane
        idx = np.random.choice(len(high_reflectivity_points), 3, replace=False)
        p1, p2, p3 = high_reflectivity_points[idx]
        
        # Calculate plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-6:
            continue
        
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        
        # Calculate distance of all points to plane
        distances = np.abs(np.dot(high_reflectivity_points, normal) + d)
        inliers = distances < CONFIG['circle_ransac_threshold']
        
        if np.sum(inliers) > (np.sum(best_inliers) if best_inliers is not None else 0):
            best_inliers = inliers
            best_plane = (normal, d)
    
    if best_inliers is None or np.sum(best_inliers) < CONFIG['min_circle_points']:
        print("  Warning: Cannot detect plane")
        return None
    
    circle_points = high_reflectivity_points[best_inliers]
    
    print(f"  RANSAC iterations: {CONFIG['circle_ransac_iterations'] // 10}")
    print(f"  Inlier points: {len(circle_points)} / {len(high_reflectivity_points)}")
    
    return circle_points


def remove_spatial_outliers(points):
    """
    Remove spatial outliers using statistical outlier removal
    
    Args:
        points: Nx3 point cloud array
        
    Returns:
        inlier_points: Filtered points after outlier removal
        inlier_indices: Indices of inlier points
    """
    print(f"\n=== Spatial Outlier Removal ===")
    print(f"  Input points: {len(points)}")
    
    if len(points) < CONFIG['nb_neighbors']:
        print(f"  Warning: Too few points for outlier removal")
        return points, np.arange(len(points))
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=CONFIG['nb_neighbors'],
        std_ratio=CONFIG['std_ratio']
    )
    
    inlier_points = points[ind]
    
    print(f"  Inlier points: {len(inlier_points)} / {len(points)}")
    print(f"  Removed outliers: {len(points) - len(inlier_points)}")
    
    return inlier_points, ind


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
    
    print(f"\n=== Circle Center Computation ===")
    print(f"  Number of points: {len(points)}")
    print(f"  Circle center: [{center_3d[0]:.4f}, {center_3d[1]:.4f}, {center_3d[2]:.4f}]")
    
    return center_3d


def save_ply(points, output_file, with_color=None):
    """
    Save points as PLY file
    
    Parameters:
        points: Nx3 array of xyz coordinates
        output_file: Output file path
        with_color: Optional Nx3 array of RGB colors (0-255)
    """
    num_points = len(points)
    
    with open(output_file, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if with_color is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Write point data
        for i, point in enumerate(points):
            if with_color is not None:
                color = with_color[i].astype(int)
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
            else:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"  Saved: {output_file} ({num_points} points)")


def process_single_capture(input_file, output_subdir):
    """
    Process a single capture file
    
    Args:
        input_file: Path to lidar.npy file
        output_subdir: Output subdirectory for this capture
        
    Returns:
        result_dict: Dictionary with processing results
    """
    capture_name = input_file.parent.name
    print(f"\n{'='*60}")
    print(f"Processing: {capture_name}")
    print(f"{'='*60}")
    
    # Create output directory for this capture
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load point cloud
        points = load_point_cloud(input_file)
        
        # Filter high-reflectivity points
        high_reflectivity_points, mask, stats = filter_high_reflectivity_points(points)
        
        if high_reflectivity_points is None:
            return {
                'capture': capture_name,
                'status': 'failed',
                'error': 'No high-reflectivity points found'
            }
        
        # Remove spatial outliers
        filtered_points_no_outliers, inlier_indices = remove_spatial_outliers(high_reflectivity_points)
        
        # Fit circle center after outlier removal
        if len(filtered_points_no_outliers) >= 3:
            circle_center = fit_circle_center_3d(filtered_points_no_outliers[:, :3])
        else:
            print("\nWarning: Too few points after outlier removal to fit circle center")
            circle_center = None
        
        # Save results
        print(f"\n=== Saving Results ===")
        
        # Save filtered points with circle center to PLY file
        if circle_center is not None:
            # Create point cloud with all filtered points + highlighted center
            all_points = filtered_points_no_outliers[:, :3]
            
            # Create a larger sphere for the center point
            center_point = circle_center.reshape(1, 3)
            
            # Save to PLY with colors: white for filtered points, red for center
            colors_filtered = np.ones((len(all_points), 3)) * 255  # White
            color_center = np.array([[255, 0, 0]])  # Red
            
            # Combine points and colors
            combined_points = np.vstack([all_points, center_point])
            combined_colors = np.vstack([colors_filtered, color_center])
            
            filtered_with_center_file = output_subdir / f"{capture_name}.ply"
            save_ply(combined_points, filtered_with_center_file, with_color=combined_colors)
        else:
            filtered_file = output_subdir / f"{capture_name}.ply"
            save_ply(filtered_points_no_outliers[:, :3], filtered_file)
        
        print(f"\n  ✓ Processing complete for {capture_name}")
        print(f"    Points used for circle center: {len(filtered_points_no_outliers)}")
        if circle_center is not None:
            print(f"    Circle center: [{circle_center[0]:.4f}, {circle_center[1]:.4f}, {circle_center[2]:.4f}]")
        
        return {
            'capture': capture_name,
            'status': 'success',
            'total_points': int(stats['total_points']),
            'high_reflectivity_points': int(stats['filtered_points']),
            'points_used_for_circle_center': int(len(filtered_points_no_outliers)),
            'outliers_removed': int(len(high_reflectivity_points) - len(filtered_points_no_outliers)),
            'circle_center': circle_center.tolist() if circle_center is not None else None,
        }
        
    except Exception as e:
        print(f"\n✗ Error processing {capture_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'capture': capture_name,
            'status': 'error',
            'error': str(e)
        }


def main():
    print("="*60)
    print("Batch Debug: Reflectivity-based Point Cloud Filtering")
    print("Processing all captures in calibration_data/0 and calibration_data/2")
    print("="*60)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all capture directories
    all_captures = []
    for input_dir in CONFIG['input_dirs']:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"\nWarning: Directory does not exist: {input_path}")
            continue
        
        # Find all capture directories
        for capture_dir in sorted(input_path.iterdir()):
            if capture_dir.is_dir() and capture_dir.name.startswith('capture_'):
                lidar_file = capture_dir / 'lidar.npy'
                if lidar_file.exists():
                    all_captures.append((lidar_file, input_path.name, capture_dir.name))
                else:
                    print(f"Warning: lidar.npy not found in {capture_dir}")
    
    if not all_captures:
        print("\nError: No capture files found!")
        print("Please check that calibration_data/0 and calibration_data/2 contain capture directories")
        return
    
    print(f"\nFound {len(all_captures)} captures to process")
    
    # Process all captures
    results = []
    for i, (lidar_file, parent_dir, capture_name) in enumerate(all_captures, 1):
        print(f"\n[{i}/{len(all_captures)}]")
        
        # Save all PLY files in a single directory named by parent_dir
        output_subdir = output_dir / parent_dir
        
        result = process_single_capture(lidar_file, output_subdir)
        results.append(result)
    
    # Save summary report
    print(f"\n{'='*60}")
    print("Processing Complete - Summary Report")
    print(f"{'='*60}")
    
    summary_file = output_dir / "batch_processing_summary.json"
    summary_data = {
        'total_captures': len(all_captures),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] != 'success'),
        'results': results,
        'config': {
            'reflectivity_percentile': CONFIG['reflectivity_percentile'],
            'min_reflectivity_ratio': CONFIG['min_reflectivity_ratio'],
            'nb_neighbors': CONFIG['nb_neighbors'],
            'std_ratio': CONFIG['std_ratio'],
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n  Total captures processed: {len(all_captures)}")
    print(f"  Successful: {summary_data['successful']}")
    print(f"  Failed: {summary_data['failed']}")
    
    print(f"\n  Summary saved to: {summary_file}")
    
    # Print table of results
    print(f"\n{'='*60}")
    print("Detailed Results:")
    print(f"{'='*60}")
    print(f"{'Capture':<40} {'Status':<10} {'Points for Center':<20}")
    print(f"{'-'*60}")
    
    for result in results:
        capture = result['capture']
        status = result['status']
        if status == 'success':
            points = result['points_used_for_circle_center']
            print(f"{capture:<40} {status:<10} {points:<20}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{capture:<40} {status:<10} Error: {error}")
    
    print(f"\n{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")
    print("\nAll PLY files saved in:")
    print(f"  - {output_dir}/0/ : PLY files from calibration_data/0")
    print(f"  - {output_dir}/2/ : PLY files from calibration_data/2")
    print("\nEach PLY file contains:")
    print("  - White points: Filtered high-reflectivity points")
    print("  - Red point: Computed circle center")
    print("\nYou can now open all PLY files in CloudCompare for manual inspection!")
    print("="*60)


if __name__ == '__main__':
    main()
