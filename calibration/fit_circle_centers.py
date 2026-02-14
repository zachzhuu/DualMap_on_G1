#!/usr/bin/env python3
"""
从高反射率点云中拟合圆心
使用RANSAC算法提高鲁棒性
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


def fit_plane_ransac(points, ransac_threshold=0.01, max_trials=1000):
    """
    使用RANSAC拟合平面
    
    Args:
        points: Nx3 点云数组
        ransac_threshold: RANSAC距离阈值
        max_trials: 最大迭代次数
        
    Returns:
        plane_model: [a, b, c, d] 平面方程 ax + by + cz + d = 0
        inliers: 内点索引
    """
    # 使用Open3D的RANSAC平面拟合
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_threshold,
        ransac_n=3,
        num_iterations=max_trials
    )
    
    return plane_model, inliers


def project_to_plane(points, plane_model):
    """
    将点投影到平面上
    
    Args:
        points: Nx3 点云数组
        plane_model: [a, b, c, d] 平面方程
        
    Returns:
        projected_points: 投影后的点
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # 计算每个点到平面的距离
    distances = (points @ normal + d).reshape(-1, 1)
    
    # 投影到平面
    projected_points = points - distances * normal
    
    return projected_points


def points_to_2d(points_3d, plane_model):
    """
    将3D点转换到平面的2D坐标系
    
    Args:
        points_3d: Nx3 点云数组
        plane_model: [a, b, c, d] 平面方程
        
    Returns:
        points_2d: Nx2 2D坐标
        basis: (origin, u, v) 2D坐标系的基
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # 选择一个不平行于法向量的向量
    if abs(normal[2]) < 0.9:
        arbitrary = np.array([0, 0, 1])
    else:
        arbitrary = np.array([1, 0, 0])
    
    # 构建正交基
    u = np.cross(arbitrary, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # 选择原点（点云的质心在平面上的投影）
    centroid = np.mean(points_3d, axis=0)
    dist_to_plane = centroid @ normal + d
    origin = centroid - dist_to_plane * normal
    
    # 转换到2D
    relative = points_3d - origin
    points_2d = np.column_stack([
        relative @ u,
        relative @ v
    ])
    
    return points_2d, (origin, u, v)


def compute_centroid_2d(points_2d):
    """
    计算2D点云的质心作为圆心
    由于高反射率点布满整个圆形区域，质心就是圆心
    
    Args:
        points_2d: Nx2 2D点坐标
        
    Returns:
        center: (cx, cy) 圆心（质心）
        radius: 半径（平均距离）
        inliers: 所有点都是内点
    """
    n_points = len(points_2d)
    if n_points < 3:
        return None, None, []
    
    # 计算质心
    center = np.mean(points_2d, axis=0)
    
    # 计算平均半径
    distances = np.linalg.norm(points_2d - center, axis=1)
    radius = np.mean(distances)
    
    # 所有点都是内点
    inliers = np.arange(n_points)
    
    return center, radius, inliers


def fit_circle_3points(points):
    """
    通过3个点拟合圆
    
    Args:
        points: 3x2 数组
        
    Returns:
        center: (cx, cy)
        radius: 半径
    """
    p1, p2, p3 = points
    
    # 计算中垂线
    ax = (p1[0] + p2[0]) / 2
    ay = (p1[1] + p2[1]) / 2
    ux = p2[1] - p1[1]
    uy = -(p2[0] - p1[0])
    
    bx = (p2[0] + p3[0]) / 2
    by = (p2[1] + p3[1]) / 2
    vx = p3[1] - p2[1]
    vy = -(p3[0] - p2[0])
    
    # 求交点（圆心）
    det = ux * vy - uy * vx
    if abs(det) < 1e-10:
        return None, None
    
    dx = bx - ax
    dy = by - ay
    t = (dx * vy - dy * vx) / det
    
    cx = ax + t * ux
    cy = ay + t * uy
    
    # 计算半径
    radius = np.linalg.norm(p1 - np.array([cx, cy]))
    
    return np.array([cx, cy]), radius


def fit_circle_least_squares(points_2d):
    """
    使用最小二乘法拟合圆
    
    Args:
        points_2d: Nx2 点坐标
        
    Returns:
        center: (cx, cy)
        radius: 半径
    """
    if len(points_2d) < 3:
        return None, None
    
    # 构建线性系统 A * [cx, cy, r^2 - cx^2 - cy^2]^T = b
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    
    try:
        # 求解
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = params
        radius = np.sqrt(c + cx**2 + cy**2)
        
        return np.array([cx, cy]), radius
    except:
        return None, None


def process_single_capture(ply_path, plane_threshold=0.01, circle_threshold=0.01):
    """
    处理单个点云文件，提取圆心
    
    Args:
        ply_path: PLY文件路径
        plane_threshold: 平面RANSAC阈值
        circle_threshold: 圆RANSAC阈值
        
    Returns:
        result: 包含圆心、半径等信息的字典
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    
    if len(points) < 10:
        return None
    
    print(f"\n处理 {ply_path.name}: {len(points)} 个点")
    
    # 步骤1: 拟合平面并去除离群点
    plane_model, inliers = fit_plane_ransac(points, plane_threshold)
    inlier_points = points[inliers]
    
    print(f"  平面拟合: {len(inliers)}/{len(points)} 内点 ({len(inliers)/len(points)*100:.1f}%)")
    
    if len(inlier_points) < 10:
        print("  ❌ 平面内点过少")
        return None
    
    # 步骤2: 投影到平面
    projected_points = project_to_plane(inlier_points, plane_model)
    
    # 步骤3: 转换到2D坐标系
    points_2d, basis = points_to_2d(projected_points, plane_model)
    origin, u, v = basis
    
    # 步骤4: 计算质心作为圆心（高反射率点布满整个圆形区域）
    center_2d, radius, circle_inliers = compute_centroid_2d(points_2d)
    
    if center_2d is None:
        print("  ❌ 质心计算失败")
        return None
    
    print(f"  圆心计算: 使用{len(circle_inliers)}个点的质心")
    print(f"  平均半径: {radius*1000:.2f} mm")
    
    # 步骤5: 转换回3D坐标
    center_3d = origin + center_2d[0] * u + center_2d[1] * v
    
    # 计算点到圆心的距离分布（作为参考）
    distances_2d = np.linalg.norm(points_2d - center_2d, axis=1)
    mean_distance = np.mean(distances_2d)
    std_distance = np.std(distances_2d)
    
    print(f"  点到圆心距离: {mean_distance*1000:.2f} ± {std_distance*1000:.2f} mm")
    print(f"  圆心3D: [{center_3d[0]:.4f}, {center_3d[1]:.4f}, {center_3d[2]:.4f}]")
    
    return {
        'capture_name': ply_path.stem,
        'center_3d': center_3d.tolist(),
        'center_2d': center_2d.tolist(),
        'radius': float(radius),
        'plane_model': plane_model.tolist() if isinstance(plane_model, np.ndarray) else plane_model,
        'n_total_points': len(points),
        'n_plane_inliers': len(inliers),
        'n_circle_inliers': len(circle_inliers),
        'mean_distance': float(mean_distance),
        'std_distance': float(std_distance),
        'basis': {
            'origin': origin.tolist(),
            'u': u.tolist(),
            'v': v.tolist()
        }
    }


def visualize_results(results, output_dir):
    """
    可视化所有圆心的3D位置
    """
    if not results:
        print("没有结果可视化")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取所有圆心
    centers = np.array([r['center_3d'] for r in results])
    
    # 绘制圆心
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
               c='red', s=100, marker='o', label='Circle Centers')
    
    # 标注每个点
    for i, result in enumerate(results):
        center = result['center_3d']
        ax.text(center[0], center[1], center[2], 
                f"  {result['capture_name']}", fontsize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Fitted Circle Centers in 3D Space')
    ax.legend()
    
    # 设置相等的坐标轴比例
    max_range = np.array([
        centers[:, 0].max() - centers[:, 0].min(),
        centers[:, 1].max() - centers[:, 1].min(),
        centers[:, 2].max() - centers[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (centers[:, 0].max() + centers[:, 0].min()) * 0.5
    mid_y = (centers[:, 1].max() + centers[:, 1].min()) * 0.5
    mid_z = (centers[:, 2].max() + centers[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'circle_centers_3d.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 3D可视化已保存到 {output_dir / 'circle_centers_3d.png'}")
    plt.close()


def main():
    # 设置路径
    input_dir = Path('debug_filtered_output')
    output_dir = Path('debug_filtered_output')
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有PLY文件
    ply_files = sorted(input_dir.glob('*.ply'))
    
    if not ply_files:
        print("错误: 未找到PLY文件")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    print("="*60)
    
    # 处理每个文件
    results = []
    failed = []
    
    for ply_path in ply_files:
        try:
            result = process_single_capture(ply_path)
            if result:
                results.append(result)
            else:
                failed.append(ply_path.name)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            failed.append(ply_path.name)
    
    print("\n" + "="*60)
    print(f"处理完成: {len(results)} 成功, {len(failed)} 失败")
    
    if failed:
        print(f"\n失败的文件:")
        for name in failed:
            print(f"  - {name}")
    
    # 保存结果
    if results:
        output_json = output_dir / 'circle_centers.json'
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果已保存到 {output_json}")
        
        # 可视化
        visualize_results(results, output_dir)
        
        # 打印统计信息
        print("\n统计信息:")
        distances = [r['mean_distance'] for r in results]
        print(f"  平均点到质心距离: {np.mean(distances)*1000:.2f} ± {np.std(distances)*1000:.2f} mm")


if __name__ == '__main__':
    main()
