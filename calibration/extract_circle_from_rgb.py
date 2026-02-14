#!/usr/bin/env python3
"""
从RGB图像中提取圆形标定板并可视化
使用椭圆拟合来检测图像中的白色圆形标定板
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json


def detect_white_circle(image, min_area=1000, max_area=100000):
    """
    在RGB图像中检测白色圆形标定板
    
    Args:
        image: 输入的RGB图像
        min_area: 最小轮廓面积
        max_area: 最大轮廓面积
    
    Returns:
        ellipse: 拟合的椭圆参数 ((cx, cy), (major, minor), angle) 或 None
        contour: 检测到的轮廓
        binary: 二值化图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值 + Otsu阈值
    # 先尝试Otsu阈值
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学操作，去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤面积不合适的轮廓
        if area < min_area or area > max_area:
            continue
        
        # 至少需要5个点才能拟合椭圆
        if len(contour) < 5:
            continue
        
        # 拟合椭圆
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            
            # 检查椭圆的合理性
            # 1. 长短轴比例不能太悬殊（标定板是圆形，投影后应该接近椭圆）
            aspect_ratio = max(major, minor) / (min(major, minor) + 1e-6)
            if aspect_ratio > 3.0:  # 长短轴比例不超过3
                continue
            
            # 2. 椭圆面积与轮廓面积接近（说明是圆形）
            ellipse_area = np.pi * major * minor / 4
            area_ratio = min(area, ellipse_area) / (max(area, ellipse_area) + 1e-6)
            if area_ratio < 0.7:  # 面积匹配度至少70%
                continue
            
            # 3. 轮廓的圆度（周长平方/面积）
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            # 综合评分：面积匹配度 + 圆度
            score = area_ratio * 0.5 + circularity * 0.5
            
            if score > best_score:
                best_score = score
                best_ellipse = ellipse
                best_contour = contour
                
        except Exception as e:
            continue
    
    return best_ellipse, best_contour, binary


def visualize_detection(image, ellipse, contour, binary, save_path=None):
    """
    可视化检测结果
    
    Args:
        image: 原始RGB图像
        ellipse: 拟合的椭圆
        contour: 检测到的轮廓
        binary: 二值化图像
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示二值化图像
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title('Binary Image')
    axes[1].axis('off')
    
    # 显示检测结果
    result_img = image.copy()
    if contour is not None:
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
    
    if ellipse is not None:
        cv2.ellipse(result_img, ellipse, (255, 0, 0), 2)
        # 绘制椭圆中心
        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
        cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)
        
        axes[2].imshow(result_img)
        axes[2].set_title(f'Detected Circle\nCenter: ({cx}, {cy})')
        
        # 添加椭圆参数信息
        major, minor = ellipse[1]
        angle = ellipse[2]
        info_text = f'Major: {major:.1f}, Minor: {minor:.1f}\nAngle: {angle:.1f}°'
        axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    else:
        axes[2].imshow(result_img)
        axes[2].set_title('No Circle Detected')
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def process_all_captures(data_dirs=['./calibration_data/0', './calibration_data/2'], 
                        output_dir='./debug_filtered_output'):
    """
    处理所有采集的数据
    
    Args:
        data_dirs: 标定数据目录列表
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 获取所有capture文件夹
    all_captures = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: Directory does not exist: {data_path}")
            continue
        
        parent_dir = data_path.name
        capture_folders = sorted([d for d in data_path.iterdir() 
                                 if d.is_dir() and d.name.startswith('capture_')])
        
        for folder in capture_folders:
            all_captures.append((folder, parent_dir))
    
    print(f"Found {len(all_captures)} capture folders")
    
    results = []
    
    for i, (capture_folder, parent_dir) in enumerate(all_captures, 1):
        rgb_path = capture_folder / 'rgb.png'
        
        if not rgb_path.exists():
            print(f"Warning: {rgb_path} not found, skipping...")
            continue
        
        print(f"\n[{i}/{len(all_captures)}] Processing {parent_dir}/{capture_folder.name}...")
        
        # 读取RGB图像
        image = cv2.imread(str(rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测圆形标定板
        ellipse, contour, binary = detect_white_circle(image)
        
        # 保存可视化结果到对应的子目录
        parent_output_dir = output_path / parent_dir
        parent_output_dir.mkdir(exist_ok=True, parents=True)
        vis_save_path = parent_output_dir / f"{capture_folder.name}_detection.png"
        
        # Don't show plot, just save
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示二值化图像
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binary Image')
        axes[1].axis('off')
        
        # 显示检测结果
        result_img = image.copy()
        if contour is not None:
            cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        
        if ellipse is not None:
            cv2.ellipse(result_img, ellipse, (255, 0, 0), 2)
            # 绘制椭圆中心
            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)
            
            axes[2].imshow(result_img)
            axes[2].set_title(f'Detected Circle\nCenter: ({cx}, {cy})')
            
            # 添加椭圆参数信息
            major, minor = ellipse[1]
            angle = ellipse[2]
            info_text = f'Major: {major:.1f}, Minor: {minor:.1f}\nAngle: {angle:.1f}°'
            axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
        else:
            axes[2].imshow(result_img)
            axes[2].set_title('No Circle Detected')
        
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录结果
        result = {
            'parent_dir': parent_dir,
            'capture_name': capture_folder.name,
            'detected': ellipse is not None
        }
        
        if ellipse is not None:
            (cx, cy), (major, minor), angle = ellipse
            result['ellipse'] = {
                'center_x': float(cx),
                'center_y': float(cy),
                'major_axis': float(major),
                'minor_axis': float(minor),
                'angle': float(angle)
            }
            print(f"  ✓ Detected ellipse at ({cx:.1f}, {cy:.1f})")
        else:
            print(f"  ✗ No circle detected")
        
        results.append(result)
    
    # 保存结果到JSON
    results_json_path = output_path / 'rgb_detection_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total captures: {len(results)}")
    print(f"Successfully detected: {sum(1 for r in results if r['detected'])}")
    print(f"Failed detections: {sum(1 for r in results if not r['detected'])}")
    print(f"Results saved to {results_json_path}")
    print(f"Visualizations saved to:")
    print(f"  - {output_path}/0/ : Detection results from calibration_data/0")
    print(f"  - {output_path}/2/ : Detection results from calibration_data/2")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Detailed Results:")
    print(f"{'='*60}")
    print(f"{'Dir':<5} {'Capture':<40} {'Status':<10} {'Center (x, y)':<20}")
    print(f"{'-'*80}")
    
    for result in results:
        parent = result['parent_dir']
        capture = result['capture_name']
        status = '✓ OK' if result['detected'] else '✗ FAIL'
        if result['detected']:
            cx = result['ellipse']['center_x']
            cy = result['ellipse']['center_y']
            center_str = f"({cx:.1f}, {cy:.1f})"
        else:
            center_str = "N/A"
        print(f"{parent:<5} {capture:<40} {status:<10} {center_str:<20}")
    
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    # 处理所有标定数据
    results = process_all_captures()
