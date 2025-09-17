#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_antialiased_highlight_visualizer.py

功能：
- 对生成的高光碎片进行抗锯齿处理
- 输出抗锯齿后的可视化图像（黑色背景，白色高光碎片）
- 支持批量处理和单个碎片处理

使用：
python color_antialiased_highlight_visualizer.py
"""

import cv2
import numpy as np
import time
from pathlib import Path
import os

def generate_antialiased_highlight_visualization(fragments, img_shape, output_dir="data/output/antialiased_highlights"):
    """
    生成极限抗锯齿处理后的高光碎片灰度掩码图像（无黑色背景，直接输出灰度，便于后续羽化/合成）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    h, w = img_shape[:2]
    # 先生成大分辨率灰度掩码
    scale_factor = 32  # 极限超采样
    blur_kernel_size = 41  # 极大模糊核
    big_h, big_w = h * scale_factor, w * scale_factor
    big_mask = np.zeros((big_h, big_w), dtype=np.uint8)
    for frag_pts in fragments:
        big_frag_pts = (frag_pts * scale_factor).astype(np.int32)
        cv2.fillPoly(big_mask, [big_frag_pts], 255)
    big_mask_blur = cv2.GaussianBlur(big_mask, (blur_kernel_size, blur_kernel_size), 0)
    # 缩回原始分辨率，保留灰度
    small_mask_blur = cv2.resize(big_mask_blur, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # 再加一次小核高斯模糊，进一步羽化边缘
    final_mask = cv2.GaussianBlur(small_mask_blur, (5, 5), 0)
    # 直接保存灰度图（0~255，边缘平滑）
    timestamp = int(time.time())
    output_filename = f"antialiased_highlight_gray_{timestamp}.png"
    output_filepath = output_path / output_filename
    cv2.imwrite(str(output_filepath), final_mask)
    print(f"[极限抗锯齿] 已保存极限抗锯齿灰度掩码: {output_filepath}")
    return output_filepath

def generate_individual_fragment_visualizations(fragments, img_shape, output_dir="data/output/individual_fragments"):
    """
    为每个高光碎片单独生成抗锯齿可视化图像
    
    Args:
        fragments: 高光碎片轮廓列表
        img_shape: 图像尺寸 (height, width, channels)
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    h, w = img_shape[:2]
    timestamp = int(time.time())
    
    for i, frag_pts in enumerate(fragments):
        # 创建黑色背景
        black_bg = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 抗锯齿处理
        scale_factor = 8
        big_h, big_w = h * scale_factor, w * scale_factor
        big_mask = np.zeros((big_h, big_w), dtype=np.uint8)
        big_frag_pts = (frag_pts * scale_factor).astype(np.int32)
        cv2.fillPoly(big_mask, [big_frag_pts], 255)
        
        # 高斯模糊抗锯齿
        blur_kernel_size = 21
        big_mask_blur = cv2.GaussianBlur(big_mask, (blur_kernel_size, blur_kernel_size), 0)
        small_mask_blur = cv2.resize(big_mask_blur, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 白色填充
        white_color = np.array([255, 255, 255], dtype=np.uint8)
        for c in range(3):
            black_bg[:, :, c] = np.where(small_mask_blur > 0, white_color[c], black_bg[:, :, c])
        
        # 保存单个碎片图像
        output_filename = f"fragment_{i+1}_antialiased_{timestamp}.png"
        output_filepath = output_path / output_filename
        cv2.imwrite(str(output_filepath), black_bg)
        print(f"[抗锯齿] 已保存碎片{i+1}抗锯齿图像: {output_filepath}")

def generate_comparison_visualization(fragments, img_shape, output_dir="data/output/comparison"):
    """
    生成对比可视化图像：原始碎片 vs 抗锯齿碎片
    
    Args:
        fragments: 高光碎片轮廓列表
        img_shape: 图像尺寸 (height, width, channels)
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    h, w = img_shape[:2]
    timestamp = int(time.time())
    
    # 创建对比图像：左半边原始，右半边抗锯齿
    comparison_img = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # 左半边：原始碎片
    left_half = comparison_img[:, :w]
    for frag_pts in fragments:
        cv2.fillPoly(left_half, [frag_pts], (255, 255, 255))
    
    # 右半边：抗锯齿碎片
    right_half = comparison_img[:, w:]
    for frag_pts in fragments:
        # 抗锯齿处理
        scale_factor = 8
        big_h, big_w = h * scale_factor, w * scale_factor
        big_mask = np.zeros((big_h, big_w), dtype=np.uint8)
        big_frag_pts = (frag_pts * scale_factor).astype(np.int32)
        cv2.fillPoly(big_mask, [big_frag_pts], 255)
        
        # 高斯模糊抗锯齿
        blur_kernel_size = 21
        big_mask_blur = cv2.GaussianBlur(big_mask, (blur_kernel_size, blur_kernel_size), 0)
        small_mask_blur = cv2.resize(big_mask_blur, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 白色填充
        white_color = np.array([255, 255, 255], dtype=np.uint8)
        for c in range(3):
            right_half[:, :, c] = np.where(small_mask_blur > 0, white_color[c], right_half[:, :, c])
    
    # 添加分隔线
    cv2.line(comparison_img, (w, 0), (w, h), (128, 128, 128), 2)
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_img, "Antialiased", (w + 10, 30), font, 1, (255, 255, 255), 2)
    
    # 保存对比图像
    output_filename = f"comparison_original_vs_antialiased_{timestamp}.png"
    output_filepath = output_path / output_filename
    cv2.imwrite(str(output_filepath), comparison_img)
    print(f"[抗锯齿] 已保存对比图像: {output_filepath}")

# 测试函数
def test_antialiased_visualization():
    """测试抗锯齿可视化功能"""
    # 创建一个简单的测试碎片
    test_fragments = [
        np.array([[100, 100], [200, 100], [200, 150], [100, 150]], dtype=np.int32)
    ]
    
    img_shape = (300, 400, 3)
    
    print("开始测试抗锯齿可视化...")
    
    # 生成各种可视化图像
    generate_antialiased_highlight_visualization(test_fragments, img_shape)
    generate_individual_fragment_visualizations(test_fragments, img_shape)
    generate_comparison_visualization(test_fragments, img_shape)
    
    print("测试完成！")

if __name__ == "__main__":
    test_antialiased_visualization() 