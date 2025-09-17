#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_antialiased_highlight.py

功能：
- 测试抗锯齿高光碎片处理功能
- 演示如何生成抗锯齿后的可视化图像

使用：
python test_antialiased_highlight.py
"""

import cv2
import numpy as np
import time
from pathlib import Path
from color_antialiased_highlight_visualizer import (
    generate_antialiased_highlight_visualization,
    generate_individual_fragment_visualizations,
    generate_comparison_visualization
)

def create_test_fragments():
    """创建测试用的高光碎片"""
    # 模拟指甲形状的碎片
    fragments = [
        # 细长条状碎片
        np.array([[100, 80], [300, 80], [300, 120], [100, 120]], dtype=np.int32),
        # 弯曲碎片
        np.array([[150, 150], [250, 140], [240, 180], [140, 190]], dtype=np.int32),
        # 小碎片
        np.array([[200, 200], [220, 200], [220, 210], [200, 210]], dtype=np.int32)
    ]
    return fragments

def test_antialiased_processing():
    """测试抗锯齿处理功能"""
    print("开始测试抗锯齿高光碎片处理...")
    
    # 创建测试碎片
    fragments = create_test_fragments()
    img_shape = (400, 600, 3)  # 高度400，宽度600，3通道
    
    print(f"生成了 {len(fragments)} 个测试碎片")
    
    # 生成各种抗锯齿可视化图像
    print("\n1. 生成整体抗锯齿图像...")
    generate_antialiased_highlight_visualization(fragments, img_shape)
    
    print("\n2. 生成单个碎片抗锯齿图像...")
    generate_individual_fragment_visualizations(fragments, img_shape)
    
    print("\n3. 生成对比图像...")
    generate_comparison_visualization(fragments, img_shape)
    
    print("\n测试完成！")
    print("请查看以下目录中的输出图像：")
    print("- data/output/antialiased_highlights/")
    print("- data/output/individual_fragments/")
    print("- data/output/comparison/")

if __name__ == "__main__":
    test_antialiased_processing() 