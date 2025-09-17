#!/usr/bin/env python3
"""
测试掩码生成功能
验证每次请求是否都生成新的掩码
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV

def create_test_image(width=800, height=600, color=(255, 0, 0)):
    """创建测试图像"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img

def test_mask_generation():
    """测试掩码生成"""
    print("开始测试掩码生成功能...")
    
    # 初始化处理器
    nail = NailSDXLInpaintOpenCV()
    
    # 创建测试目录
    test_dir = "data/test_masks_debug"
    os.makedirs(test_dir, exist_ok=True)
    
    # 测试多次生成掩码
    for i in range(3):
        print(f"\n--- 第 {i+1} 次测试 ---")
        
        # 创建不同的测试图像
        if i == 0:
            # 红色图像
            test_img = create_test_image(800, 600, (255, 0, 0))
            print("创建红色测试图像")
        elif i == 1:
            # 绿色图像
            test_img = create_test_image(800, 600, (0, 255, 0))
            print("创建绿色测试图像")
        else:
            # 蓝色图像
            test_img = create_test_image(800, 600, (0, 0, 255))
            print("创建蓝色测试图像")
        
        # 生成掩码
        mask_path = os.path.join(test_dir, f"test_mask_{i+1}.png")
        print(f"生成掩码: {mask_path}")
        
        start_time = time.time()
        nail.save_mask(test_img, mask_path)
        end_time = time.time()
        
        print(f"掩码生成耗时: {end_time - start_time:.2f} 秒")
        
        # 检查掩码文件是否存在
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"掩码生成成功，尺寸: {mask.shape}")
                print(f"掩码像素统计: min={mask.min()}, max={mask.max()}, mean={mask.mean():.2f}")
                
                # 计算掩码的哈希值用于比较
                mask_hash = hash(mask.tobytes())
                print(f"掩码哈希值: {mask_hash}")
            else:
                print("❌ 掩码文件读取失败")
        else:
            print("❌ 掩码文件不存在")
    
    print("\n--- 测试完成 ---")
    print("请检查生成的掩码文件，确认每次生成的掩码内容是否不同")

if __name__ == "__main__":
    test_mask_generation() 