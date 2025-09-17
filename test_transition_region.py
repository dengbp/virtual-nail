#!/usr/bin/env python3
"""
测试皮肤-指甲过渡区域修复效果
"""

import cv2
import numpy as np
import os
from nail_active_contour_enhancer import enhance_with_active_contour_simple

def test_transition_region():
    """测试过渡区域修复效果"""
    
    # 创建测试图像和掩码
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    test_image[100:300, 200:400] = [255, 200, 150]  # 模拟手部区域
    
    # 创建模拟的指甲掩码（精确边缘）
    test_mask = np.zeros((400, 600), dtype=np.uint8)
    test_mask[120:140, 220:280] = 255  # 水平指甲，精确边缘
    
    print("测试皮肤-指甲过渡区域修复...")
    
    # 测试1：原始掩码（无过渡区域）
    print("1. 原始掩码（无过渡区域）")
    cv2.imwrite('test_output/original_mask_no_transition.png', test_mask)
    
    # 测试2：Active Contour增强（保留过渡区域）
    print("2. Active Contour增强（保留过渡区域）")
    enhanced_mask = enhance_with_active_contour_simple(
        test_mask, 
        test_image, 
        iterations=20,
        edge_expansion=4,   # 边缘扩展4像素
        feather_width=6     # 羽化宽度6像素
    )
    cv2.imwrite('test_output/enhanced_mask_with_transition.png', enhanced_mask)
    
    # 测试3：不同参数对比
    print("3. 不同参数对比")
    
    # 小扩展
    enhanced_small = enhance_with_active_contour_simple(
        test_mask, test_image, iterations=20, edge_expansion=2, feather_width=3
    )
    cv2.imwrite('test_output/enhanced_small_transition.png', enhanced_small)
    
    # 大扩展
    enhanced_large = enhance_with_active_contour_simple(
        test_mask, test_image, iterations=20, edge_expansion=6, feather_width=8
    )
    cv2.imwrite('test_output/enhanced_large_transition.png', enhanced_large)
    
    # 保存测试图像
    cv2.imwrite('test_output/test_image.png', test_image)
    
    print("测试完成！结果保存在 test_output 目录")
    print("请查看以下文件对比效果：")
    print("- original_mask_no_transition.png: 原始掩码（无过渡区域）")
    print("- enhanced_mask_with_transition.png: 增强掩码（有过渡区域）")
    print("- enhanced_small_transition.png: 小过渡区域")
    print("- enhanced_large_transition.png: 大过渡区域")

if __name__ == "__main__":
    os.makedirs('test_output', exist_ok=True)
    test_transition_region() 