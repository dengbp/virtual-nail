#!/usr/bin/env python3
"""
测试模板拟合功能
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nail_template_fitter import NailTemplateFitter, apply_template_fitting_simple

def test_template_fitting():
    """测试模板拟合功能"""
    print("=== 测试模板拟合功能 ===")
    
    # 创建测试图像
    test_image = create_test_nail_image()
    cv2.imwrite('test_nail_image.png', test_image)
    print("已创建测试图像: test_nail_image.png")
    
    # 创建测试掩码
    test_mask = create_test_nail_mask()
    cv2.imwrite('test_nail_mask.png', test_mask)
    print("已创建测试掩码: test_nail_mask.png")
    
    # 测试模板拟合
    print("\n开始测试模板拟合...")
    
    try:
        # 使用简化的模板拟合函数
        fitted_mask = apply_template_fitting_simple(
            test_mask,
            template_name='auto',
            smooth_factor=0.8
        )
        
        cv2.imwrite('test_fitted_mask.png', fitted_mask)
        print("模板拟合完成，结果保存为: test_fitted_mask.png")
        
        # 显示对比
        show_comparison(test_mask, fitted_mask)
        
    except Exception as e:
        print(f"模板拟合测试失败: {e}")
        import traceback
        traceback.print_exc()

def create_test_nail_image():
    """创建测试指甲图像"""
    # 创建白色背景
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 绘制手部轮廓
    hand_contour = np.array([
        [100, 50], [150, 30], [200, 40], [250, 60], [280, 100],
        [290, 150], [280, 200], [250, 250], [200, 280], [150, 290],
        [100, 280], [50, 250], [30, 200], [20, 150], [30, 100],
        [50, 60], [100, 50]
    ], dtype=np.int32)
    
    cv2.fillPoly(image, [hand_contour], (240, 220, 200))
    
    # 绘制指甲
    nail_contours = [
        # 拇指
        np.array([[80, 80], [120, 70], [140, 90], [120, 110], [80, 100]], dtype=np.int32),
        # 食指
        np.array([[180, 60], [220, 50], [240, 70], [220, 90], [180, 80]], dtype=np.int32),
        # 中指
        np.array([[250, 70], [290, 60], [310, 80], [290, 100], [250, 90]], dtype=np.int32),
        # 无名指
        np.array([[300, 80], [340, 70], [360, 90], [340, 110], [300, 100]], dtype=np.int32),
        # 小指
        np.array([[350, 90], [390, 80], [410, 100], [390, 120], [350, 110]], dtype=np.int32)
    ]
    
    for nail_contour in nail_contours:
        cv2.fillPoly(image, [nail_contour], (255, 240, 220))
    
    return image

def create_test_nail_mask():
    """创建测试指甲掩码"""
    mask = np.zeros((400, 600), dtype=np.uint8)
    
    # 绘制指甲掩码（稍微不规则）
    nail_contours = [
        # 拇指（椭圆形）
        np.array([[80, 80], [125, 70], [145, 95], [120, 115], [75, 100]], dtype=np.int32),
        # 食指（杏仁形）
        np.array([[180, 60], [225, 50], [245, 75], [220, 95], [175, 80]], dtype=np.int32),
        # 中指（方形）
        np.array([[250, 70], [295, 60], [315, 85], [290, 105], [245, 90]], dtype=np.int32),
        # 无名指（圆形）
        np.array([[300, 80], [345, 70], [365, 95], [340, 115], [295, 100]], dtype=np.int32),
        # 小指（尖形）
        np.array([[350, 90], [395, 80], [415, 105], [390, 125], [345, 110]], dtype=np.int32)
    ]
    
    for nail_contour in nail_contours:
        cv2.fillPoly(mask, [nail_contour], 255)
    
    return mask

def show_comparison(original_mask, fitted_mask):
    """显示对比结果"""
    # 创建对比图像
    comparison = np.zeros((400, 1200, 3), dtype=np.uint8)
    
    # 原始掩码（红色）
    comparison[:, :600, 2] = original_mask
    
    # 拟合后掩码（绿色）
    comparison[:, 600:, 1] = fitted_mask
    
    # 添加文字
    cv2.putText(comparison, "Original Mask", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Template Fitted Mask", (650, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite('comparison_mask.png', comparison)
    print("对比图像已保存: comparison_mask.png")

def test_template_selection():
    """测试模板自动选择功能"""
    print("\n=== 测试模板自动选择 ===")
    
    fitter = NailTemplateFitter()
    
    # 创建不同形状的测试轮廓
    test_contours = {
        'ellipse': np.array([[0, 10], [10, 0], [0, -10], [-10, 0]], dtype=np.int32),
        'almond': np.array([[0, 15], [10, 0], [0, -5], [-10, 0]], dtype=np.int32),
        'square': np.array([[0, 10], [10, 10], [10, -10], [0, -10]], dtype=np.int32),
        'round': np.array([[0, 8], [8, 0], [0, -8], [-8, 0]], dtype=np.int32),
        'pointed': np.array([[0, 20], [8, 0], [0, -2], [-8, 0]], dtype=np.int32)
    }
    
    for shape_name, contour in test_contours.items():
        selected_template = fitter._select_best_template(contour)
        print(f"{shape_name} 轮廓 -> 选择模板: {selected_template}")

def test_real_image():
    """测试真实图像"""
    print("\n=== 测试真实图像 ===")
    
    # 查找测试图像
    test_images = [
        'data/test_images/test1.jpg',
        'data/test_images/test2.jpg',
        'data/test_images/test3.jpg',
        'test_images/test1.jpg',
        'test_images/test2.jpg',
        'test_images/test3.jpg'
    ]
    
    test_image_path = None
    for path in test_images:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("未找到测试图像，跳过真实图像测试")
        return
    
    print(f"使用测试图像: {test_image_path}")
    
    # 读取图像
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return
    
    # 创建简单的掩码（这里只是示例，实际应该使用U2Net）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 应用模板拟合
    try:
        fitted_mask = apply_template_fitting_simple(mask, 'auto', 0.8)
        
        # 保存结果
        cv2.imwrite('real_image_original_mask.png', mask)
        cv2.imwrite('real_image_fitted_mask.png', fitted_mask)
        print("真实图像测试完成")
        
    except Exception as e:
        print(f"真实图像测试失败: {e}")

if __name__ == "__main__":
    # 运行所有测试
    test_template_fitting()
    test_template_selection()
    test_real_image()
    
    print("\n=== 测试完成 ===")
    print("请查看生成的图像文件:")
    print("- test_nail_image.png: 测试指甲图像")
    print("- test_nail_mask.png: 原始掩码")
    print("- test_fitted_mask.png: 模板拟合后掩码")
    print("- comparison_mask.png: 对比图像")
    print("- debug_template_fitting_*.png: 调试图像") 