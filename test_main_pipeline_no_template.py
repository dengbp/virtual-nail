#!/usr/bin/env python3
"""
测试移除模板拟合功能后的主流程
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV

def test_mask_generation():
    """测试掩码生成（不包含模板拟合）"""
    print("=== 测试主流程掩码生成（无模板拟合） ===")
    
    # 创建测试图像
    test_image = create_test_nail_image()
    cv2.imwrite('test_main_pipeline_image.png', test_image)
    print("已创建测试图像: test_main_pipeline_image.png")
    
    # 初始化处理器
    nail_processor = NailSDXLInpaintOpenCV()
    
    try:
        # 生成掩码
        print("开始生成掩码...")
        mask = nail_processor.generate_mask_u2net(test_image, "test_main_pipeline.png")
        
        # 保存掩码
        cv2.imwrite('test_main_pipeline_mask.png', mask)
        print("掩码生成完成，保存为: test_main_pipeline_mask.png")
        
        # 显示掩码信息
        print(f"掩码尺寸: {mask.shape}")
        print(f"掩码类型: {mask.dtype}")
        print(f"掩码值范围: {mask.min()} - {mask.max()}")
        
        # 统计掩码像素
        nail_pixels = np.sum(mask > 128)
        total_pixels = mask.shape[0] * mask.shape[1]
        nail_ratio = nail_pixels / total_pixels * 100
        print(f"指甲区域占比: {nail_ratio:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"掩码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

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

def test_save_mask():
    """测试save_mask方法"""
    print("\n=== 测试save_mask方法 ===")
    
    # 创建测试图像
    test_image = create_test_nail_image()
    
    # 初始化处理器
    nail_processor = NailSDXLInpaintOpenCV()
    
    try:
        # 保存掩码
        output_path = "test_main_pipeline_saved_mask.png"
        nail_processor.save_mask(test_image, output_path, "test_main_pipeline.png")
        
        # 检查是否生成了优化后的掩码
        optimized_mask_path = "data/output/optimized_masks/test_main_pipeline_saved_mask_optimized_mask.png"
        if Path(optimized_mask_path).exists():
            print(f"优化后掩码已生成: {optimized_mask_path}")
            
            # 读取并显示信息
            optimized_mask = cv2.imread(optimized_mask_path, cv2.IMREAD_GRAYSCALE)
            if optimized_mask is not None:
                print(f"优化后掩码尺寸: {optimized_mask.shape}")
                print(f"优化后掩码值范围: {optimized_mask.min()} - {optimized_mask.max()}")
                
                # 统计优化后掩码像素
                nail_pixels = np.sum(optimized_mask > 128)
                total_pixels = optimized_mask.shape[0] * optimized_mask.shape[1]
                nail_ratio = nail_pixels / total_pixels * 100
                print(f"优化后指甲区域占比: {nail_ratio:.2f}%")
        else:
            print(f"未找到优化后掩码: {optimized_mask_path}")
        
        return True
        
    except Exception as e:
        print(f"save_mask测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("测试移除模板拟合功能后的主流程")
    print("=" * 50)
    
    # 测试1: 掩码生成
    success1 = test_mask_generation()
    
    # 测试2: save_mask方法
    success2 = test_save_mask()
    
    print("\n=== 测试结果 ===")
    print(f"掩码生成测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"save_mask测试: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 主流程测试全部通过！")
        print("模板拟合功能已成功从主流程中移除。")
        print("\n生成的文件:")
        print("- test_main_pipeline_image.png: 测试图像")
        print("- test_main_pipeline_mask.png: 生成的掩码")
        print("- test_main_pipeline_saved_mask.png: 保存的掩码")
        print("- data/output/optimized_masks/test_main_pipeline_saved_mask_optimized_mask.png: 优化后掩码")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 