#!/usr/bin/env python3
"""
测试边缘引导和ControlNet结构控制增强功能
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

def test_edge_guided_enhancement():
    """测试边缘引导增强"""
    print("=== 测试边缘引导掩码增强 ===")
    
    # 创建测试图像和掩码
    image = create_test_nail_image()
    cv2.imwrite('test_edge_controlnet_image.png', image)
    
    # 创建掩码
    mask = np.zeros((400, 600), dtype=np.uint8)
    nail_contours = [
        np.array([[80, 80], [120, 70], [140, 90], [120, 110], [80, 100]], dtype=np.int32),
        np.array([[180, 60], [220, 50], [240, 70], [220, 90], [180, 80]], dtype=np.int32),
        np.array([[250, 70], [290, 60], [310, 80], [290, 100], [250, 90]], dtype=np.int32),
    ]
    
    for nail_contour in nail_contours:
        cv2.fillPoly(mask, [nail_contour], 255)
    
    cv2.imwrite('test_edge_controlnet_mask.png', mask)
    
    # 测试边缘引导增强
    try:
        from nail_edge_guided_enhancer import enhance_mask_with_edge_guidance
        
        methods = ['canny', 'sobel', 'combined']
        
        for method in methods:
            print(f"\n测试边缘引导方法: {method}")
            start_time = time.time()
            
            enhanced_mask = enhance_mask_with_edge_guidance(
                mask, image, 
                method=method,
                edge_weight=0.7,
                smooth_factor=0.8,
                save_debug=True
            )
            
            end_time = time.time()
            print(f"处理时间: {end_time - start_time:.2f}秒")
            
            # 保存结果
            cv2.imwrite(f'test_edge_guided_result_{method}.png', enhanced_mask)
            print(f"结果已保存: test_edge_guided_result_{method}.png")
            
        return True
        
    except ImportError as e:
        print(f"边缘引导模块不可用: {e}")
        return False

def test_controlnet_structure():
    """测试ControlNet结构图生成"""
    print("\n=== 测试ControlNet结构图生成 ===")
    
    # 创建测试图像和掩码
    image = create_test_nail_image()
    mask = np.zeros((400, 600), dtype=np.uint8)
    nail_contours = [
        np.array([[80, 80], [120, 70], [140, 90], [120, 110], [80, 100]], dtype=np.int32),
        np.array([[180, 60], [220, 50], [240, 70], [220, 90], [180, 80]], dtype=np.int32),
        np.array([[250, 70], [290, 60], [310, 80], [290, 100], [250, 90]], dtype=np.int32),
    ]
    
    for nail_contour in nail_contours:
        cv2.fillPoly(mask, [nail_contour], 255)
    
    # 测试ControlNet结构图生成
    try:
        from nail_controlnet_structure_enhancer import generate_controlnet_structure_maps, create_controlnet_inputs
        
        print("生成结构图...")
        start_time = time.time()
        
        structure_maps = generate_controlnet_structure_maps(
            image, mask,
            methods=['canny', 'depth', 'normal', 'edge', 'gradient'],
            save_debug=True
        )
        
        end_time = time.time()
        print(f"结构图生成时间: {end_time - start_time:.2f}秒")
        
        print(f"生成了 {len(structure_maps)} 种结构图:")
        for method, structure_map in structure_maps.items():
            print(f"- {method}: {structure_map.shape}")
        
        # 测试ControlNet输入创建
        print("\n创建ControlNet输入...")
        controlnet_inputs = create_controlnet_inputs(
            image, mask,
            structure_methods=['canny', 'depth'],
            save_debug=True
        )
        
        print(f"ControlNet输入包含 {len(controlnet_inputs['control_images'])} 个控制图像:")
        for control_image in controlnet_inputs['control_images']:
            print(f"- {control_image['method']}: 强度 {control_image['strength']}")
        
        return True
        
    except ImportError as e:
        print(f"ControlNet结构增强器不可用: {e}")
        return False

def test_main_pipeline_integration():
    """测试主流程集成"""
    print("\n=== 测试主流程集成 ===")
    
    try:
        from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
        
        # 创建测试图像
        image = create_test_nail_image()
        
        # 初始化处理器
        nail_processor = NailSDXLInpaintOpenCV()
        
        print("测试掩码生成（包含边缘引导）...")
        start_time = time.time()
        
        # 生成掩码（会自动应用边缘引导）
        mask = nail_processor.generate_mask_u2net(image, "test_edge_controlnet.png")
        
        end_time = time.time()
        print(f"掩码生成时间: {end_time - start_time:.2f}秒")
        
        # 保存结果
        cv2.imwrite('test_main_pipeline_edge_guided_mask.png', mask)
        print("掩码已保存: test_main_pipeline_edge_guided_mask.png")
        
        # 显示掩码信息
        print(f"掩码尺寸: {mask.shape}")
        print(f"掩码值范围: {mask.min()} - {mask.max()}")
        
        # 统计掩码像素
        nail_pixels = np.sum(mask > 128)
        total_pixels = mask.shape[0] * mask.shape[1]
        nail_ratio = nail_pixels / total_pixels * 100
        print(f"指甲区域占比: {nail_ratio:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"主流程集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("测试边缘引导和ControlNet结构控制增强功能")
    print("=" * 60)
    
    # 测试1: 边缘引导增强
    success1 = test_edge_guided_enhancement()
    
    # 测试2: ControlNet结构图生成
    success2 = test_controlnet_structure()
    
    # 测试3: 主流程集成
    success3 = test_main_pipeline_integration()
    
    print("\n=== 测试结果 ===")
    print(f"边缘引导增强测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"ControlNet结构图测试: {'✅ 成功' if success2 else '❌ 失败'}")
    print(f"主流程集成测试: {'✅ 成功' if success3 else '❌ 失败'}")
    
    if success1 and success2 and success3:
        print("\n🎉 所有测试通过！")
        print("边缘引导和ControlNet结构控制功能已成功集成。")
        print("\n生成的文件:")
        print("- test_edge_controlnet_image.png: 测试图像")
        print("- test_edge_controlnet_mask.png: 原始掩码")
        print("- test_edge_guided_result_*.png: 边缘引导结果")
        print("- test_main_pipeline_edge_guided_mask.png: 主流程集成结果")
        print("- data/output/edge_guided_debug/: 边缘引导调试图像")
        print("- data/output/controlnet_structure_debug/: ControlNet结构调试图像")
        
        print("\n功能特点:")
        print("✅ 边缘引导：使用Canny/Sobel边缘辅助掩码边界")
        print("✅ 结构控制：生成多种结构图约束AI渲染")
        print("✅ 多级增强：Active Contour + 边缘引导 + 结构控制")
        print("✅ 调试输出：详细的调试图像和对比图")
        print("✅ 参数可调：边缘权重、平滑因子、控制强度等")
        
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        print("可能的原因:")
        print("- 缺少必要的依赖模块")
        print("- 模块导入路径问题")
        print("- 文件权限问题")

if __name__ == "__main__":
    main() 