#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全流程灰度掩码测试脚本
测试灰度掩码在颜色准确性和边缘质量方面的表现
"""

import cv2
import numpy as np
import os
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV

def test_color_accuracy():
    """
    测试全流程灰度掩码的颜色准确性
    """
    print("=== 全流程灰度掩码颜色准确性测试 ===")
    
    # 初始化处理器
    processor = NailSDXLInpaintOpenCV()
    
    # 测试颜色列表
    test_colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 192, 203),  # 粉色
        (128, 0, 128),    # 紫色
        (255, 165, 0),    # 橙色
    ]
    
    # 测试图片
    test_image_path = "data/test_images/11111.png"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return
    
    # 加载测试图片
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法加载测试图片: {test_image_path}")
        return
    
    # 生成掩码
    print("生成U²-Net掩码...")
    mask = processor.generate_mask_u2net(image)
    
    # 保存掩码用于调试
    cv2.imwrite("data/output/debug_gray_mask.png", mask)
    print("掩码已保存到: data/output/debug_gray_mask.png")
    
    results = {}
    
    for i, target_color in enumerate(test_colors):
        print(f"\n测试颜色 {i+1}/{len(test_colors)}: RGB{target_color}")
        
        # 创建颜色块作为参考
        color_block = np.full(image.shape, target_color, dtype=np.uint8)
        
        # 使用全流程灰度掩码处理
        try:
            result = processor.process_with_ai_fusion(
                color_transfer_img=color_block,
                mask=mask,
                target_color=target_color
            )
            
            # 计算颜色相似度
            similarity = calculate_color_similarity(result, target_color, mask)
            results[target_color] = similarity
            
            print(f"颜色相似度: {similarity:.2%}")
            
            # 保存结果
            output_path = f"data/output/gray_mask_test_color_{i+1}.png"
            cv2.imwrite(output_path, result)
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"处理失败: {e}")
            results[target_color] = 0.0
    
    # 输出总结
    print("\n=== 测试结果总结 ===")
    for color, similarity in results.items():
        print(f"RGB{color}: {similarity:.2%}")
    
    avg_similarity = np.mean(list(results.values()))
    print(f"\n平均颜色相似度: {avg_similarity:.2%}")
    
    if avg_similarity >= 0.95:
        print("✅ 全流程灰度掩码颜色准确性测试通过！")
    else:
        print("⚠️ 颜色准确性需要进一步优化")

def calculate_color_similarity(image, target_color, mask):
    """
    计算颜色相似度
    """
    # 在掩码区域计算平均颜色
    mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0
    masked_pixels = image[mask_3d[:, :, 0] > 0.5]
    
    if len(masked_pixels) == 0:
        return 0.0
    
    current_color = np.mean(masked_pixels, axis=0)
    target_color_array = np.array(target_color)
    
    # 计算颜色距离
    color_distance = np.linalg.norm(current_color - target_color_array)
    max_distance = np.sqrt(255**2 * 3)  # 最大可能距离
    
    similarity = 1.0 - (color_distance / max_distance)
    return similarity

def test_edge_quality():
    """
    测试全流程灰度掩码的边缘质量
    """
    print("\n=== 全流程灰度掩码边缘质量测试 ===")
    
    # 初始化处理器
    processor = NailSDXLInpaintOpenCV()
    
    # 测试图片
    test_image_path = "data/test_images/11111.png"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return
    
    # 加载测试图片
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法加载测试图片: {test_image_path}")
        return
    
    # 生成掩码
    print("生成U²-Net掩码...")
    mask = processor.generate_mask_u2net(image)
    
    # 测试目标颜色
    target_color = (255, 192, 203)  # 粉色
    
    # 创建颜色块
    color_block = np.full(image.shape, target_color, dtype=np.uint8)
    
    # 使用全流程灰度掩码处理
    try:
        result = processor.process_with_ai_fusion(
            color_transfer_img=color_block,
            mask=mask,
            target_color=target_color
        )
        
        # 保存结果用于边缘质量分析
        output_path = "data/output/gray_mask_edge_test.png"
        cv2.imwrite(output_path, result)
        print(f"边缘质量测试结果已保存到: {output_path}")
        
        # 分析边缘质量
        edge_quality = analyze_edge_quality(result, mask)
        print(f"边缘质量评分: {edge_quality:.2f}/10")
        
    except Exception as e:
        print(f"边缘质量测试失败: {e}")

def analyze_edge_quality(image, mask):
    """
    分析边缘质量
    """
    # 计算掩码边缘
    mask_edges = cv2.Canny(mask, 50, 150)
    
    # 计算图像边缘
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(gray, 50, 150)
    
    # 计算边缘一致性
    edge_overlap = np.logical_and(mask_edges > 0, image_edges > 0)
    overlap_ratio = np.sum(edge_overlap) / (np.sum(mask_edges > 0) + 1e-6)
    
    # 计算边缘平滑度
    edge_smoothness = 1.0 - (np.std(mask) / 255.0)
    
    # 综合评分
    quality_score = (overlap_ratio * 5 + edge_smoothness * 5)
    return quality_score

def compare_binary_vs_gray_mask():
    """
    对比二值掩码和灰度掩码的效果
    """
    print("\n=== 二值掩码 vs 灰度掩码对比测试 ===")
    
    # 初始化处理器
    processor = NailSDXLInpaintOpenCV()
    
    # 测试图片
    test_image_path = "data/test_images/11111.png"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return
    
    # 加载测试图片
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法加载测试图片: {test_image_path}")
        return
    
    # 生成掩码
    print("生成U²-Net掩码...")
    gray_mask = processor.generate_mask_u2net(image)
    
    # 创建二值掩码
    _, binary_mask = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_BINARY)
    
    # 测试颜色
    target_color = (255, 192, 203)  # 粉色
    color_block = np.full(image.shape, target_color, dtype=np.uint8)
    
    # 测试灰度掩码
    print("测试灰度掩码...")
    try:
        gray_result = processor.process_with_ai_fusion(
            color_transfer_img=color_block,
            mask=gray_mask,
            target_color=target_color
        )
        
        gray_similarity = calculate_color_similarity(gray_result, target_color, gray_mask)
        print(f"灰度掩码颜色相似度: {gray_similarity:.2%}")
        
        cv2.imwrite("data/output/gray_mask_comparison.png", gray_result)
        
    except Exception as e:
        print(f"灰度掩码测试失败: {e}")
    
    # 测试二值掩码（使用修改前的函数）
    print("测试二值掩码...")
    try:
        # 这里需要临时修改函数来测试二值掩码
        binary_result = test_binary_mask_processing(processor, color_block, binary_mask, target_color)
        
        binary_similarity = calculate_color_similarity(binary_result, target_color, binary_mask)
        print(f"二值掩码颜色相似度: {binary_similarity:.2%}")
        
        cv2.imwrite("data/output/binary_mask_comparison.png", binary_result)
        
    except Exception as e:
        print(f"二值掩码测试失败: {e}")

def test_binary_mask_processing(processor, color_block, binary_mask, target_color):
    """
    临时函数用于测试二值掩码处理
    """
    # 这里实现二值掩码的处理逻辑
    # 由于我们修改了主流程，这里需要临时实现
    return color_block  # 临时返回

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("data/output", exist_ok=True)
    
    # 运行测试
    test_color_accuracy()
    test_edge_quality()
    compare_binary_vs_gray_mask()
    
    print("\n=== 全流程灰度掩码测试完成 ===") 