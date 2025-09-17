#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Reference-Only美甲迁移方案
验证新方案的效果和性能
"""

import os
import cv2
import numpy as np
from nail_reference_only_transfer import NailReferenceOnlyTransfer
import time

def create_test_images():
    """创建测试图像（如果不存在）"""
    test_dir = "data"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建简单的手部测试图像
    hand_image = np.ones((512, 512, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制简单的手部轮廓
    cv2.rectangle(hand_image, (200, 100), (312, 400), (200, 180, 160), -1)  # 手部
    cv2.rectangle(hand_image, (200, 80), (312, 120), (220, 200, 180), -1)   # 指甲区域
    
    hand_path = os.path.join(test_dir, "hand_photo.jpg")
    cv2.imwrite(hand_path, hand_image)
    print(f"创建测试手部图像: {hand_path}")
    
    # 创建指甲样板图像
    nail_sample = np.ones((256, 256, 3), dtype=np.uint8) * 255
    # 绘制彩色指甲样板
    cv2.rectangle(nail_sample, (50, 50), (206, 206), (255, 100, 150), -1)  # 粉色指甲
    
    nail_path = os.path.join(test_dir, "nail_sample.jpg")
    cv2.imwrite(nail_path, nail_sample)
    print(f"创建测试指甲样板: {nail_path}")
    
    return hand_path, nail_path

def test_api_connection():
    """测试API连接"""
    print("=" * 50)
    print("测试API连接...")
    
    transfer = NailReferenceOnlyTransfer()
    
    if transfer.check_api_health():
        print("✅ API连接正常")
        return True
    else:
        print("❌ API连接失败")
        print("请确保:")
        print("1. sd-webui 正在运行")
        print("2. sd-webui-controlnet 扩展已安装并更新到 v1.1.400+")
        print("3. API服务在 http://127.0.0.1:7860 可用")
        return False

def test_mask_generation():
    """测试掩码生成"""
    print("=" * 50)
    print("测试掩码生成...")
    
    # 创建测试图像
    hand_path, nail_path = create_test_images()
    
    # 加载图像
    hand_image = cv2.imread(hand_path)
    transfer = NailReferenceOnlyTransfer()
    
    # 生成掩码
    mask = transfer.generate_nail_mask(hand_image)
    enhanced_mask = transfer.enhance_mask_quality(mask)
    
    # 保存掩码用于检查
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "original_mask.jpg"), mask)
    cv2.imwrite(os.path.join(output_dir, "enhanced_mask.jpg"), enhanced_mask)
    
    print("✅ 掩码生成完成")
    print(f"掩码已保存到: {output_dir}/")
    
    return hand_path, nail_path

def test_growth_effects():
    """测试生长效果"""
    print("=" * 50)
    print("测试生长效果...")
    
    # 创建测试图像
    test_image = np.ones((400, 300, 3), dtype=np.uint8) * 200
    cv2.rectangle(test_image, (100, 100), (200, 300), (255, 150, 200), -1)
    
    # 创建测试掩码
    mask = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (200, 300), 255, -1)
    
    transfer = NailReferenceOnlyTransfer()
    result = transfer.add_growth_effects(test_image, mask)
    
    # 保存结果
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "growth_effects_test.jpg"), result)
    print("✅ 生长效果测试完成")
    print(f"结果已保存到: {output_dir}/growth_effects_test.jpg")

def test_full_pipeline():
    """测试完整流程"""
    print("=" * 50)
    print("测试完整美甲迁移流程...")
    
    # 检查API连接
    if not test_api_connection():
        print("跳过完整流程测试（API不可用）")
        return
    
    # 准备测试图像
    hand_path, nail_path = test_mask_generation()
    
    # 初始化迁移器
    transfer = NailReferenceOnlyTransfer()
    
    # 设置输出路径
    output_path = "test_output/reference_only_full_test.jpg"
    
    # 执行完整迁移
    start_time = time.time()
    result = transfer.process_nail_transfer(
        hand_image_path=hand_path,
        nail_sample_path=nail_path,
        output_path=output_path
    )
    end_time = time.time()
    
    if result is not None:
        print(f"✅ 完整流程测试成功")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        print(f"结果已保存到: {output_path}")
        
        # 显示结果
        cv2.imshow("完整流程测试结果", result)
        cv2.waitKey(3000)  # 显示3秒
        cv2.destroyAllWindows()
    else:
        print("❌ 完整流程测试失败")

def test_parameter_tuning():
    """测试参数调优"""
    print("=" * 50)
    print("测试参数调优...")
    
    if not test_api_connection():
        print("跳过参数调优测试（API不可用）")
        return
    
    # 准备测试图像
    hand_path, nail_path = create_test_images()
    
    # 加载图像
    hand_image = cv2.imread(hand_path)
    nail_sample = cv2.imread(nail_path)
    
    transfer = NailReferenceOnlyTransfer()
    
    # 生成掩码
    mask = transfer.generate_nail_mask(hand_image)
    mask = transfer.enhance_mask_quality(mask)
    
    # 测试不同参数组合
    test_params = [
        {"strength": 0.6, "cfg_scale": 7.0, "steps": 15},
        {"strength": 0.8, "cfg_scale": 7.0, "steps": 20},
        {"strength": 1.0, "cfg_scale": 8.0, "steps": 25},
    ]
    
    output_dir = "test_output/parameter_tuning"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, params in enumerate(test_params):
        print(f"测试参数组合 {i+1}: {params}")
        
        result = transfer.transfer_with_reference_only(
            hand_image=hand_image,
            nail_sample=nail_sample,
            mask=mask,
            **params
        )
        
        output_path = os.path.join(output_dir, f"param_test_{i+1}.jpg")
        cv2.imwrite(output_path, result)
        print(f"结果已保存到: {output_path}")
    
    print("✅ 参数调优测试完成")

def main():
    """主测试函数"""
    print("Reference-Only 美甲迁移方案测试")
    print("=" * 60)
    
    # 1. 测试API连接
    test_api_connection()
    
    # 2. 测试掩码生成
    test_mask_generation()
    
    # 3. 测试生长效果
    test_growth_effects()
    
    # 4. 测试完整流程
    test_full_pipeline()
    
    # 5. 测试参数调优
    test_parameter_tuning()
    
    print("=" * 60)
    print("所有测试完成！")
    print("请检查 test_output/ 目录中的结果文件")

if __name__ == "__main__":
    main() 