#!/usr/bin/env python3
"""
测试SDXL指甲生成的进度回调功能
"""

import cv2
import numpy as np
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
import time

def test_progress_callback():
    """测试进度回调功能"""
    
    # 初始化美甲处理器
    nail = NailSDXLInpaintOpenCV()
    
    # 创建测试图像和掩码
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # 白色背景
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    # 创建一个简单的指甲形状掩码
    cv2.rectangle(test_mask, (200, 200), (312, 312), 255, -1)
    
    # 进度回调函数
    def progress_callback(progress, current_step, total_steps):
        if total_steps > 0:
            print(f"AI生成进度: {progress:.1%} ({current_step}/{total_steps})")
        else:
            print(f"处理进度: {progress:.1%}")
    
    print("开始测试SDXL指甲生成进度回调...")
    print("=" * 50)
    
    try:
        # 测试基础SDXL推理
        print("\n1. 测试基础SDXL推理:")
        result1 = nail.sdxl_inpaint(
            image=test_img,
            mask=test_mask,
            prompt="realistic nail with glossy finish",
            callback=progress_callback
        )
        print("基础SDXL推理完成!")
        
        # 测试ControlNet SDXL推理
        print("\n2. 测试ControlNet SDXL推理:")
        result2 = nail.sdxl_inpaint_controlnet_canny(
            image=test_img,
            mask=test_mask,
            prompt="ultra realistic nail with mirror-like shine",
            negative_prompt="blur, low quality, cartoon",
            callback=progress_callback
        )
        print("ControlNet SDXL推理完成!")
        
        # 保存结果
        cv2.imwrite("test_result_basic.png", result1)
        cv2.imwrite("test_result_controlnet.png", result2)
        print("\n测试结果已保存到 test_result_basic.png 和 test_result_controlnet.png")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("请确保已正确安装所有依赖和模型")

def test_progress_with_real_image():
    """使用真实图像测试进度回调"""
    
    # 检查是否有测试图像
    test_img_path = "data/test_images"
    if not Path(test_img_path).exists():
        print(f"测试图像目录 {test_img_path} 不存在，跳过真实图像测试")
        return
    
    nail = NailSDXLInpaintOpenCV()
    
    # 进度回调函数
    def progress_callback(progress, current_step, total_steps):
        if total_steps > 0:
            print(f"AI生成进度: {progress:.1%} ({current_step}/{total_steps})")
        else:
            print(f"处理进度: {progress:.1%}")
    
    # 查找第一个图像文件
    img_files = list(Path(test_img_path).glob("*.*"))
    if not img_files:
        print("未找到测试图像文件")
        return
    
    test_img_path = img_files[0]
    print(f"\n使用真实图像测试: {test_img_path}")
    
    # 读取图像
    img = cv2.imread(str(test_img_path))
    if img is None:
        print(f"无法读取图像: {test_img_path}")
        return
    
    # 创建简单掩码（实际应用中应该使用U2Net生成）
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # 在图像中心创建一个矩形作为指甲区域
    center_x, center_y = w // 2, h // 2
    nail_size = min(w, h) // 4
    cv2.rectangle(mask, 
                 (center_x - nail_size, center_y - nail_size),
                 (center_x + nail_size, center_y + nail_size), 
                 255, -1)
    
    try:
        result = nail.sdxl_inpaint_controlnet_canny(
            image=img,
            mask=mask,
            prompt="ultra realistic nail with glossy finish and natural highlights",
            negative_prompt="blur, low quality, cartoon, fake",
            callback=progress_callback
        )
        
        # 保存结果
        output_path = f"test_real_result_{Path(test_img_path).stem}.png"
        cv2.imwrite(output_path, result)
        print(f"\n真实图像测试结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"真实图像测试过程中出现错误: {e}")

if __name__ == "__main__":
    print("SDXL指甲生成进度回调测试")
    print("=" * 50)
    
    # 测试基础功能
    test_progress_callback()
    
    # 测试真实图像
    test_progress_with_real_image()
    
    print("\n测试完成!") 