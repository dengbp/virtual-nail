#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试清理功能是否正确保留原图、最终效果图和参考色块图
"""

import os
import sys
import cv2
import numpy as np
import base64
import json
import time
import requests
from pathlib import Path

def create_test_images():
    """创建测试图像"""
    print("创建测试图像...")
    
    # 创建原图（模拟手指图像）
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    # 添加一些指甲形状
    cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 360, (200, 200, 200), -1)
    cv2.imwrite("test_original.jpg", img)
    
    # 创建参考色图（纯色）
    ref_img = np.ones((100, 100, 3), dtype=np.uint8) * [255, 100, 100]  # 红色
    cv2.imwrite("test_reference.jpg", ref_img)
    
    print("测试图像创建完成")

def encode_image_to_base64(image_path):
    """将图像编码为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    print(f"  {description}: {'✅ 存在' if exists else '❌ 不存在'} - {file_path}")
    return exists

def test_cleanup_preserve_files():
    """测试清理功能是否正确保留重要文件"""
    print("测试清理功能是否正确保留重要文件")
    print("=" * 60)
    
    # 创建测试图像
    create_test_images()
    
    # 编码图像
    original_b64 = encode_image_to_base64("test_original.jpg")
    reference_b64 = encode_image_to_base64("test_reference.jpg")
    
    # API配置
    base_url = "http://localhost:80"
    
    print("\n1. 提交任务...")
    response = requests.post(f"{base_url}/edit_nail", data={
        "img": original_b64,
        "ref_img": reference_b64
    }, timeout=30)
    
    if response.status_code != 200:
        print(f"❌ 提交任务失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False
    
    result = response.json()
    if result['statusCode'] != 200:
        print(f"❌ 提交任务失败: {result['message']}")
        return False
    
    task_id = result['task_id']
    print(f"✅ 任务提交成功，任务ID: {task_id}")
    
    print("\n2. 等待任务完成...")
    start_time = time.time()
    max_wait_time = 300  # 5分钟超时
    
    while time.time() - start_time < max_wait_time:
        response = requests.get(f"{base_url}/get_progress/{task_id}")
        
        if response.status_code != 200:
            print(f"❌ 查询进度失败: {response.status_code}")
            break
        
        result = response.json()
        
        if result['statusCode'] != 200:
            print(f"❌ 查询进度失败: {result['message']}")
            break
        
        is_completed = result.get('is_completed', False)
        
        if is_completed:
            print("✅ 任务完成！")
            break
        
        time.sleep(1)  # 每秒查询一次
    else:
        print("❌ 任务超时")
        return False
    
    print("\n3. 检查任务完成后的文件状态...")
    
    # 定义文件路径
    img_path = os.path.join("data/test_images", task_id + ".jpg")
    ref_path = os.path.join("data/reference", task_id + "_reference.jpg")
    mask_path = os.path.join("data/test_masks", task_id + "_mask.jpg")
    final_path = os.path.join("data/output/final", task_id + "_final.png")
    
    print("任务完成后的文件状态:")
    original_exists = check_file_exists(img_path, "原图")
    reference_exists = check_file_exists(ref_path, "参考色块图")
    mask_exists = check_file_exists(mask_path, "掩码文件")
    final_exists = check_file_exists(final_path, "最终效果图")
    
    print("\n4. 执行清理操作...")
    response = requests.get(f"{base_url}/cleanup_task/{task_id}")
    
    if response.status_code != 200:
        print(f"❌ 清理任务失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False
    
    result = response.json()
    if result['statusCode'] != 200:
        print(f"❌ 清理任务失败: {result['message']}")
        return False
    
    print(f"✅ 清理结果: {result['message']}")
    print(f"清理的文件: {result.get('cleaned_files', [])}")
    print(f"保留的文件: {result.get('preserved_files', [])}")
    
    print("\n5. 检查清理后的文件状态...")
    print("清理后的文件状态:")
    original_after = check_file_exists(img_path, "原图")
    reference_after = check_file_exists(ref_path, "参考色块图")
    mask_after = check_file_exists(mask_path, "掩码文件")
    final_after = check_file_exists(final_path, "最终效果图")
    
    print("\n6. 验证清理结果...")
    
    # 验证重要文件是否被保留
    important_files_preserved = True
    if not original_after:
        print("❌ 原图文件被意外删除")
        important_files_preserved = False
    if not reference_after:
        print("❌ 参考色块图文件被意外删除")
        important_files_preserved = False
    if not final_after:
        print("❌ 最终效果图文件被意外删除")
        important_files_preserved = False
    
    # 验证中间文件是否被清理
    intermediate_files_cleaned = True
    if mask_after:
        print("❌ 掩码文件未被清理")
        intermediate_files_cleaned = False
    
    if important_files_preserved and intermediate_files_cleaned:
        print("✅ 清理功能验证通过：")
        print("  - 重要文件（原图、参考色块图、最终效果图）已正确保留")
        print("  - 中间处理文件（掩码）已正确清理")
        return True
    else:
        print("❌ 清理功能验证失败")
        return False

if __name__ == "__main__":
    print("清理功能保留文件测试")
    print("=" * 60)
    
    try:
        success = test_cleanup_preserve_files()
        if success:
            print("\n✅ 测试完成！")
        else:
            print("\n❌ 测试失败！")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc() 