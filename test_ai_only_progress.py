#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试只有AI生成阶段才返回进度的功能
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

def test_ai_only_progress():
    """测试只有AI生成阶段才返回进度的功能"""
    print("测试只有AI生成阶段才返回进度的功能")
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
    
    print("\n2. 监控进度（应该只显示AI生成进度）...")
    start_time = time.time()
    max_wait_time = 300  # 5分钟超时
    
    progress_updates = []
    last_progress = -1
    
    while time.time() - start_time < max_wait_time:
        response = requests.get(f"{base_url}/get_progress/{task_id}")
        
        if response.status_code != 200:
            print(f"❌ 查询进度失败: {response.status_code}")
            break
        
        result = response.json()
        
        if result['statusCode'] != 200:
            print(f"❌ 查询进度失败: {result['message']}")
            break
        
        progress = result['progress']
        message = result['message']
        is_completed = result.get('is_completed', False)
        
        # 只记录有变化的进度更新
        if progress != last_progress:
            progress_updates.append({
                'time': time.time() - start_time,
                'progress': progress,
                'message': message,
                'current_step': result.get('current_step', 0),
                'total_steps': result.get('total_steps', 0)
            })
            last_progress = progress
            
            print(f"进度更新: {progress:.1%} - {message}")
            if result.get('total_steps', 0) > 0:
                print(f"  AI步骤: {result.get('current_step', 0)}/{result.get('total_steps', 0)}")
        
        if is_completed:
            print("✅ 任务完成！")
            
            # 验证返回的Data URL格式
            data_url = result.get('data', '')
            if data_url.startswith('data:image/png;base64,'):
                print("✅ 返回格式正确：Data URL")
                
                # 保存结果图像
                try:
                    # 提取base64数据
                    base64_data = data_url.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                    
                    with open(f"test_result_{task_id}.png", "wb") as f:
                        f.write(image_data)
                    print(f"✅ 结果图像已保存到: test_result_{task_id}.png")
                except Exception as e:
                    print(f"❌ 保存结果图像失败: {e}")
            else:
                print("❌ 返回格式错误")
            
            break
        
        time.sleep(1)  # 每秒查询一次
    else:
        print("❌ 任务超时")
        return False
    
    print(f"\n3. 进度更新统计:")
    print(f"总共收到 {len(progress_updates)} 次进度更新")
    
    # 分析进度更新
    ai_progress_count = 0
    other_progress_count = 0
    
    for update in progress_updates:
        if update['total_steps'] > 0:
            ai_progress_count += 1
            print(f"  AI生成进度: {update['progress']:.1%} (步骤 {update['current_step']}/{update['total_steps']})")
        else:
            other_progress_count += 1
            print(f"  其他进度: {update['progress']:.1%} - {update['message']}")
    
    print(f"\nAI生成进度更新次数: {ai_progress_count}")
    print(f"其他进度更新次数: {other_progress_count}")
    
    # 验证是否符合预期
    if other_progress_count == 0:
        print("✅ 符合预期：只有AI生成阶段返回进度")
    else:
        print("❌ 不符合预期：还有其他阶段返回进度")
    
    # 清理任务
    print(f"\n4. 清理任务...")
    response = requests.get(f"{base_url}/cleanup_task/{task_id}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 清理结果: {result['message']}")
    else:
        print("❌ 清理任务失败")
    
    # 清理测试文件
    for file in ["test_original.jpg", "test_reference.jpg"]:
        if os.path.exists(file):
            os.remove(file)
    
    return True

if __name__ == "__main__":
    print("AI生成进度专用测试")
    print("=" * 60)
    
    try:
        success = test_ai_only_progress()
        if success:
            print("\n✅ 测试完成！")
        else:
            print("\n❌ 测试失败！")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc() 