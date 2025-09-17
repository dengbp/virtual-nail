#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试任务ID返回功能
"""

import requests
import base64
import cv2
import numpy as np
import time
import json

def create_test_image(width=100, height=100, color=(255, 0, 0)):
    """创建测试图像"""
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img

def image_to_base64(img):
    """将图像转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def test_task_id_return():
    """测试任务ID返回功能"""
    print("开始测试任务ID返回功能...")
    
    # 创建测试图像
    test_img = create_test_image(200, 200, (255, 0, 0))  # 红色图像
    ref_img = create_test_image(200, 200, (0, 0, 255))   # 蓝色图像
    
    # 转换为base64
    img_b64 = image_to_base64(test_img)
    ref_b64 = image_to_base64(ref_img)
    
    print(f"测试图像大小: {len(img_b64)} 字符")
    print(f"参考图像大小: {len(ref_b64)} 字符")
    
    # 发送请求
    url = "http://localhost:80/edit_nail"
    data = {
        "img": img_b64,
        "ref_img": ref_b64
    }
    
    try:
        print(f"发送请求到: {url}")
        response = requests.post(url, data=data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查任务ID
            if "task_id" in result:
                task_id = result["task_id"]
                print(f"✅ 成功获取任务ID: {task_id}")
                
                # 测试进度查询
                print(f"\n测试进度查询...")
                progress_url = f"http://localhost:80/get_progress/{task_id}"
                progress_response = requests.get(progress_url, timeout=10)
                
                if progress_response.status_code == 200:
                    progress_result = progress_response.json()
                    print(f"进度查询成功: {json.dumps(progress_result, indent=2, ensure_ascii=False)}")
                else:
                    print(f"❌ 进度查询失败: {progress_response.status_code}")
                    print(f"错误信息: {progress_response.text}")
                
                return task_id
            else:
                print("❌ 响应中没有task_id字段")
                return None
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ 其他异常: {str(e)}")
        return None

def test_multiple_requests():
    """测试多个请求的任务ID唯一性"""
    print("\n" + "="*50)
    print("测试多个请求的任务ID唯一性...")
    
    task_ids = []
    for i in range(3):
        print(f"\n第 {i+1} 次请求:")
        task_id = test_task_id_return()
        if task_id:
            task_ids.append(task_id)
        time.sleep(1)  # 避免请求过于频繁
    
    print(f"\n获取到的任务ID列表: {task_ids}")
    
    # 检查唯一性
    if len(task_ids) == len(set(task_ids)):
        print("✅ 所有任务ID都是唯一的")
    else:
        print("❌ 存在重复的任务ID")
    
    return task_ids

if __name__ == "__main__":
    print("="*60)
    print("任务ID返回功能测试")
    print("="*60)
    
    # 测试单个请求
    task_id = test_task_id_return()
    
    # 测试多个请求
    task_ids = test_multiple_requests()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60) 