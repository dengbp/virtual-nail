#!/usr/bin/env python3
"""
测试大文件上传功能
验证413错误是否已修复
"""

import requests
import base64
import cv2
import numpy as np
import time

def create_large_test_image(width=1920, height=1080):
    """创建一个较大的测试图像"""
    # 创建一个彩色渐变图像
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建渐变效果
    for i in range(height):
        for j in range(width):
            img[i, j] = [
                int(255 * i / height),  # R
                int(255 * j / width),   # G
                128                     # B
            ]
    
    return img

def image_to_base64(img):
    """将OpenCV图像转换为base64字符串"""
    # 编码为JPEG
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # 转换为base64
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64

def test_large_upload():
    """测试大文件上传"""
    print("开始测试大文件上传功能...")
    
    # 创建较大的测试图像
    print("创建测试图像...")
    test_img = create_large_test_image(1920, 1080)  # 1080p图像
    ref_img = create_large_test_image(800, 600)     # 较小的参考图像
    
    # 转换为base64
    img_b64 = image_to_base64(test_img)
    ref_b64 = image_to_base64(ref_img)
    
    print(f"测试图像大小: {len(img_b64)} 字符 ({len(img_b64) / 1024 / 1024:.2f} MB)")
    print(f"参考图像大小: {len(ref_b64)} 字符 ({len(ref_b64) / 1024 / 1024:.2f} MB)")
    
    # 发送请求
    url = "http://localhost:80/edit_nail"
    data = {
        "img": img_b64,
        "ref_img": ref_b64
    }
    
    try:
        print(f"发送请求到: {url}")
        start_time = time.time()
        
        response = requests.post(url, data=data, timeout=60)
        
        end_time = time.time()
        print(f"请求耗时: {end_time - start_time:.2f} 秒")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"请求成功: {result}")
            
            if 'task_id' in result:
                task_id = result['task_id']
                print(f"任务ID: {task_id}")
                
                # 等待一段时间后查询进度
                print("等待5秒后查询进度...")
                time.sleep(5)
                
                progress_url = f"http://localhost:80/get_progress/{task_id}"
                progress_response = requests.get(progress_url)
                
                if progress_response.status_code == 200:
                    progress_result = progress_response.json()
                    print(f"进度查询结果: {progress_result}")
                else:
                    print(f"进度查询失败: {progress_response.status_code}")
                    
        elif response.status_code == 413:
            print("❌ 仍然出现413错误 - 文件太大")
            return False
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.RequestEntityTooLarge:
        print("❌ 客户端检测到文件太大")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    
    print("✅ 大文件上传测试通过")
    return True

if __name__ == "__main__":
    success = test_large_upload()
    if success:
        print("\n🎉 大文件上传功能修复成功！")
    else:
        print("\n❌ 大文件上传功能仍有问题") 