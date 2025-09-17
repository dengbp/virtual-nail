#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试下载最终效果图接口
"""

import os
import requests
import base64
from pathlib import Path

def test_download_final():
    """测试下载最终效果图接口"""
    print("测试下载最终效果图接口")
    print("=" * 60)
    
    # 使用一个已存在的任务ID
    task_id = "160206571"
    
    # API配置
    base_url = "http://localhost:80"
    
    # 定义文件路径
    final_path = os.path.join("data/output/final", task_id + "_final.png")
    
    print(f"测试任务ID: {task_id}")
    print(f"预期文件路径: {final_path}")
    
    # 检查文件是否存在
    if not os.path.exists(final_path):
        print(f"❌ 文件不存在: {final_path}")
        print("请确保该任务已完成并生成了最终效果图")
        return False
    
    print(f"✅ 文件存在，大小: {os.path.getsize(final_path)} 字节")
    
    print("\n1. 调用下载接口...")
    response = requests.get(f"{base_url}/download_final/{task_id}")
    
    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False
    
    result = response.json()
    print(f"✅ 请求成功")
    print(f"状态码: {result.get('statusCode')}")
    print(f"消息: {result.get('message')}")
    
    if result.get('statusCode') != 200:
        print(f"❌ 业务逻辑失败: {result.get('message')}")
        return False
    
    # 检查返回的数据
    data = result.get('data', '')
    if not data:
        print("❌ 返回的data字段为空")
        return False
    
    print(f"✅ 返回数据长度: {len(data)} 字符")
    
    # 验证Data URL格式
    if not data.startswith('data:image/png;base64,'):
        print("❌ 返回的数据不是有效的Data URL格式")
        return False
    
    print("✅ Data URL格式正确")
    
    # 提取base64数据并解码验证
    try:
        base64_data = data.replace('data:image/png;base64,', '')
        decoded_data = base64.b64decode(base64_data)
        print(f"✅ Base64解码成功，解码后大小: {len(decoded_data)} 字节")
        
        # 保存下载的图片进行验证
        download_path = f"downloaded_{task_id}_final.png"
        with open(download_path, "wb") as f:
            f.write(decoded_data)
        
        print(f"✅ 下载的图片已保存到: {download_path}")
        
        # 比较文件大小
        original_size = os.path.getsize(final_path)
        downloaded_size = len(decoded_data)
        
        if original_size == downloaded_size:
            print("✅ 文件大小一致，下载成功")
        else:
            print(f"⚠️ 文件大小不一致: 原始={original_size}, 下载={downloaded_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Base64解码失败: {e}")
        return False

def test_download_nonexistent():
    """测试下载不存在的文件"""
    print("\n测试下载不存在的文件")
    print("=" * 60)
    
    # 使用一个不存在的任务ID
    task_id = "999999999"
    
    # API配置
    base_url = "http://localhost:80"
    
    print(f"测试任务ID: {task_id}")
    
    print("1. 调用下载接口...")
    response = requests.get(f"{base_url}/download_final/{task_id}")
    
    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False
    
    result = response.json()
    print(f"✅ 请求成功")
    print(f"状态码: {result.get('statusCode')}")
    print(f"消息: {result.get('message')}")
    
    if result.get('statusCode') == -1:
        print("✅ 正确处理了文件不存在的情况")
        return True
    else:
        print("❌ 应该返回错误状态码")
        return False

if __name__ == "__main__":
    print("下载最终效果图接口测试")
    print("=" * 60)
    
    try:
        # 测试正常下载
        success1 = test_download_final()
        
        # 测试下载不存在的文件
        success2 = test_download_nonexistent()
        
        if success1 and success2:
            print("\n✅ 所有测试通过！")
        else:
            print("\n❌ 部分测试失败！")
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc() 