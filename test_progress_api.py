#!/usr/bin/env python3
"""
测试美甲生成进度API
"""

import requests
import base64
import time
import json
from pathlib import Path

def test_progress_api():
    """测试进度API功能"""
    
    # 服务器地址
    base_url = "http://localhost"
    
    # 检查是否有测试图像
    test_img_path = "data/test_images"
    ref_img_path = "data/reference"
    
    if not Path(test_img_path).exists() or not Path(ref_img_path).exists():
        print("测试图像目录不存在，创建示例图像...")
        create_test_images()
    
    # 查找测试图像
    img_files = list(Path(test_img_path).glob("*.*"))
    ref_files = list(Path(ref_img_path).glob("*.*"))
    
    if not img_files or not ref_files:
        print("未找到测试图像文件")
        return
    
    img_path = img_files[0]
    ref_path = ref_files[0]
    
    print(f"使用测试图像: {img_path}")
    print(f"使用参考图像: {ref_path}")
    
    # 读取并编码图像
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    with open(ref_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # 1. 提交任务
    print("\n1. 提交美甲生成任务...")
    response = requests.post(f"{base_url}/edit_nail", data={
        "img": img_b64,
        "ref_img": ref_b64
    })
    
    if response.status_code != 200:
        print(f"提交任务失败: {response.status_code}")
        return
    
    result = response.json()
    if result["status"] != 0:
        print(f"提交任务失败: {result['msg']}")
        return
    
    task_id = result["task_id"]
    print(f"任务已提交，任务ID: {task_id}")
    
    # 2. 轮询进度
    print("\n2. 开始轮询进度...")
    while True:
        response = requests.get(f"{base_url}/get_progress/{task_id}")
        
        if response.status_code != 200:
            print(f"获取进度失败: {response.status_code}")
            break
        
        result = response.json()
        if result["status"] != 0:
            print(f"获取进度失败: {result['msg']}")
            break
        
        progress = result["progress"]
        message = result["message"]
        is_completed = result["is_completed"]
        
        print(f"进度: {progress:.1%} - {message}")
        
        if is_completed:
            print("任务完成!")
            if "result" in result and result["result"]:
                print("已获取到生成结果")
                # 保存结果图像
                save_result_image(result["result"], f"test_result_{task_id}.png")
            break
        
        time.sleep(1)  # 每秒查询一次
    
    # 3. 清理任务
    print("\n3. 清理任务...")
    response = requests.get(f"{base_url}/cleanup_task/{task_id}")
    if response.status_code == 200:
        result = response.json()
        print(f"清理结果: {result['msg']}")
    else:
        print("清理任务失败")

def create_test_images():
    """创建测试图像"""
    import cv2
    import numpy as np
    
    # 创建测试目录
    os.makedirs("data/test_images", exist_ok=True)
    os.makedirs("data/reference", exist_ok=True)
    
    # 创建示例手部图像（白色背景，中心有矩形模拟指甲）
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_img, (200, 200), (312, 312), (200, 200, 200), -1)
    cv2.imwrite("data/test_images/test_hand.jpg", test_img)
    
    # 创建示例参考色块（红色）
    ref_img = np.ones((100, 100, 3), dtype=np.uint8) * [0, 0, 255]  # BGR红色
    cv2.imwrite("data/reference/test_red.jpg", ref_img)
    
    print("测试图像已创建")

def save_result_image(result_b64, filename):
    """保存结果图像"""
    try:
        result_data = base64.b64decode(result_b64)
        with open(filename, "wb") as f:
            f.write(result_data)
        print(f"结果图像已保存到: {filename}")
    except Exception as e:
        print(f"保存结果图像失败: {e}")

def test_multiple_tasks():
    """测试多个并发任务"""
    print("\n=== 测试多个并发任务 ===")
    
    # 这里可以测试多个任务同时处理的情况
    # 由于代码较长，这里只提供框架
    pass

if __name__ == "__main__":
    print("美甲生成进度API测试")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        response = requests.get("http://localhost/get_progress/test", timeout=5)
        print("服务器连接正常")
    except requests.exceptions.RequestException:
        print("无法连接到服务器，请确保服务器正在运行")
        print("启动命令: python editor_image_server.py")
        exit(1)
    
    # 运行测试
    test_progress_api()
    
    print("\n测试完成!") 