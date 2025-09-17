#!/usr/bin/env python3
"""
API Data URL格式测试
验证修改后的API接口返回的Data URL格式是否正确
"""

import requests
import base64
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_images():
    """创建测试图像"""
    # 创建原图像
    original_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # 绘制手部轮廓
    hand_contour = np.array([
        [100, 50], [150, 30], [200, 40], [250, 60], [280, 100],
        [290, 150], [280, 200], [250, 250], [200, 280], [150, 290],
        [100, 280], [50, 250], [30, 200], [20, 150], [30, 100],
        [50, 60], [100, 50]
    ], dtype=np.int32)
    
    cv2.fillPoly(original_image, [hand_contour], (240, 220, 200))
    
    # 绘制指甲
    nail_contours = [
        np.array([[80, 80], [120, 70], [140, 90], [120, 110], [80, 100]], dtype=np.int32),
        np.array([[180, 60], [220, 50], [240, 70], [220, 90], [180, 80]], dtype=np.int32),
        np.array([[250, 70], [290, 60], [310, 80], [290, 100], [250, 90]], dtype=np.int32),
    ]
    
    for nail_contour in nail_contours:
        cv2.fillPoly(original_image, [nail_contour], (255, 240, 220))
    
    # 创建参考色图像（纯色）
    reference_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(reference_image, (10, 10), (90, 90), (255, 0, 0), -1)  # 红色
    
    return original_image, reference_image

def encode_image_to_data_url(image, format='jpeg'):
    """将图像编码为Data URL格式"""
    # 编码图像
    if format == 'jpeg':
        _, buffer = cv2.imencode('.jpg', image)
    else:
        _, buffer = cv2.imencode('.png', image)
    
    # 转换为base64
    b64_data = base64.b64encode(buffer).decode('utf-8')
    
    # 返回Data URL格式
    return f"data:image/{format};base64,{b64_data}"

def test_api_data_url_format():
    """测试API Data URL格式"""
    print("=== 测试API Data URL格式 ===")
    
    # 创建测试图像
    original_image, reference_image = create_test_images()
    
    # 保存测试图像
    cv2.imwrite('test_api_original.jpg', original_image)
    cv2.imwrite('test_api_reference.jpg', reference_image)
    
    # 编码为Data URL
    original_data_url = encode_image_to_data_url(original_image, 'jpeg')
    reference_data_url = encode_image_to_data_url(reference_image, 'jpeg')
    
    print(f"原图像Data URL长度: {len(original_data_url)} 字符")
    print(f"参考图像Data URL长度: {len(reference_data_url)} 字符")
    print(f"原图像Data URL前缀: {original_data_url[:30]}...")
    
    # 验证Data URL格式
    if original_data_url.startswith("data:image/jpeg;base64,"):
        print("✅ 原图像Data URL格式正确")
    else:
        print("❌ 原图像Data URL格式错误")
    
    if reference_data_url.startswith("data:image/jpeg;base64,"):
        print("✅ 参考图像Data URL格式正确")
    else:
        print("❌ 参考图像Data URL格式错误")
    
    return original_data_url, reference_data_url

def test_api_integration(original_data_url, reference_data_url):
    """测试API集成"""
    print("\n=== 测试API集成 ===")
    
    base_url = "http://localhost:80"
    
    try:
        # 1. 提交任务
        print("1. 提交美甲生成任务...")
        response = requests.post(f"{base_url}/edit_nail", data={
            'img': original_data_url,
            'ref_img': reference_data_url
        })
        
        if response.status_code != 200:
            print(f"❌ 提交任务失败: {response.status_code}")
            return False
        
        result = response.json()
        if result['statusCode'] != 200:
            print(f"❌ 提交任务失败: {result['message']}")
            return False
        
        task_id = result['task_id']
        print(f"✅ 任务提交成功，任务ID: {task_id}")
        
        # 2. 查询进度
        print("2. 查询任务进度...")
        max_wait_time = 300  # 最多等待5分钟
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = requests.get(f"{base_url}/get_progress/{task_id}")
            
            if response.status_code != 200:
                print(f"❌ 查询进度失败: {response.status_code}")
                return False
            
            result = response.json()
            
            if result['statusCode'] != 200:
                print(f"❌ 查询进度失败: {result['message']}")
                return False
            
            progress = result['progress']
            message = result['message']
            is_completed = result.get('is_completed', False)
            
            print(f"进度: {progress:.1%} - {message}")
            
            if is_completed:
                print("✅ 任务完成！")
                
                # 验证返回的Data URL格式
                data_url = result.get('data', '')
                if data_url:
                    print(f"返回的Data URL长度: {len(data_url)} 字符")
                    print(f"返回的Data URL前缀: {data_url[:30]}...")
                    
                    if data_url.startswith("data:image/png;base64,"):
                        print("✅ 返回的Data URL格式正确")
                        
                        # 测试解码
                        try:
                            b64_data = data_url.split(',')[1]
                            decoded_data = base64.b64decode(b64_data)
                            decoded_image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
                            
                            if decoded_image is not None:
                                cv2.imwrite('test_api_result.png', decoded_image)
                                print("✅ Data URL解码成功，结果已保存为: test_api_result.png")
                                return True
                            else:
                                print("❌ Data URL解码失败")
                                return False
                                
                        except Exception as e:
                            print(f"❌ Data URL解码异常: {e}")
                            return False
                    else:
                        print("❌ 返回的Data URL格式错误")
                        return False
                else:
                    print("❌ 未返回图像数据")
                    return False
            
            time.sleep(2)  # 等待2秒后再次查询
        
        print("❌ 任务超时")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        return False

def main():
    """主函数"""
    print("API Data URL格式测试")
    print("=" * 50)
    
    # 测试Data URL格式
    original_data_url, reference_data_url = test_api_data_url_format()
    
    # 测试API集成
    success = test_api_integration(original_data_url, reference_data_url)
    
    print("\n=== 测试结果 ===")
    if success:
        print("🎉 所有测试通过！")
        print("API Data URL格式修改成功。")
        print("\n修改验证:")
        print("✅ 前端可以直接使用返回的Data URL")
        print("✅ 无需手动拼接Data URL前缀")
        print("✅ 图像数据格式标准化")
        print("✅ API集成流程正常")
        
        print("\n生成的文件:")
        print("- test_api_original.jpg: 测试原图像")
        print("- test_api_reference.jpg: 测试参考图像")
        print("- test_api_result.png: API返回的结果图像")
        
    else:
        print("⚠️  测试失败，请检查错误信息")
        print("\n可能的原因:")
        print("- 服务器未启动")
        print("- 网络连接问题")
        print("- API接口错误")

if __name__ == "__main__":
    main() 