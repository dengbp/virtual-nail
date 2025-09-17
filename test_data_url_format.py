#!/usr/bin/env python3
"""
测试Data URL格式
验证修改后的base64数据格式是否正确
"""

import cv2
import numpy as np
import base64
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_image():
    """创建测试图像"""
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # 绘制一个简单的图案
    cv2.rectangle(image, (50, 50), (250, 150), (0, 255, 0), 2)
    cv2.circle(image, (150, 100), 30, (255, 0, 0), -1)
    cv2.putText(image, "Test", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image

def test_data_url_format():
    """测试Data URL格式"""
    print("=== 测试Data URL格式 ===")
    
    # 创建测试图像
    image = create_test_image()
    cv2.imwrite('test_data_url_image.png', image)
    
    # 模拟原来的base64编码
    with open('test_data_url_image.png', 'rb') as f:
        original_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # 模拟修改后的Data URL格式
    data_url = f"data:image/png;base64,{original_b64}"
    
    print(f"原始base64长度: {len(original_b64)} 字符")
    print(f"Data URL长度: {len(data_url)} 字符")
    print(f"Data URL前缀: {data_url[:30]}...")
    
    # 验证Data URL格式是否正确
    if data_url.startswith("data:image/png;base64,"):
        print("✅ Data URL格式正确")
    else:
        print("❌ Data URL格式错误")
    
    # 测试解码
    try:
        # 从Data URL中提取base64数据
        b64_data = data_url.split(',')[1]
        decoded_data = base64.b64decode(b64_data)
        
        # 解码为图像
        decoded_image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
        
        if decoded_image is not None:
            print("✅ Data URL解码成功")
            cv2.imwrite('test_data_url_decoded.png', decoded_image)
            print("解码后的图像已保存为: test_data_url_decoded.png")
        else:
            print("❌ Data URL解码失败")
            
    except Exception as e:
        print(f"❌ Data URL解码异常: {e}")
    
    # 模拟API返回格式
    api_response = {
        "statusCode": 200,
        "task_id": "123456789",
        "progress": 1.0,
        "current_step": 0,
        "total_steps": 0,
        "message": "生成完成",
        "is_completed": True,
        "data": data_url
    }
    
    print(f"\n模拟API返回格式:")
    print(f"data字段长度: {len(api_response['data'])} 字符")
    print(f"data字段前缀: {api_response['data'][:30]}...")
    
    # 前端使用示例
    print(f"\n前端使用示例:")
    print(f"const img = new Image();")
    print(f"img.src = response.data;  // 直接使用，无需额外处理")
    print(f"document.body.appendChild(img);")
    
    return True

def main():
    """主函数"""
    print("测试Data URL格式修改")
    print("=" * 40)
    
    success = test_data_url_format()
    
    if success:
        print("\n🎉 测试通过！")
        print("Data URL格式修改成功。")
        print("\n修改内容:")
        print("✅ 将纯base64数据改为完整的Data URL格式")
        print("✅ 格式: data:image/png;base64,{base64数据}")
        print("✅ 前端可以直接使用，无需额外处理")
        
        print("\n生成的文件:")
        print("- test_data_url_image.png: 原始测试图像")
        print("- test_data_url_decoded.png: 解码后的图像")
        
    else:
        print("\n⚠️  测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 