import cv2
import numpy as np
import os
from pathlib import Path

def create_reference_color_image(output_path="data/test_images/ref_color.png", color=(180, 105, 255), size=(200, 200)):
    """
    创建一个参考色图像
    :param output_path: 输出路径
    :param color: BGR颜色值，默认是紫色 (180, 105, 255)
    :param size: 图像尺寸，默认200x200
    """
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建纯色图像
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    
    # 保存图像
    cv2.imwrite(output_path, image)
    print(f"✅ 参考色图像已创建: {output_path}")
    print(f"   🎨 颜色: BGR{color}")
    print(f"   📏 尺寸: {size[0]}x{size[1]}")
    
    return output_path

def create_multiple_reference_colors():
    """
    创建多个不同颜色的参考图像
    """
    colors = {
        "ref_color.png": (180, 105, 255),      # 紫色
        "ref_red.png": (0, 0, 255),            # 红色
        "ref_blue.png": (255, 0, 0),           # 蓝色
        "ref_green.png": (0, 255, 0),          # 绿色
        "ref_pink.png": (147, 20, 255),        # 粉色
        "ref_orange.png": (0, 165, 255),       # 橙色
        "ref_yellow.png": (0, 255, 255),       # 黄色
        "ref_black.png": (0, 0, 0),            # 黑色
        "ref_white.png": (255, 255, 255),      # 白色
    }
    
    output_dir = "data/test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, color in colors.items():
        output_path = os.path.join(output_dir, filename)
        create_reference_color_image(output_path, color)
    
    print(f"\n🎨 已创建 {len(colors)} 个参考色图像")
    print(f"📁 位置: {output_dir}")

if __name__ == "__main__":
    print("🎨 创建参考色图像")
    print("=" * 50)
    
    # 创建单个参考色图像（默认紫色）
    create_reference_color_image()
    
    print("\n" + "=" * 50)
    print("是否创建多个颜色的参考图像？(y/n): ", end="")
    
    # 注释掉用户输入，直接创建多个颜色
    # user_input = input().lower()
    # if user_input == 'y':
    create_multiple_reference_colors()
    
    print("\n✅ 参考色图像创建完成！")
    print("💡 现在可以运行测试脚本了") 