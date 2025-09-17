import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nail_color_transfer import transfer_nail_color, calculate_color_similarity
import logging
import os

def create_comparison_image(input_img, ref_img, result_img, mask):
    """
    创建对比图，显示原图、参考图、结果图的对比
    
    Args:
        input_img: 输入图像 (RGB)
        ref_img: 参考图像 (RGB)
        result_img: 结果图像 (RGB)
        mask: 掩码 (bool)
    
    Returns:
        PIL.Image: 对比图
    """
    # 调整参考图大小以匹配输入图
    ref_img = cv2.resize(ref_img, (input_img.shape[1], input_img.shape[0]))
    
    # 创建掩码可视化
    mask_vis = np.zeros_like(input_img)
    mask_vis[mask] = [255, 255, 255]
    mask_vis = cv2.addWeighted(input_img, 0.7, mask_vis, 0.3, 0)
    
    # 创建结果图的掩码可视化
    result_mask_vis = np.zeros_like(result_img)
    result_mask_vis[mask] = [255, 255, 255]
    result_mask_vis = cv2.addWeighted(result_img, 0.7, result_mask_vis, 0.3, 0)
    
    # 水平拼接图像
    comparison = np.hstack([
        input_img,          # 原图
        mask_vis,           # 原图+掩码
        ref_img,            # 参考图
        result_img,         # 结果图
        result_mask_vis     # 结果图+掩码
    ])
    
    return Image.fromarray(comparison)

def evaluate_color_transfer(input_img_path, ref_img_path, mask_path, output_dir="test_results"):
    """
    测试颜色迁移效果
    
    Args:
        input_img_path: 输入图像路径
        ref_img_path: 参考图像路径
        mask_path: 掩码路径
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置输出路径
        base_name = os.path.splitext(os.path.basename(input_img_path))[0]
        result_path = os.path.join(output_dir, f"{base_name}_result.png")
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
        
        # 读取图像
        input_img = np.array(Image.open(input_img_path).convert('RGB'))
        ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 128)
        
        # 执行颜色迁移
        print(f"\n处理图像: {input_img_path}")
        print(f"参考图像: {ref_img_path}")
        print(f"掩码图像: {mask_path}")
        
        transfer_nail_color(
            input_img_path=input_img_path,
            ref_img_path=ref_img_path,
            nail_mask_path=mask_path,
            output_img_path=result_path,
            target_similarity=0.95
        )
        
        # 读取结果图像
        result_img = np.array(Image.open(result_path).convert('RGB'))
        
        # 创建对比图
        comparison = create_comparison_image(input_img, ref_img, result_img, mask)
        comparison.save(comparison_path)
        
        # 计算评估指标
        metrics = {
            "输入图像与参考图的相似度": calculate_color_similarity(input_img, ref_img, mask),
            "结果图像与参考图的相似度": calculate_color_similarity(result_img, ref_img, mask),
            "输入图像与结果图的相似度": calculate_color_similarity(input_img, result_img, mask)
        }
        
        # 保存评估指标
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("颜色迁移效果评估\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"输入图像: {input_img_path}\n")
            f.write(f"参考图像: {ref_img_path}\n")
            f.write(f"掩码图像: {mask_path}\n\n")
            f.write("相似度评估:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.3f}\n")
        
        # 打印评估结果
        print("\n评估结果:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        print("\n输出文件:")
        print(f"结果图像: {result_path}")
        print(f"对比图: {comparison_path}")
        print(f"评估指标: {metrics_path}")
        
        # 显示对比图
        plt.figure(figsize=(20, 4))
        plt.imshow(np.array(comparison))
        plt.axis('off')
        plt.title("从左到右: 原图 | 原图+掩码 | 参考图 | 结果图 | 结果图+掩码")
        plt.show()
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试单张图片
    evaluate_color_transfer(
        input_img_path="data/images/nail.jpg",
        ref_img_path="data/reference/colorblock.jpg",
        mask_path="data/masks/nail_mask.png"
    ) 