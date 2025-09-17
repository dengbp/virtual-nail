import time
from pathlib import Path
import os
import torch

# 导入新的AI核心类和处理函数
from nail_sdxl_inpaint_ip_adapter import NailSdxlInpaintIpAdapter
from nail_sdxl_inpaint_purecolor import process_one

def main():
    """
    为 IP-Adapter 版本的原始方案提供独立的测试脚本。
    该脚本会预加载模型，然后循环处理所有测试图片，以提高效率。
    """
    print("--- IP-Adapter 风格迁移测试脚本 ---")
    
    # --- 配置 ---
    input_dir = "data/test_images"
    ref_path = "data/reference/reference.png" # 固定使用此风格参考图
    output_dir = "data/output/final_ip_adapter_test"
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # --- 检查文件 ---
    if not Path(ref_path).exists():
        print(f"错误: 固定的风格参考图 '{ref_path}' 不存在。")
        print("请将一个风格参考图（例如 517.png）放到 'data/reference/' 目录下，并重命名为 'reference.png'。")
        return

    img_files = sorted(list(Path(input_dir).glob("*.*")))
    if not img_files:
        print(f"错误：在 '{input_dir}' 目录中未找到任何图片。")
        return

    # --- 预加载模型 ---
    nail_processor = None
    try:
        print("\n正在预加载 IP-Adapter 模型...")
        start_load_time = time.time()
        nail_processor = NailSdxlInpaintIpAdapter()
        load_time = time.time() - start_load_time
        print(f"模型预加载完成，耗时: {load_time:.2f}秒")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接、模型文件和依赖库。")
        return

    # --- 循环处理 ---
    total_start_time = time.time()
    print(f"\n找到 {len(img_files)} 张图片，开始处理...")

    for img_path in img_files:
        print("-" * 50)
        print(f"处理中: {img_path.name}")
        
        try:
            # 调用处理函数，并传入预加载的模型实例
            process_one(
                img_path=str(img_path),
                ref_path=ref_path,
                nail_processor=nail_processor # <-- 传入预加载的模型
            )
            print(f"处理成功: {img_path.name}")
        except Exception as e:
            print(f"处理失败: {img_path.name}, 错误: {e}")

    # --- 清理与总结 ---
    if nail_processor:
        nail_processor.clean_up()
        print("\nGPU内存已清理。")

    total_time = time.time() - total_start_time
    print("-" * 50)
    print(f"所有图像处理完成，总耗时: {total_time:.2f}秒")
    print(f"最终效果图请查看: {output_dir}") # 提示：process_one仍会按其内部逻辑保存，这里仅作提示

if __name__ == "__main__":
    main() 