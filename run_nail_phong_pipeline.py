#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 一键运行美甲流水线主入口脚本
# 功能：自动掩码生成、高保真色块迁移、Phong高光、无缝融合、批量处理
# 用法：python run_nail_phong_pipeline.py [--input_dir INPUT_DIR] [--ref_img REF_IMG] [--output_dir OUTPUT_DIR] [--target_color TARGET_COLOR] [--blend_mode BLEND_MODE]

import os
import argparse
import logging
import sys
import sys
import sys
import torch
from nail_phong_pipeline import main as phong_main, TARGET_COLOR as DEFAULT_TARGET_COLOR, BLEND_MODE as DEFAULT_BLEND_MODE, PHONG_PARAMS as DEFAULT_PHONG_PARAMS
from nail_color_transfer import process_directory as color_transfer_process_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="一键运行美甲流水线主入口脚本")
    parser.add_argument("--input_dir", type=str, default="data/test_images", help="输入图片目录，默认为 data/test_images")
    parser.add_argument("--ref_img", type=str, default="data/reference/reference.jpg", help="参考色块/纹理图路径，默认为 data/reference/reference.jpg")
    parser.add_argument("--output_dir", type=str, default="data/output", help="输出目录，默认为 data/output")
    parser.add_argument("--target_color", type=str, default="180,105,255", help="纯色迁移的目标颜色（BGR，逗号分隔），默认为 180,105,255")
    parser.add_argument("--blend_mode", type=str, choices=["seamless", "copy"], default=DEFAULT_BLEND_MODE, help="融合方式，可选 seamless 或 copy，默认为 seamless")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    ref_img = args.ref_img
    output_dir = args.output_dir
    target_color_str = args.target_color
    blend_mode = args.blend_mode

    # 解析目标颜色（BGR）
    try:
        target_color = tuple(map(int, target_color_str.split(",")))
        if len(target_color) != 3:
            raise ValueError("目标颜色必须为三个整数（BGR），以逗号分隔")
    except Exception as e:
        logger.error(f"解析目标颜色失败: {e}，使用默认值 {DEFAULT_TARGET_COLOR}")
        target_color = DEFAULT_TARGET_COLOR

    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录 {input_dir} 不存在，请检查路径。")
        sys.exit(1)

    # 判断参考图是否存在，若存在则调用 nail_color_transfer.py 进行复杂纹理迁移，否则调用 nail_phong_pipeline.py 进行纯色迁移
    if os.path.isfile(ref_img):
        logger.info(f"参考图 {ref_img} 存在，调用 nail_color_transfer.py 进行高保真纹理迁移...")
        # 注意：nail_color_transfer.py 的 process_directory 函数内部会调用 generate_masks.py 自动生成掩码，并输出到 data/test_masks 目录
        color_transfer_process_directory(input_dir=input_dir, ref_img_path=ref_img)
        logger.info("高保真纹理迁移完成，输出在 data/output 目录。")
    else:
        logger.info(f"参考图 {ref_img} 不存在，调用 nail_phong_pipeline.py 进行纯色迁移...")
        # 修改 nail_phong_pipeline.py 中的全局变量，以便使用自定义目标颜色和融合方式
        import nail_phong_pipeline
        nail_phong_pipeline.TARGET_COLOR = target_color
        nail_phong_pipeline.BLEND_MODE = blend_mode
        nail_phong_pipeline.OUTPUT_DIR = output_dir
        phong_main()
        logger.info("纯色迁移完成，输出在 {output_dir} 目录。")

    # 打印 CUDA 信息
    print("\nCUDA 信息:")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数: {torch.cuda.device_count()}")
    print(f"CUDA 设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}") 