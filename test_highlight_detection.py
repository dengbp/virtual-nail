#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_highlight_detection.py

功能：批量测试高光检测和掩码生成系统，对data/test_images下所有图片分别输出掩码和高光检测结果。
"""

import cv2
import numpy as np
from pathlib import Path
from color_nail_highlight_shader import (
    add_nail_highlight_with_adaptive_detection
)
from nail_color_transfer import U2NetMasker

def test_highlight_detection_batch():
    """批量测试高光检测系统"""
    input_dir = Path("data/test_images")
    output_dir = Path("data/output/highlight_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    if not image_files:
        print(f"❌ 未找到图片: {input_dir}")
        return
    print(f"共找到{len(image_files)}张图片，开始批量处理...")
    
    masker = U2NetMasker()
    for img_path in image_files:
        print("=" * 60)
        print(f"处理: {img_path.name}")
        input_img = cv2.imread(str(img_path))
        if input_img is None:
            print(f"❌ 无法读取图像: {img_path}")
            continue
        print(f"图像尺寸: {input_img.shape}")
        # 生成掩码
        try:
            nail_mask = masker.get_mask(input_img, str(img_path), disable_cache=True)
            # 严格二值化掩码后再保存
            if nail_mask.dtype != np.uint8:
                nail_mask = nail_mask.astype(np.uint8)
            _, nail_mask_bin = cv2.threshold(nail_mask, 128, 255, cv2.THRESH_BINARY)
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), nail_mask_bin)
            print(f"掩码已保存: {mask_path}")
        except Exception as e:
            print(f"❌ 指甲掩码生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        # 高光检测
        try:
            highlight_path = output_dir / f"{img_path.stem}_highlight.png"
            result = add_nail_highlight_with_adaptive_detection(
                input_img=input_img,
                output_path=str(highlight_path),
                nail_mask=nail_mask,
                debug_mode=True
            )
            if result is not None:
                print(f"✅ 高光检测完成: {highlight_path}")
            else:
                print("❌ 高光检测失败")
        except Exception as e:
            print(f"❌ 高光检测异常: {e}")
            import traceback
            traceback.print_exc()
    print("\n批量测试完成！请在data/output/highlight_test/下查看所有结果。")

if __name__ == "__main__":
    test_highlight_detection_batch() 