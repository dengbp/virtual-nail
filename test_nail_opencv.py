import cv2
import os
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
import numpy as np

input_dir = "data/test_images"
ref_img_path = "data/reference/reference.jpg"
output_mask_dir = "data/output/masks"
output_color_dir = "data/output/color_transfer"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_color_dir, exist_ok=True)

nail = NailSDXLInpaintOpenCV()

def process_one(img_file):
    stem = img_file.stem
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"无法读取图片: {img_file}")
        return

    mask_path = f"{output_mask_dir}/{stem}_mask.png"
    color_transfer_path = f"{output_color_dir}/{stem}.png"

    print(f"处理: {img_file}")
    # 1. 生成掩码
    nail.save_mask(img, mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. OpenCV颜色迁移（tile平铺方案）
    nail.generate_color_transfer(img, ref_img_path, color_transfer_path)
    print(f"完成: {color_transfer_path}")

if __name__ == "__main__":
    for img_file in Path(input_dir).glob("*.*"):
        process_one(img_file) 