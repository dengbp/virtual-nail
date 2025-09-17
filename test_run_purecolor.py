from pathlib import Path
from nail_sdxl_inpaint_purecolor import process_one
import cv2

# 保持原始版本目录结构
input_dir = "data/test_images"
ref_dir = "data/reference"
output_mask_dir = "data/output/masks"
output_color_dir = "data/output/color_transfer"
output_final_dir = "data/output/final"

if __name__ == "__main__":
    img_files = sorted(list(Path(input_dir).glob("*.*")))
    ref_path = Path("data/reference/reference.jpg")
    if not ref_path.exists():
        print(f"错误: 参考图不存在: {ref_path}")
        exit(1)
    for img_path in img_files:
        print(f"处理: {img_path.name} + {ref_path.name}")
        process_one(img_path, ref_path) 