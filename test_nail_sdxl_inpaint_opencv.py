import cv2
import os
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
import numpy as np

input_dir = "data/test_images"
ref_img_path = "data/reference/reference.jpg"
output_mask_dir = "data/output/masks"
output_color_dir = "data/output/color_transfer"
output_final_dir = "data/output/final"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_color_dir, exist_ok=True)
os.makedirs(output_final_dir, exist_ok=True)

nail = NailSDXLInpaintOpenCV()

first_debug = True

def process_one(img_file):
    global first_debug
    stem = img_file.stem
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"无法读取图片: {img_file}")
        return

    # 1. 入口统一分辨率
    orig_h, orig_w = img.shape[:2]
    max_side = max(orig_h, orig_w)
    if max_side > 1024:
        scale = 1024 / max_side
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    mask_path = f"{output_mask_dir}/{stem}_mask.png"
    color_transfer_path = f"{output_color_dir}/{stem}.png"
    final_path = f"{output_final_dir}/{stem}_final.png"

    print(f"处理: {img_file}")
    # 2. 生成掩码（与img同分辨率）
    nail.save_mask(img, mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 3. OpenCV颜色迁移（tile平铺方案，分辨率与img一致）
    nail.generate_color_transfer(img, ref_img_path, color_transfer_path)
    color_transfer_img = cv2.imread(color_transfer_path)
    # 4. AI L通道+迁移ab通道融合
    prompt = (
        "ultra realistic nail, natural nail shape, photorealistic, glossy, smooth, with highlights and reflections, "
        "keep the original nail color, do not change color, keep the color from the input image, no extra patterns, no distortion"
    )
    negative_prompt = (
        "color shift, color change, extra pattern, cartoon, painting, distortion, blur, artifacts, watermark, text, logo, red, dark red, blood, stain, spot"
    )
    final = nail.process_with_ai_fusion(
        color_transfer_img=color_transfer_img,
        mask=mask,
        prompt=prompt
    )
    # 只在第一张图片时生成 debug 掩码
    if first_debug:
        nail.process_image(img)
        first_debug = False
    # 保证输出分辨率与原图一致（如需）
    if final.shape[:2] != (orig_h, orig_w):
        final = cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(final_path, final)
    print(f"完成: {final_path}")

if __name__ == "__main__":
    for img_file in Path(input_dir).glob("*.*"):
        process_one(img_file) 