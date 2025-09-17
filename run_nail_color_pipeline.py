import os
from generate_masks import main as generate_masks_main
from nail_color_transfer import transfer_nail_color

# 路径参数
IMAGE_DIR = 'data/test_images'  # 强制只用 data/test_images
MASK_DIR = 'data/test_masks'    # 强制只用 data/test_masks
OUTPUT_DIR = 'data/output'
REF_IMG_PATH = 'data/reference/5001.jpg'

def pipeline():
    # 步骤1：批量生成掩码
    print("=== Step 1: 生成灰度掩码 ===")
    generate_masks_main()

    # 步骤2：批量颜色迁移
    print("=== Step 2: 颜色迁移 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in image_files:
        input_img_path = os.path.join(IMAGE_DIR, img_name)
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        mask_path = os.path.join(MASK_DIR, mask_name)
        output_img_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + '_transfer.png')
        if not os.path.exists(mask_path):
            print(f"掩码不存在，跳过: {mask_path}")
            continue
        try:
            transfer_nail_color(
                input_img_path=input_img_path,
                ref_img_path=REF_IMG_PATH,
                nail_mask_path=mask_path,
                output_img_path=output_img_path,
                target_similarity=0.95
            )
            print(f"已完成: {output_img_path}")
        except Exception as e:
            print(f"处理失败: {img_name}，原因: {e}")

if __name__ == "__main__":
    pipeline() 