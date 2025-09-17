import os
import logging
import sys
import sys
from tqdm import tqdm
from PIL import Image
from nail_dalle2_inpaint import NailDalle3Editor
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    # 写死目标原图像、掩码图像及输出目录
    img_dir = "data/test_images"
    mask_dir = "data/test_masks"
    out_dir = "data/output/dalle2"
    img_name = "11111.jpg"
    mask_name = "11111_mask.png"
    prompt = (
        "Paint the fingernails in the masked area with a pure, deep green nail polish. "
        "The color should be rich, dark green, glossy, and even. Only change the nails, keep the skin and background unchanged. "
        "Do not add any patterns, text, or reflections of the environment. The result should look like a professional manicure, with clean edges and natural highlights."
    )

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 初始化 NailDalle3Editor 实例
    editor = NailDalle3Editor()

    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)
    fixed_mask_path = os.path.join(mask_dir, os.path.splitext(mask_name)[0] + '_fixed.png')
    # 修正掩码为黑底白指甲
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(f"掩码文件 {mask_path} 不存在，退出。")
        return
    fixed_mask = (mask > 128).astype('uint8') * 255
    cv2.imwrite(fixed_mask_path, fixed_mask)

    try:
        # 调用 editor.process_image 处理图片，传入自定义 prompt（如果提供）
        result_img = editor.process_image(img_path, mask_path=fixed_mask_path, prompt=prompt)
        out_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + "_dalle2_nail.png")
        result_img.save(out_path)
        logger.info(f"处理完成，结果保存至 {out_path}。")
    except Exception as e:
        logger.error(f"处理图片 {img_name} 时出错: {str(e)}")
        return

    logger.info(f"全部处理完成，结果已保存到 {out_dir}")

if __name__ == '__main__':
    main() 