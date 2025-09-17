import cv2
import os
from pathlib import Path
from nail_color_transfer import U2NetMasker # 确保这个import路径是正确的
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_masks_for_input_images():
    """
    为 data/test_images 目录下的所有图片生成指甲掩码，
    并保存到 data/test_masks 目录，使用与服务器版本一致的命名格式。
    """
    input_dir = "data/test_images"
    output_dir = "data/test_masks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("正在初始化U2Net掩码生成器...")
    try:
        # 假设U2NetMasker的初始化不需要特殊参数，或使用默认参数
        masker = U2NetMasker()
        logger.info("掩码生成器初始化完成。")
    except Exception as e:
        logger.error(f"初始化U2NetMasker失败: {e}", exc_info=True)
        return

    image_files = sorted(list(Path(input_dir).glob("*.*")))
    if not image_files:
        logger.warning(f"在 '{input_dir}' 目录下没有找到任何图片。")
        return

    logger.info(f"找到 {len(image_files)} 张图片，开始生成掩码...")

    for img_path in image_files:
        stem = img_path.stem
        # 命名格式与服务器版本和测试脚本对齐
        output_mask_name = f"{stem}_mask_input_mask.png"
        output_mask_path = Path(output_dir) / output_mask_name

        logger.info(f"正在处理: {img_path.name} -> {output_mask_name}")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"  - 无法读取图片 {img_path.name}，跳过。")
                continue

            # 使用U2Net生成掩码 (0-1的numpy数组)
            mask_prob = masker.get_mask(img, str(img_path), disable_cache=True) 
            
            # 转换为0-255的灰度图
            mask_uint8 = (mask_prob * 255).astype(np.uint8)

            cv2.imwrite(str(output_mask_path), mask_uint8)
            logger.info(f"  - 成功：掩码已保存到 {output_mask_path}")
        
        except Exception as e:
            logger.error(f"  - 处理 {img_path.name} 时发生错误: {e}", exc_info=True)

    logger.info("\n所有掩码生成完毕！")

if __name__ == "__main__":
    generate_masks_for_input_images() 