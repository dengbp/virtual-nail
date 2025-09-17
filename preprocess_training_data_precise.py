import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_training_data_precise.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def calculate_precise_size(h, w, max_long_edge=1024):
    """
    计算精确缩放尺寸，确保与推理时完全一致
    添加8的倍数对齐，有利于AI模型处理
    """
    if max(h, w) <= max_long_edge:
        # 即使不缩放，也确保尺寸是8的倍数
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        return new_h, new_w, False
    
    # 计算缩放比例（与推理时完全一致）
    scale = max_long_edge / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 确保尺寸是8的倍数（有利于AI模型处理）
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8
    
    return new_h, new_w, True

def resize_image_precise(img, target_size):
    """
    精确图像缩放，与推理时完全一致
    使用INTER_LANCZOS4确保最高质量
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

def resize_mask_precise_only(mask, target_size):
    """
    掩码精确缩放，只进行等比缩放，不进行任何额外处理
    保持原始标注的精确性，使用INTER_NEAREST保持边缘清晰
    """
    # 只进行等比缩放，使用INTER_NEAREST保持边缘清晰
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # 不进行任何形态学操作、噪声去除或边缘处理
    # 保持原始标注的完整性
    return resized_mask

def verify_precise_alignment_only(image, mask):
    """
    精确验证图像和掩码的对齐程度，不进行质量修改
    添加更严格的验证标准
    """
    if image.shape[:2] != mask.shape[:2]:
        return False, f"尺寸不匹配: 图像{image.shape[:2]} vs 掩码{mask.shape[:2]}"
    
    # 检查掩码是否为空
    mask_area = np.sum(mask > 0)
    if mask_area == 0:
        return False, "掩码为空"
    
    # 检查覆盖率（使用更宽松的范围，因为我们要保持原始标注）
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = mask_area / total_pixels
    
    # 只检查极端异常情况，不强制修改
    if coverage < 0.0001 or coverage > 0.8:
        return False, f"掩码覆盖率异常: {coverage:.4f}"
    
    # 检查尺寸是否为8的倍数（有利于AI模型处理）
    h, w = mask.shape[:2]
    if h % 8 != 0 or w % 8 != 0:
        return False, f"尺寸不是8的倍数: {w}x{h}"
    
    return True, f"对齐验证通过，覆盖率: {coverage:.4f}, 尺寸: {w}x{h}"

def preprocess_single_pair_precise(image_path, mask_path, output_image_path, output_mask_path):
    """
    精确版本的单对图像和掩码预处理
    只进行等比缩放，不进行任何可能改变原始标注的处理
    确保与主流程推理时完全一致
    """
    try:
        # 1. 读取原图像和掩码
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return False, f"无法读取图像: {image_path}"
        if mask is None:
            return False, f"无法读取掩码: {mask_path}"
        
        # 2. 记录原始尺寸
        orig_h, orig_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        # 3. 检查尺寸一致性
        if (orig_h, orig_w) != (mask_h, mask_w):
            logging.warning(f"原始尺寸不匹配: 图像{image_path} {orig_w}x{orig_h} vs 掩码{mask_path} {mask_w}x{mask_h}")
            # 只进行简单的尺寸调整，不进行质量处理
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 4. 计算目标尺寸（与推理时完全一致）
        target_h, target_w, needs_resize = calculate_precise_size(orig_h, orig_w, max_long_edge=1024)
        
        if needs_resize:
            # 5. 精确图像缩放
            processed_image = resize_image_precise(image, (target_w, target_h))
            
            # 6. 精确掩码缩放（只等比缩放，不进行任何额外处理）
            processed_mask = resize_mask_precise_only(mask, (target_w, target_h))
            
            logging.info(f"缩放: {orig_w}x{orig_h} -> {target_w}x{target_h}")
        else:
            processed_image = image.copy()
            processed_mask = mask.copy()
            logging.info(f"保持原尺寸: {orig_w}x{orig_h}")
        
        # 7. 验证对齐（不进行质量修改）
        is_aligned, message = verify_precise_alignment_only(processed_image, processed_mask)
        if not is_aligned:
            return False, f"对齐验证失败: {message}"
        
        # 8. 保存处理后的图像和掩码
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        # 使用高质量保存
        cv2.imwrite(str(output_image_path), processed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(output_mask_path), processed_mask)
        
        return True, f"处理成功: {orig_w}x{orig_h} -> {target_w}x{target_h}, {message}"
        
    except Exception as e:
        return False, f"处理失败: {str(e)}"

def find_image_mask_pairs(images_dir, masks_dir):
    """
    查找图像和掩码的配对
    """
    pairs = []
    
    # 获取所有掩码文件
    mask_files = list(Path(masks_dir).glob('*_mask.png'))
    
    for mask_file in mask_files:
        # 从掩码文件名推导图像文件名
        mask_stem = mask_file.stem.replace('_mask', '')
        
        # 查找对应的图像文件
        image_candidates = [
            Path(images_dir) / f"{mask_stem}.jpg",
            Path(images_dir) / f"{mask_stem}.jpeg",
            Path(images_dir) / f"{mask_stem}.png",
            Path(images_dir) / f"{mask_stem}.JPG",
            Path(images_dir) / f"{mask_stem}.JPEG",
            Path(images_dir) / f"{mask_stem}.PNG",
        ]
        
        for image_candidate in image_candidates:
            if image_candidate.exists():
                pairs.append((image_candidate, mask_file))
                break
        else:
            logging.warning(f"未找到掩码 {mask_file.name} 对应的图像文件")
    
    return pairs

def preprocess_training_dataset_precise(images_dir, masks_dir, output_images_dir, output_masks_dir):
    """
    精确版本的训练数据集预处理
    确保与推理时完全一致，不引入任何额外处理
    """
    logging.info("开始精确预处理训练数据集...")
    
    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # 查找图像和掩码配对
    pairs = find_image_mask_pairs(images_dir, masks_dir)
    logging.info(f"找到 {len(pairs)} 对图像-掩码文件")
    
    if len(pairs) == 0:
        logging.error("未找到任何图像-掩码配对")
        return
    
    # 统计信息
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # 处理每对文件
    for image_path, mask_path in tqdm(pairs, desc="精确预处理训练数据"):
        # 构建输出路径
        image_name = image_path.stem
        output_image_path = Path(output_images_dir) / f"{image_name}.jpg"
        output_mask_path = Path(output_masks_dir) / f"{image_name}_mask.png"
        
        # 检查是否已经处理过
        if output_image_path.exists() and output_mask_path.exists():
            skipped_count += 1
            continue
        
        # 预处理
        success, message = preprocess_single_pair_precise(
            image_path, mask_path, 
            output_image_path, output_mask_path
        )
        
        if success:
            success_count += 1
            logging.debug(f"✓ {image_name}: {message}")
        else:
            error_count += 1
            logging.error(f"✗ {image_name}: {message}")
    
    # 输出统计结果
    logging.info(f"\n精确预处理完成！")
    logging.info(f"成功处理: {success_count} 个文件")
    logging.info(f"处理失败: {error_count} 个文件")
    logging.info(f"跳过处理: {skipped_count} 个文件")
    logging.info(f"输出目录:")
    logging.info(f"  图像: {output_images_dir}")
    logging.info(f"  掩码: {output_masks_dir}")

def verify_preprocessed_data_precise(images_dir, masks_dir, sample_size=50):
    """
    精确版本的数据质量验证
    """
    logging.info("开始验证精确预处理后的数据质量...")
    
    # 随机选择样本进行验证
    import random
    image_files = list(Path(images_dir).glob('*.jpg'))
    mask_files = list(Path(masks_dir).glob('*_mask.png'))
    
    if len(image_files) == 0 or len(mask_files) == 0:
        logging.error("未找到预处理后的文件")
        return
    
    # 随机采样
    sample_images = random.sample(image_files, min(sample_size, len(image_files)))
    
    alignment_errors = 0
    coverage_stats = []
    size_stats = []
    
    for image_file in sample_images:
        image_name = image_file.stem
        mask_file = Path(masks_dir) / f"{image_name}_mask.png"
        
        if not mask_file.exists():
            logging.warning(f"掩码文件不存在: {mask_file}")
            alignment_errors += 1
            continue
        
        # 读取文件
        image = cv2.imread(str(image_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            logging.warning(f"无法读取文件: {image_file} 或 {mask_file}")
            alignment_errors += 1
            continue
        
        # 验证对齐
        is_aligned, message = verify_precise_alignment_only(image, mask)
        if not is_aligned:
            logging.warning(f"对齐验证失败 {image_name}: {message}")
            alignment_errors += 1
        
        # 统计覆盖率
        mask_area = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = mask_area / total_pixels
        coverage_stats.append(coverage)
        
        # 统计尺寸
        size_stats.append((mask.shape[1], mask.shape[0]))  # (width, height)
    
    # 输出验证结果
    logging.info(f"\n精确数据质量验证完成！")
    logging.info(f"验证样本数: {len(sample_images)}")
    logging.info(f"对齐错误: {alignment_errors}")
    logging.info(f"对齐成功率: {(len(sample_images) - alignment_errors) / len(sample_images) * 100:.1f}%")
    
    if coverage_stats:
        avg_coverage = np.mean(coverage_stats)
        min_coverage = np.min(coverage_stats)
        max_coverage = np.max(coverage_stats)
        logging.info(f"掩码覆盖率统计:")
        logging.info(f"  平均: {avg_coverage:.4f}")
        logging.info(f"  最小: {min_coverage:.4f}")
        logging.info(f"  最大: {max_coverage:.4f}")
    
    if size_stats:
        unique_sizes = set(size_stats)
        logging.info(f"图像尺寸统计:")
        logging.info(f"  唯一尺寸数: {len(unique_sizes)}")
        logging.info(f"  尺寸列表: {sorted(unique_sizes)}")
        
        # 检查是否都是8的倍数
        non_multiple_8 = [(w, h) for w, h in unique_sizes if w % 8 != 0 or h % 8 != 0]
        if non_multiple_8:
            logging.warning(f"发现非8倍数尺寸: {non_multiple_8}")
        else:
            logging.info("所有尺寸都是8的倍数 ✓")

def main():
    """
    主函数
    """
    # 设置路径
    images_dir = "data/images"                    # 原始图像目录
    masks_dir = "data/masks"                      # 原始掩码目录
    output_images_dir = "data/training_precise/images"    # 精确预处理后图像目录
    output_masks_dir = "data/training_precise/masks"      # 精确预处理后掩码目录
    
    # 检查输入目录
    if not os.path.exists(images_dir):
        logging.error(f"图像目录不存在: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        logging.error(f"掩码目录不存在: {masks_dir}")
        return
    
    # 精确预处理数据集
    preprocess_training_dataset_precise(images_dir, masks_dir, output_images_dir, output_masks_dir)
    
    # 验证精确预处理质量
    verify_preprocessed_data_precise(output_images_dir, output_masks_dir)

if __name__ == "__main__":
    main() 