import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_conversion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def validate_polygon(polygon, image_shape):
    """验证多边形坐标是否有效"""
    if len(polygon) < 3:
        return False, "多边形至少需要3个点"
    
    height, width = image_shape
    for i, point in enumerate(polygon):
        if len(point) != 2:
            return False, f"点 {i} 坐标格式错误: {point}"
        
        x, y = point
        if not (0 <= x <= width and 0 <= y <= height):
            return False, f"点 {i} 坐标越界: ({x}, {y}), 图像尺寸: {width}x{height}"
    
    return True, "坐标验证通过"

def create_mask_from_polygon(polygon, image_shape):
    """从多边形创建二值掩码，带验证"""
    # 验证坐标
    is_valid, message = validate_polygon(polygon, image_shape)
    if not is_valid:
        raise ValueError(f"多边形坐标无效: {message}")
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32)
    
    # 使用cv2.fillPoly填充多边形
    cv2.fillPoly(mask, [polygon], 255)
    
    return mask

def verify_mask_quality(mask, original_shape, shape_info):
    """验证生成的掩码质量"""
    # 检查掩码是否为空
    if np.sum(mask) == 0:
        return False, "生成的掩码为空"
    
    # 检查掩码面积是否合理（至少占图像的0.01%）
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.sum(mask > 0)
    area_ratio = mask_pixels / total_pixels
    
    if area_ratio < 0.0001:
        return False, f"掩码面积过小: {area_ratio:.4f} (< 0.01%)"
    
    if area_ratio > 0.9:
        return False, f"掩码面积过大: {area_ratio:.4f} (> 90%)"
    
    return True, f"掩码质量验证通过，面积比例: {area_ratio:.4f}"

def find_image_file(image_name, image_dirs):
    """在指定目录中查找图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tiff', '.TIFF']
    
    for image_dir in image_dirs:
        for ext in image_extensions:
            image_path = os.path.join(image_dir, image_name + ext)
            if os.path.exists(image_path):
                return image_path
    return None

def validate_labelme_data(data):
    """验证LabelMe标注数据的完整性"""
    required_fields = ['imageHeight', 'imageWidth', 'shapes']
    for field in required_fields:
        if field not in data:
            return False, f"缺少必需字段: {field}"
    
    if not isinstance(data['shapes'], list):
        return False, "shapes字段必须是列表"
    
    if data['imageHeight'] <= 0 or data['imageWidth'] <= 0:
        return False, f"图像尺寸无效: {data['imageWidth']}x{data['imageHeight']}"
    
    return True, "标注数据验证通过"

def convert_labelme_to_masks(labelme_dir, output_dir, image_dirs=None):
    """将 LabelMe 标注转换为掩码，带完整验证"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认图像目录列表 - 只扫描标注文件所在目录
    if image_dirs is None:
        image_dirs = [labelme_dir]  # 只在标注文件所在目录查找图像
    
    # 获取所有 JSON 文件
    json_files = list(Path(labelme_dir).glob('*.json'))
    
    logging.info(f"找到 {len(json_files)} 个标注文件")
    logging.info(f"将在以下目录中查找图像文件: {image_dirs}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for json_file in tqdm(json_files, desc="Converting annotations"):
        try:
            # 读取 JSON 文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证标注数据
            is_valid, message = validate_labelme_data(data)
            if not is_valid:
                logging.error(f"{json_file.name}: {message}")
                error_count += 1
                continue
            
            # 获取图片尺寸
            image_shape = (data['imageHeight'], data['imageWidth'])
            
            # 查找对应的图像文件
            image_name = json_file.stem
            image_path = find_image_file(image_name, image_dirs)
            
            if image_path is None:
                logging.warning(f"{json_file.name}: 未找到对应的图像文件，跳过")
                skip_count += 1
                continue
            
            # 创建掩码
            mask = np.zeros(image_shape, dtype=np.uint8)
            
            # 处理每个标注
            nail_count = 0
            valid_shapes = []
            
            for i, shape in enumerate(data['shapes']):
                if shape['label'] == 'nail':
                    nail_count += 1
                    
                    # 验证标注格式
                    if 'points' not in shape:
                        logging.warning(f"{json_file.name}: 标注 {i} 缺少points字段")
                        continue
                    
                    polygon = shape['points']
                    
                    try:
                        # 创建单个标注的掩码
                        single_mask = create_mask_from_polygon(polygon, image_shape)
                        
                        # 合并到总掩码（不进行面积检查）
                        mask = cv2.bitwise_or(mask, single_mask)
                        valid_shapes.append(i)
                        
                    except Exception as e:
                        logging.error(f"{json_file.name}: 标注 {i} 处理失败: {str(e)}")
                        continue
            
            if nail_count > 0:
                # 保存掩码（只要有指甲标注就保存）
                output_path = os.path.join(output_dir, f"{json_file.stem}_mask.png")
                cv2.imwrite(output_path, mask)
                
                logging.info(f"{json_file.name}: 成功处理 {len(valid_shapes)}/{nail_count} 个指甲标注，对应图像: {os.path.basename(image_path)}")
                success_count += 1
            else:
                logging.warning(f"{json_file.name}: 未找到指甲标注，跳过")
                skip_count += 1
                
        except json.JSONDecodeError as e:
            logging.error(f"{json_file.name}: JSON解析错误: {str(e)}")
            error_count += 1
        except Exception as e:
            logging.error(f"{json_file.name}: 处理失败: {str(e)}")
            error_count += 1
    
    logging.info(f"\n转换完成！")
    logging.info(f"成功处理: {success_count} 个文件")
    logging.info(f"跳过处理: {skip_count} 个文件")
    logging.info(f"错误处理: {error_count} 个文件")

def main():
    # 设置路径
    labelme_dir = "data/images"       # LabelMe 标注文件目录（包含.json文件）
    output_dir = "data/masks"         # 输出掩码目录
    image_dirs = ["data/images"]      # 图像文件目录 - 只扫描标注文件所在目录
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入目录
    if not os.path.exists(labelme_dir):
        logging.error(f"标注目录不存在: {labelme_dir}")
        return
    
    # 转换标注
    convert_labelme_to_masks(labelme_dir, output_dir, image_dirs)

if __name__ == "__main__":
    main() 