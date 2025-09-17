import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import math
from dataclasses import dataclass
from enum import Enum
import os

class NailShape(Enum):
    """指甲形状枚举"""
    ROUND = "round"           # 圆形
    SQUARE = "square"         # 方形
    OVAL = "oval"            # 椭圆形
    ALMOND = "almond"        # 杏仁形
    STILETTO = "stiletto"    # 尖形
    COFFIN = "coffin"        # 棺材形

class NailAngle(Enum):
    """指甲角度枚举"""
    FRONT = "front"          # 正面
    SIDE_LEFT = "side_left"  # 左侧面
    SIDE_RIGHT = "side_right" # 右侧面
    ANGLE_30 = "angle_30"    # 30度角
    ANGLE_45 = "angle_45"    # 45度角
    ANGLE_60 = "angle_60"    # 60度角

@dataclass
class NailTemplate:
    """指甲模板数据类"""
    shape: NailShape
    angle: NailAngle
    size: Tuple[int, int]  # (width, height)
    mask: np.ndarray       # 掩码图像
    thickness_map: Optional[np.ndarray] = None  # 厚度图
    normal_map: Optional[np.ndarray] = None     # 法线图

class NailTemplateGenerator:
    """指甲模板生成器"""
    
    def __init__(self, base_size: Tuple[int, int] = (256, 256)):
        self.base_size = base_size
        self.templates: Dict[str, NailTemplate] = {}
    
    def generate_round_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成圆形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建圆形
        radius = min(size) // 3
        cv2.circle(mask, center, radius, 255, -1)
        
        # 添加指甲形状的细节
        # 指甲根部稍微扁平
        cv2.ellipse(mask, center, (radius, radius * 0.8), 0, 0, 180, 255, -1)
        
        return mask
    
    def generate_square_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成方形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建方形
        width = min(size) // 2
        height = width * 0.8
        x1 = center[0] - width // 2
        y1 = center[1] - height // 2
        x2 = center[0] + width // 2
        y2 = center[1] + height // 2
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 稍微圆角
        cv2.ellipse(mask, (x1, y1), (10, 10), 0, 0, 90, 255, -1)
        cv2.ellipse(mask, (x2, y1), (10, 10), 0, 90, 180, 255, -1)
        cv2.ellipse(mask, (x2, y2), (10, 10), 0, 180, 270, 255, -1)
        cv2.ellipse(mask, (x1, y2), (10, 10), 0, 270, 360, 255, -1)
        
        return mask
    
    def generate_oval_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成椭圆形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建椭圆形
        width = min(size) // 2
        height = width * 0.6
        cv2.ellipse(mask, center, (width // 2, height // 2), 0, 0, 360, 255, -1)
        
        return mask
    
    def generate_almond_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成杏仁形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建杏仁形（椭圆形 + 尖角）
        width = min(size) // 2
        height = width * 0.8
        
        # 基础椭圆形
        cv2.ellipse(mask, center, (width // 2, height // 2), 0, 0, 360, 255, -1)
        
        # 添加尖角
        points = np.array([
            [center[0], center[1] - height // 2],
            [center[0] - width // 4, center[1] - height // 2 - 10],
            [center[0] + width // 4, center[1] - height // 2 - 10]
        ], np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def generate_stiletto_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成尖形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建尖形
        width = min(size) // 3
        height = width * 1.5
        
        points = np.array([
            [center[0], center[1] - height // 2],  # 尖端
            [center[0] - width // 2, center[1] + height // 2],  # 左下
            [center[0] + width // 2, center[1] + height // 2]   # 右下
        ], np.int32)
        
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def generate_coffin_nail_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """生成棺材形指甲掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        
        # 创建棺材形（方形 + 尖角）
        width = min(size) // 2
        height = width * 1.2
        
        # 基础方形
        x1 = center[0] - width // 2
        y1 = center[1] - height // 2
        x2 = center[0] + width // 2
        y2 = center[1] + height // 2
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 添加尖角
        points = np.array([
            [center[0], center[1] - height // 2],
            [center[0] - width // 4, center[1] - height // 2 - 15],
            [center[0] + width // 4, center[1] - height // 2 - 15]
        ], np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def apply_angle_transformation(self, mask: np.ndarray, angle: NailAngle) -> np.ndarray:
        """应用角度变换"""
        h, w = mask.shape
        center = (w // 2, h // 2)
        
        if angle == NailAngle.FRONT:
            return mask
        elif angle == NailAngle.SIDE_LEFT:
            # 左侧视角
            matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            matrix[0, 2] += 20  # 向右平移
            return cv2.warpAffine(mask, matrix, (w, h))
        elif angle == NailAngle.SIDE_RIGHT:
            # 右侧视角
            matrix = cv2.getRotationMatrix2D(center, -15, 1.0)
            matrix[0, 2] -= 20  # 向左平移
            return cv2.warpAffine(mask, matrix, (w, h))
        elif angle == NailAngle.ANGLE_30:
            matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
            return cv2.warpAffine(mask, matrix, (w, h))
        elif angle == NailAngle.ANGLE_45:
            matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
            return cv2.warpAffine(mask, matrix, (w, h))
        elif angle == NailAngle.ANGLE_60:
            matrix = cv2.getRotationMatrix2D(center, 60, 1.0)
            return cv2.warpAffine(mask, matrix, (w, h))
        
        return mask
    
    def generate_thickness_map(self, mask: np.ndarray) -> np.ndarray:
        """生成厚度图"""
        thickness_map = np.zeros_like(mask, dtype=np.float32)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 归一化到0-1
        max_dist = np.max(dist_transform)
        if max_dist > 0:
            thickness_map = dist_transform / max_dist
        
        return thickness_map
    
    def generate_normal_map(self, mask: np.ndarray, thickness_map: np.ndarray) -> np.ndarray:
        """生成法线图"""
        h, w = mask.shape
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        
        # 从厚度图计算法线
        for y in range(1, h-1):
            for x in range(1, w-1):
                if mask[y, x] > 0:
                    # 计算梯度
                    dx = (thickness_map[y, x+1] - thickness_map[y, x-1]) / 2.0
                    dy = (thickness_map[y+1, x] - thickness_map[y-1, x]) / 2.0
                    
                    # 构造法向量
                    normal = np.array([-dx, -dy, 1.0])
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    
                    normal_map[y, x] = normal
        
        return normal_map
    
    def generate_template(self, shape: NailShape, angle: NailAngle, 
                         size: Tuple[int, int] = None) -> NailTemplate:
        """生成完整的指甲模板"""
        if size is None:
            size = self.base_size
        
        # 生成基础掩码
        if shape == NailShape.ROUND:
            mask = self.generate_round_nail_mask(size)
        elif shape == NailShape.SQUARE:
            mask = self.generate_square_nail_mask(size)
        elif shape == NailShape.OVAL:
            mask = self.generate_oval_nail_mask(size)
        elif shape == NailShape.ALMOND:
            mask = self.generate_almond_nail_mask(size)
        elif shape == NailShape.STILETTO:
            mask = self.generate_stiletto_nail_mask(size)
        elif shape == NailShape.COFFIN:
            mask = self.generate_coffin_nail_mask(size)
        else:
            mask = self.generate_round_nail_mask(size)
        
        # 应用角度变换
        mask = self.apply_angle_transformation(mask, angle)
        
        # 生成厚度图
        thickness_map = self.generate_thickness_map(mask)
        
        # 生成法线图
        normal_map = self.generate_normal_map(mask, thickness_map)
        
        # 创建模板
        template = NailTemplate(
            shape=shape,
            angle=angle,
            size=size,
            mask=mask,
            thickness_map=thickness_map,
            normal_map=normal_map
        )
        
        return template
    
    def generate_all_templates(self) -> Dict[str, NailTemplate]:
        """生成所有形状和角度的模板"""
        shapes = list(NailShape)
        angles = list(NailAngle)
        
        for shape in shapes:
            for angle in angles:
                template = self.generate_template(shape, angle)
                key = f"{shape.value}_{angle.value}"
                self.templates[key] = template
        
        return self.templates
    
    def save_templates(self, output_dir: str = "nail_templates"):
        """保存所有模板到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for key, template in self.templates.items():
            # 保存掩码
            mask_path = os.path.join(output_dir, f"{key}_mask.png")
            cv2.imwrite(mask_path, template.mask)
            
            # 保存厚度图
            if template.thickness_map is not None:
                thickness_path = os.path.join(output_dir, f"{key}_thickness.png")
                thickness_img = (template.thickness_map * 255).astype(np.uint8)
                cv2.imwrite(thickness_path, thickness_img)
            
            # 保存法线图
            if template.normal_map is not None:
                normal_path = os.path.join(output_dir, f"{key}_normal.png")
                normal_img = ((template.normal_map + 1) * 127.5).astype(np.uint8)
                cv2.imwrite(normal_path, normal_img)
    
    def load_template(self, shape: NailShape, angle: NailAngle) -> Optional[NailTemplate]:
        """加载指定模板"""
        key = f"{shape.value}_{angle.value}"
        return self.templates.get(key)

# 使用示例
def create_nail_template_library():
    """创建指甲模板库"""
    generator = NailTemplateGenerator()
    
    # 生成所有模板
    templates = generator.generate_all_templates()
    
    # 保存到文件
    generator.save_templates()
    
    print(f"生成了 {len(templates)} 个指甲模板")
    for key in templates.keys():
        print(f"- {key}")
    
    return generator

if __name__ == "__main__":
    # 创建模板库
    generator = create_nail_template_library()
    
    # 测试特定模板
    template = generator.load_template(NailShape.ALMOND, NailAngle.ANGLE_45)
    if template:
        print(f"加载模板: {template.shape.value}_{template.angle.value}")
        print(f"尺寸: {template.size}")
        print(f"掩码形状: {template.mask.shape}") 