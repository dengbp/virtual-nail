import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import math
from dataclasses import dataclass
from enum import Enum

class MaterialType(Enum):
    """材质类型枚举"""
    GLOSSY = "glossy"           # 光泽材质
    MATTE = "matte"            # 哑光材质
    SEMI_TRANSPARENT = "semi_transparent"  # 半透明材质
    METALLIC = "metallic"      # 金属材质

@dataclass
class LightSource:
    """光源定义"""
    position: np.ndarray  # 光源位置 [x, y, z]
    intensity: float      # 光源强度
    color: np.ndarray     # 光源颜色 [r, g, b]
    light_type: str       # 光源类型: "point", "directional", "area"
    falloff: float = 1.0  # 衰减系数

@dataclass
class MaterialProperties:
    """材质属性"""
    albedo: np.ndarray        # 基础颜色 [r, g, b]
    roughness: float          # 粗糙度 (0-1)
    metallic: float           # 金属度 (0-1)
    ior: float               # 折射率
    transmission: float       # 透射率 (0-1)
    thickness: float          # 厚度 (mm)
    subsurface_scattering: float = 0.0  # 次表面散射强度

class PhysicalLightingSystem:
    """基于物理的光照系统"""
    
    def __init__(self):
        self.lights: List[LightSource] = []
        self.ambient_light = np.array([0.1, 0.1, 0.1])  # 环境光
        self.camera_position = np.array([0, 0, 5])
        
    def add_light(self, light: LightSource):
        """添加光源"""
        self.lights.append(light)
    
    def calculate_normal_map(self, height_map: np.ndarray) -> np.ndarray:
        """从高度图计算法线图"""
        h, w = height_map.shape
        normal_map = np.zeros((h, w, 3))
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                # Sobel算子计算梯度
                dx = (height_map[y, x+1] - height_map[y, x-1]) / 2.0
                dy = (height_map[y+1, x] - height_map[y-1, x]) / 2.0
                
                # 构造法向量
                normal = np.array([-dx, -dy, 1.0])
                normal = normal / np.linalg.norm(normal)
                normal_map[y, x] = normal
        
        return normal_map
    
    def fresnel_schlick(self, cos_theta: float, f0: np.ndarray) -> np.ndarray:
        """Schlick菲涅尔方程"""
        return f0 + (1.0 - f0) * np.power(1.0 - cos_theta, 5.0)
    
    def distribution_ggx(self, n_dot_h: float, roughness: float) -> float:
        """GGX/Trowbridge-Reitz分布函数"""
        alpha = roughness * roughness
        alpha_sq = alpha * alpha
        denom = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0
        return alpha_sq / (math.pi * denom * denom)
    
    def geometry_smith(self, n_dot_v: float, n_dot_l: float, roughness: float) -> float:
        """Smith几何函数"""
        def geometry_schlick_ggx(n_dot_v: float, roughness: float) -> float:
            r = roughness + 1.0
            k = r * r / 8.0
            denom = n_dot_v * (1.0 - k) + k
            return n_dot_v / denom
        
        ggx1 = geometry_schlick_ggx(n_dot_v, roughness)
        ggx2 = geometry_schlick_ggx(n_dot_l, roughness)
        return ggx1 * ggx2
    
    def calculate_brdf(self, view_dir: np.ndarray, light_dir: np.ndarray, 
                      normal: np.ndarray, material: MaterialProperties) -> np.ndarray:
        """计算BRDF (双向反射分布函数)"""
        # 半程向量
        half_dir = (view_dir + light_dir) / np.linalg.norm(view_dir + light_dir)
        
        # 各种点积
        n_dot_v = np.clip(np.dot(normal, view_dir), 0.0, 1.0)
        n_dot_l = np.clip(np.dot(normal, light_dir), 0.0, 1.0)
        n_dot_h = np.clip(np.dot(normal, half_dir), 0.0, 1.0)
        h_dot_v = np.clip(np.dot(half_dir, view_dir), 0.0, 1.0)
        
        # 菲涅尔反射率
        f0 = np.array([0.04, 0.04, 0.04])  # 非金属基础反射率
        f0 = np.lerp(f0, material.albedo, material.metallic)
        fresnel = self.fresnel_schlick(h_dot_v, f0)
        
        # 分布函数
        distribution = self.distribution_ggx(n_dot_h, material.roughness)
        
        # 几何函数
        geometry = self.geometry_smith(n_dot_v, n_dot_l, material.roughness)
        
        # Cook-Torrance BRDF
        numerator = distribution * geometry * fresnel
        denominator = 4.0 * n_dot_v * n_dot_l + 0.0001
        specular = numerator / denominator
        
        # 漫反射
        kd = (1.0 - fresnel) * (1.0 - material.metallic)
        diffuse = kd * material.albedo / math.pi
        
        return diffuse + specular
    
    def calculate_subsurface_scattering(self, light_dir: np.ndarray, normal: np.ndarray,
                                      material: MaterialProperties) -> np.ndarray:
        """计算次表面散射"""
        # 简化的次表面散射模型
        n_dot_l = np.clip(np.dot(normal, light_dir), 0.0, 1.0)
        
        # 使用高斯函数模拟散射
        scatter_intensity = material.subsurface_scattering * np.exp(-2.0 * (1.0 - n_dot_l))
        
        # 散射颜色偏向红色（模拟血液）
        scatter_color = np.array([0.8, 0.3, 0.2]) * scatter_intensity
        
        return scatter_color
    
    def render_nail_with_physical_lighting(self, nail_image: np.ndarray, 
                                         normal_map: np.ndarray,
                                         material: MaterialProperties,
                                         nail_shape_mask: np.ndarray) -> np.ndarray:
        """使用物理光照渲染指甲"""
        h, w = nail_image.shape[:2]
        result = np.zeros_like(nail_image, dtype=np.float32)
        
        # 为每个像素计算光照
        for y in range(h):
            for x in range(w):
                if nail_shape_mask[y, x] == 0:
                    continue
                
                # 获取法向量
                normal = normal_map[y, x]
                if np.linalg.norm(normal) == 0:
                    normal = np.array([0, 0, 1])
                
                # 计算视线方向
                pixel_pos = np.array([x - w/2, y - h/2, 0])
                view_dir = (self.camera_position - pixel_pos)
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                # 初始化颜色
                final_color = np.zeros(3)
                
                # 环境光
                final_color += self.ambient_light * material.albedo
                
                # 计算所有光源的贡献
                for light in self.lights:
                    # 光源方向
                    light_dir = (light.position - pixel_pos)
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    
                    # 计算BRDF
                    brdf = self.calculate_brdf(view_dir, light_dir, normal, material)
                    
                    # 计算光照强度
                    n_dot_l = np.clip(np.dot(normal, light_dir), 0.0, 1.0)
                    light_intensity = light.intensity * n_dot_l
                    
                    # 距离衰减
                    distance = np.linalg.norm(light.position - pixel_pos)
                    attenuation = 1.0 / (1.0 + light.falloff * distance * distance)
                    
                    # 最终光照
                    light_contribution = brdf * light.color * light_intensity * attenuation
                    final_color += light_contribution
                    
                    # 次表面散射
                    subsurface = self.calculate_subsurface_scattering(light_dir, normal, material)
                    final_color += subsurface * light.color * light_intensity * attenuation
                
                # 应用基础颜色
                result[y, x] = final_color * nail_image[y, x]
        
        return np.clip(result, 0, 255).astype(np.uint8)

class NailMaterialLibrary:
    """指甲材质库"""
    
    @staticmethod
    def get_natural_nail_material() -> MaterialProperties:
        """自然指甲材质"""
        return MaterialProperties(
            albedo=np.array([0.9, 0.85, 0.8]),
            roughness=0.3,
            metallic=0.0,
            ior=1.55,
            transmission=0.1,
            thickness=0.5,
            subsurface_scattering=0.2
        )
    
    @staticmethod
    def get_glossy_polish_material() -> MaterialProperties:
        """光泽指甲油材质"""
        return MaterialProperties(
            albedo=np.array([0.8, 0.8, 0.8]),
            roughness=0.1,
            metallic=0.0,
            ior=1.6,
            transmission=0.05,
            thickness=0.1,
            subsurface_scattering=0.0
        )
    
    @staticmethod
    def get_metallic_polish_material() -> MaterialProperties:
        """金属指甲油材质"""
        return MaterialProperties(
            albedo=np.array([0.9, 0.8, 0.6]),
            roughness=0.2,
            metallic=0.8,
            ior=1.8,
            transmission=0.0,
            thickness=0.1,
            subsurface_scattering=0.0
        )

def create_realistic_lighting_setup() -> PhysicalLightingSystem:
    """创建真实的光照设置"""
    lighting_system = PhysicalLightingSystem()
    
    # 主光源（模拟太阳光）
    main_light = LightSource(
        position=np.array([2, 1, 3]),
        intensity=1.0,
        color=np.array([1.0, 0.95, 0.9]),  # 暖白光
        light_type="directional",
        falloff=0.1
    )
    
    # 填充光（模拟环境反射）
    fill_light = LightSource(
        position=np.array([-1, 0.5, 2]),
        intensity=0.3,
        color=np.array([0.9, 0.95, 1.0]),  # 冷白光
        light_type="directional",
        falloff=0.05
    )
    
    # 边缘光（模拟轮廓光）
    rim_light = LightSource(
        position=np.array([0, -1, 1]),
        intensity=0.2,
        color=np.array([1.0, 1.0, 1.0]),
        light_type="directional",
        falloff=0.02
    )
    
    lighting_system.add_light(main_light)
    lighting_system.add_light(fill_light)
    lighting_system.add_light(rim_light)
    
    return lighting_system

def generate_height_map_from_nail_shape(nail_mask: np.ndarray, 
                                       curvature: float = 0.1) -> np.ndarray:
    """从指甲形状生成高度图"""
    h, w = nail_mask.shape
    height_map = np.zeros((h, w), dtype=np.float32)
    
    # 使用高斯模糊创建平滑的高度变化
    blurred_mask = cv2.GaussianBlur(nail_mask.astype(np.float32), (21, 21), 5)
    
    # 应用曲率
    for y in range(h):
        for x in range(w):
            if nail_mask[y, x] > 0:
                # 计算到中心的距离
                center_x, center_y = w // 2, h // 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                # 创建弧形高度
                normalized_distance = distance / max_distance
                height = curvature * (1.0 - normalized_distance**2)
                height_map[y, x] = height * blurred_mask[y, x]
    
    return height_map

# 使用示例
def apply_physical_lighting_to_nail(nail_image: np.ndarray, 
                                   nail_mask: np.ndarray,
                                   material_type: MaterialType = MaterialType.GLOSSY) -> np.ndarray:
    """对指甲图像应用物理光照"""
    
    # 创建光照系统
    lighting_system = create_realistic_lighting_setup()
    
    # 选择材质
    if material_type == MaterialType.GLOSSY:
        material = NailMaterialLibrary.get_glossy_polish_material()
    elif material_type == MaterialType.METALLIC:
        material = NailMaterialLibrary.get_metallic_polish_material()
    else:
        material = NailMaterialLibrary.get_natural_nail_material()
    
    # 生成高度图和法线图
    height_map = generate_height_map_from_nail_shape(nail_mask)
    normal_map = lighting_system.calculate_normal_map(height_map)
    
    # 渲染
    result = lighting_system.render_nail_with_physical_lighting(
        nail_image, normal_map, material, nail_mask
    )
    
    return result 