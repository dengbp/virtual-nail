import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import os
import time

logger = logging.getLogger(__name__)

class NailTemplateFitter:
    """
    指甲模板拟合器
    使用预定义的美甲形状模板对实际指甲轮廓进行拟合，获得更美观、自然的指甲边界
    """
    
    def __init__(self):
        self.templates = self._create_nail_templates()
        self.default_params = {
            'fitting_method': 'affine',  # 'affine', 'similarity', 'rigid'
            'smooth_factor': 0.8,        # 模板与实际轮廓的融合权重
            'min_contour_points': 20,    # 最小轮廓点数
            'max_iterations': 100,       # 最大迭代次数
            'convergence_threshold': 0.01, # 收敛阈值
            'debug_save': True           # 是否保存调试图像
        }
    
    def _create_nail_templates(self) -> Dict[str, np.ndarray]:
        """
        创建标准指甲形状模板
        
        Returns:
            包含各种指甲形状模板的字典
        """
        templates = {}
        
        # === 基础形状模板 ===
        
        # 1. 椭圆形模板（最自然）
        def create_ellipse_template(width=30, height=20, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            return np.column_stack([x, y])
        
        templates['ellipse'] = create_ellipse_template(30, 20, 100)
        
        # 2. 杏仁形模板（优雅）
        def create_almond_template(width=30, height=20, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            # 杏仁形：顶部尖，底部圆
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部变尖
            y[angles < np.pi] *= 0.7
            return np.column_stack([x, y])
        
        templates['almond'] = create_almond_template(30, 20, 100)
        
        # 3. 方形模板（现代）
        def create_square_template(width=30, height=20, points=100):
            # 方形但边缘圆润
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部和底部变平
            y[angles < np.pi/4] = height/2
            y[angles > 3*np.pi/4] = -height/2
            return np.column_stack([x, y])
        
        templates['square'] = create_square_template(30, 20, 100)
        
        # 4. 圆形模板（可爱）
        def create_round_template(width=30, height=20, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            radius = min(width, height) / 2
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            return np.column_stack([x, y])
        
        templates['round'] = create_round_template(30, 20, 100)
        
        # 5. 尖形模板（时尚）
        def create_pointed_template(width=30, height=20, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部更尖
            y[angles < np.pi/2] *= 0.5
            y[angles > 3*np.pi/2] *= 0.5
            return np.column_stack([x, y])
        
        templates['pointed'] = create_pointed_template(30, 20, 100)
        
        # === 扩展形状模板 ===
        
        # 6. 长椭圆形（长指甲）
        templates['long_ellipse'] = create_ellipse_template(25, 35, 100)
        
        # 7. 短椭圆形（短指甲）
        templates['short_ellipse'] = create_ellipse_template(35, 15, 100)
        
        # 8. 宽椭圆形（宽指甲）
        templates['wide_ellipse'] = create_ellipse_template(40, 18, 100)
        
        # 9. 窄椭圆形（窄指甲）
        templates['narrow_ellipse'] = create_ellipse_template(20, 22, 100)
        
        # 10. 长杏仁形（长优雅指甲）
        def create_long_almond_template(width=25, height=35, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部更尖，底部更圆
            y[angles < np.pi] *= 0.6
            return np.column_stack([x, y])
        
        templates['long_almond'] = create_long_almond_template(25, 35, 100)
        
        # 11. 短杏仁形（短优雅指甲）
        def create_short_almond_template(width=35, height=15, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部稍微尖，底部圆
            y[angles < np.pi] *= 0.8
            return np.column_stack([x, y])
        
        templates['short_almond'] = create_short_almond_template(35, 15, 100)
        
        # 12. 长尖形（长时尚指甲）
        def create_long_pointed_template(width=20, height=40, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部非常尖
            y[angles < np.pi/2] *= 0.3
            y[angles > 3*np.pi/2] *= 0.3
            return np.column_stack([x, y])
        
        templates['long_pointed'] = create_long_pointed_template(20, 40, 100)
        
        # 13. 短尖形（短时尚指甲）
        def create_short_pointed_template(width=35, height=15, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部稍微尖
            y[angles < np.pi/2] *= 0.7
            y[angles > 3*np.pi/2] *= 0.7
            return np.column_stack([x, y])
        
        templates['short_pointed'] = create_short_pointed_template(35, 15, 100)
        
        # 14. 宽方形（宽现代指甲）
        def create_wide_square_template(width=40, height=18, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部和底部变平，但保持圆角
            y[angles < np.pi/6] = height/2
            y[angles > 5*np.pi/6] = -height/2
            return np.column_stack([x, y])
        
        templates['wide_square'] = create_wide_square_template(40, 18, 100)
        
        # 15. 窄方形（窄现代指甲）
        def create_narrow_square_template(width=20, height=22, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 顶部和底部变平
            y[angles < np.pi/4] = height/2
            y[angles > 3*np.pi/4] = -height/2
            return np.column_stack([x, y])
        
        templates['narrow_square'] = create_narrow_square_template(20, 22, 100)
        
        # 16. 宽圆形（宽可爱指甲）
        templates['wide_round'] = create_round_template(40, 18, 100)
        
        # 17. 窄圆形（窄可爱指甲）
        templates['narrow_round'] = create_round_template(20, 22, 100)
        
        # 18. 长圆形（长可爱指甲）
        templates['long_round'] = create_round_template(25, 35, 100)
        
        # 19. 短圆形（短可爱指甲）
        templates['short_round'] = create_round_template(35, 15, 100)
        
        # 20. 自然指甲（不规则椭圆）
        def create_natural_template(width=30, height=20, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 添加一些自然的不规则性
            noise = np.random.normal(0, 0.1, len(angles))
            x += noise * width/10
            y += noise * height/10
            return np.column_stack([x, y])
        
        templates['natural'] = create_natural_template(30, 20, 100)
        
        # 21. 扁平指甲（扁平椭圆）
        def create_flat_template(width=35, height=12, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            return np.column_stack([x, y])
        
        templates['flat'] = create_flat_template(35, 12, 100)
        
        # 22. 高拱指甲（高拱椭圆）
        def create_high_arch_template(width=25, height=25, points=100):
            angles = np.linspace(0, 2*np.pi, points)
            x = width/2 * np.cos(angles)
            y = height/2 * np.sin(angles)
            # 增加拱度
            y *= 1.2
            return np.column_stack([x, y])
        
        templates['high_arch'] = create_high_arch_template(25, 25, 100)
        
        return templates
    
    def fit_template_to_contour(self, 
                               contour: np.ndarray, 
                               template_name: str = 'auto',
                               params: Optional[dict] = None) -> np.ndarray:
        """
        将模板拟合到实际轮廓
        
        Args:
            contour: 实际指甲轮廓点
            template_name: 模板名称，'auto'表示自动选择最佳模板
            params: 拟合参数
            
        Returns:
            拟合后的轮廓点
        """
        if params is None:
            params = self.default_params.copy()  # 使用copy避免修改默认参数
        
        # 预处理轮廓
        contour = self._preprocess_contour(contour, params)
        
        if len(contour) < params.get('min_contour_points', 20):  # 使用get方法提供默认值
            logger.warning(f"轮廓点数不足: {len(contour)} < {params.get('min_contour_points', 20)}")
            return contour
        
        # 自动选择最佳模板
        if template_name == 'auto':
            template_name = self._select_best_template(contour)
            logger.info(f"自动选择模板: {template_name}")
        
        if template_name not in self.templates:
            logger.warning(f"模板 {template_name} 不存在，使用椭圆形")
            template_name = 'ellipse'
        
        template = self.templates[template_name].copy()
        
        # === 新增：重采样contour为与template相同的点数 ===
        def resample_contour(contour, n_points):
            contour = contour.squeeze()
            if len(contour) < 2:
                return np.tile(contour, (n_points, 1))
            # 计算每段长度
            dists = np.sqrt(np.sum(np.diff(contour, axis=0, append=contour[:1])**2, axis=1))
            total_length = np.sum(dists)
            if total_length == 0:
                return np.tile(contour[0], (n_points, 1))
            cumulative = np.cumsum(dists)
            cumulative = np.insert(cumulative, 0, 0)
            # 均匀采样点
            sample_points = np.linspace(0, total_length, n_points, endpoint=False)
            resampled = []
            for sp in sample_points:
                idx = np.searchsorted(cumulative, sp, side='right') - 1
                idx_next = (idx + 1) % len(contour)
                t = (sp - cumulative[idx]) / (cumulative[idx_next] - cumulative[idx] + 1e-8)
                pt = (1 - t) * contour[idx] + t * contour[idx_next]
                resampled.append(pt)
            return np.array(resampled)
        # 使contour和template点数一致
        if len(contour) != len(template):
            contour = resample_contour(contour, len(template))
        # === END ===
        
        # 执行拟合
        fitted_contour = self._perform_fitting(contour, template, params)
        
        # 保存调试图像
        if params.get('debug_save', True):
            self._save_debug_image(contour, template, fitted_contour, template_name)
        
        return fitted_contour
    
    def _preprocess_contour(self, contour: np.ndarray, params: dict) -> np.ndarray:
        """预处理轮廓"""
        # 确保轮廓是2D数组
        if contour.ndim == 3:
            contour = contour.squeeze()
        
        # 移除重复点
        unique_points = []
        for point in contour:
            if len(unique_points) == 0 or not np.allclose(point, unique_points[-1]):
                unique_points.append(point)
        
        contour = np.array(unique_points)
        
        # 如果点数太多，进行采样
        if len(contour) > 200:
            indices = np.linspace(0, len(contour)-1, 100, dtype=int)
            contour = contour[indices]
        
        return contour
    
    def _select_best_template(self, contour: np.ndarray) -> str:
        """
        自动选择最佳模板
        
        Args:
            contour: 实际轮廓
            
        Returns:
            最佳模板名称
        """
        # 计算轮廓的特征
        features = self._extract_contour_features(contour)
        
        # 基于特征选择模板
        aspect_ratio = features['aspect_ratio']
        curvature = features['curvature']
        symmetry = features['symmetry']
        solidity = features['solidity']
        
        # 更智能的模板选择逻辑
        if aspect_ratio > 2.0:  # 非常长的指甲
            if curvature > 0.6:  # 较圆润
                return 'long_ellipse'
            elif curvature > 0.4:  # 中等圆润
                return 'long_almond'
            else:  # 较尖
                return 'long_pointed'
        elif aspect_ratio > 1.5:  # 长指甲
            if curvature > 0.6:
                return 'long_ellipse'
            elif curvature > 0.4:
                return 'long_almond'
            else:
                return 'long_pointed'
        elif aspect_ratio < 0.8:  # 短指甲
            if curvature > 0.7:  # 很圆
                return 'short_round'
            elif curvature > 0.5:  # 较圆
                return 'short_ellipse'
            elif symmetry > 0.8:  # 很对称
                return 'short_almond'
            else:
                return 'short_pointed'
        elif aspect_ratio > 1.8:  # 宽指甲
            if curvature > 0.7:
                return 'wide_round'
            elif curvature > 0.5:
                return 'wide_ellipse'
            elif symmetry > 0.8:
                return 'wide_square'
            else:
                return 'wide_ellipse'
        elif aspect_ratio < 0.9:  # 窄指甲
            if curvature > 0.7:
                return 'narrow_round'
            elif curvature > 0.5:
                return 'narrow_ellipse'
            elif symmetry > 0.8:
                return 'narrow_square'
            else:
                return 'narrow_ellipse'
        else:  # 标准比例指甲
            if curvature > 0.7:  # 很圆
                return 'round'
            elif curvature > 0.6:  # 较圆
                return 'ellipse'
            elif symmetry > 0.8:  # 很对称
                if solidity > 0.9:  # 很实心
                    return 'square'
                else:
                    return 'ellipse'
            elif curvature < 0.3:  # 很尖
                return 'pointed'
            elif solidity < 0.7:  # 不规则
                return 'natural'
            else:  # 默认
                return 'ellipse'
    
    def _extract_contour_features(self, contour: np.ndarray) -> dict:
        """提取轮廓特征"""
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # 计算凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        
        # 计算曲率（简化）
        if len(contour) > 10:
            # 计算轮廓的曲率变化
            curvature = self._calculate_curvature(contour)
        else:
            curvature = 0.5
        
        # 计算对称性
        symmetry = self._calculate_symmetry(contour)
        
        return {
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'curvature': curvature,
            'symmetry': symmetry
        }
    
    def _calculate_curvature(self, contour: np.ndarray) -> float:
        """计算轮廓曲率"""
        if len(contour) < 3:
            return 0.5
        
        # 简化的曲率计算
        angles = []
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            p3 = contour[(i + 2) % len(contour)]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            return np.mean(angles) / np.pi
        return 0.5
    
    def _calculate_symmetry(self, contour: np.ndarray) -> float:
        """计算轮廓对称性"""
        if len(contour) < 4:
            return 0.5
        
        # 找到轮廓中心
        center = np.mean(contour, axis=0)
        
        # 计算到中心的距离
        distances = np.linalg.norm(contour - center, axis=1)
        
        # 计算对称性（距离的标准差越小越对称）
        symmetry = 1.0 / (1.0 + np.std(distances))
        
        return symmetry
    
    def _perform_fitting(self, contour: np.ndarray, template: np.ndarray, params: dict) -> np.ndarray:
        """
        执行模板拟合
        
        Args:
            contour: 实际轮廓
            template: 模板轮廓
            params: 拟合参数
            
        Returns:
            拟合后的轮廓
        """
        method = params.get('fitting_method', 'affine')
        
        if method == 'affine':
            return self._affine_fitting(contour, template, params)
        elif method == 'similarity':
            return self._similarity_fitting(contour, template, params)
        elif method == 'rigid':
            return self._rigid_fitting(contour, template, params)
        else:
            return self._affine_fitting(contour, template, params)
    
    def _affine_fitting(self, contour: np.ndarray, template: np.ndarray, params: dict) -> np.ndarray:
        """仿射变换拟合"""
        # 计算轮廓的质心
        contour_center = np.mean(contour, axis=0)
        template_center = np.mean(template, axis=0)
        
        # 计算轮廓的主方向
        contour_principal = self._get_principal_direction(contour)
        template_principal = self._get_principal_direction(template)
        
        # 计算旋转角度
        angle_diff = np.arctan2(contour_principal[1], contour_principal[0]) - \
                    np.arctan2(template_principal[1], template_principal[0])
        
        # 计算缩放因子
        contour_scale = np.std(contour, axis=0)
        template_scale = np.std(template, axis=0)
        scale_x = contour_scale[0] / template_scale[0] if template_scale[0] > 0 else 1
        scale_y = contour_scale[1] / template_scale[1] if template_scale[1] > 0 else 1
        
        # 构建仿射变换矩阵
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)
        
        # 旋转矩阵
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 缩放矩阵
        S = np.array([[scale_x, 0], [0, scale_y]])
        
        # 变换模板
        transformed_template = (template - template_center) @ (R @ S).T + contour_center
        
        # 融合模板和实际轮廓
        smooth_factor = params.get('smooth_factor', 0.8)
        fitted_contour = smooth_factor * transformed_template + (1 - smooth_factor) * contour
        
        return fitted_contour.astype(np.int32)
    
    def _similarity_fitting(self, contour: np.ndarray, template: np.ndarray, params: dict) -> np.ndarray:
        """相似变换拟合（保持形状）"""
        # 计算轮廓的质心
        contour_center = np.mean(contour, axis=0)
        template_center = np.mean(template, axis=0)
        
        # 计算主方向
        contour_principal = self._get_principal_direction(contour)
        template_principal = self._get_principal_direction(template)
        
        # 计算旋转角度
        angle_diff = np.arctan2(contour_principal[1], contour_principal[0]) - \
                    np.arctan2(template_principal[1], template_principal[0])
        
        # 计算统一缩放因子
        contour_scale = np.linalg.norm(contour_principal)
        template_scale = np.linalg.norm(template_principal)
        scale = contour_scale / template_scale if template_scale > 0 else 1
        
        # 构建相似变换矩阵
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)
        
        # 旋转+缩放矩阵
        T = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 变换模板
        transformed_template = (template - template_center) @ T.T + contour_center
        
        # 融合
        smooth_factor = params.get('smooth_factor', 0.8)
        fitted_contour = smooth_factor * transformed_template + (1 - smooth_factor) * contour
        
        return fitted_contour.astype(np.int32)
    
    def _rigid_fitting(self, contour: np.ndarray, template: np.ndarray, params: dict) -> np.ndarray:
        """刚体变换拟合（只旋转平移）"""
        # 计算轮廓的质心
        contour_center = np.mean(contour, axis=0)
        template_center = np.mean(template, axis=0)
        
        # 计算主方向
        contour_principal = self._get_principal_direction(contour)
        template_principal = self._get_principal_direction(template)
        
        # 计算旋转角度
        angle_diff = np.arctan2(contour_principal[1], contour_principal[0]) - \
                    np.arctan2(template_principal[1], template_principal[0])
        
        # 构建旋转矩阵
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 变换模板
        transformed_template = (template - template_center) @ R.T + contour_center
        
        # 融合
        smooth_factor = params.get('smooth_factor', 0.8)
        fitted_contour = smooth_factor * transformed_template + (1 - smooth_factor) * contour
        
        return fitted_contour.astype(np.int32)
    
    def _get_principal_direction(self, points: np.ndarray) -> np.ndarray:
        """获取点集的主方向"""
        if len(points) < 2:
            return np.array([1, 0])
        
        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered.T)
        
        # 计算特征值和特征向量
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 返回最大特征值对应的特征向量
        return eigenvecs[:, -1]
    
    def _save_debug_image(self, original_contour: np.ndarray, template: np.ndarray, 
                         fitted_contour: np.ndarray, template_name: str):
        """保存调试图像"""
        try:
            # 创建画布
            canvas_size = 400
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
            
            # 计算偏移量使轮廓居中
            all_points = np.vstack([original_contour, template, fitted_contour])
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            
            scale = min(canvas_size * 0.8 / (max_coords[0] - min_coords[0]),
                       canvas_size * 0.8 / (max_coords[1] - min_coords[1]))
            
            offset = np.array([canvas_size//2, canvas_size//2]) - \
                    scale * (min_coords + max_coords) / 2
            
            # 绘制原始轮廓（红色）
            original_scaled = (original_contour * scale + offset).astype(np.int32)
            cv2.polylines(canvas, [original_scaled], True, (0, 0, 255), 2)
            
            # 绘制模板轮廓（蓝色）
            template_scaled = (template * scale + offset).astype(np.int32)
            cv2.polylines(canvas, [template_scaled], True, (255, 0, 0), 2)
            
            # 绘制拟合轮廓（绿色）
            fitted_scaled = (fitted_contour * scale + offset).astype(np.int32)
            cv2.polylines(canvas, [fitted_scaled], True, (0, 255, 0), 2)
            
            # 添加文字
            cv2.putText(canvas, f"Template: {template_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(canvas, "Red: Original", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(canvas, "Blue: Template", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(canvas, "Green: Fitted", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 确保目录存在
            debug_dir = "data/output/template_fitting_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存图像到指定目录
            debug_path = os.path.join(debug_dir, f'debug_template_fitting_{template_name}.png')
            cv2.imwrite(debug_path, canvas)
            logger.info(f"调试图像已保存: {debug_path}")
            
        except Exception as e:
            logger.warning(f"保存调试图像失败: {e}")
    
    def apply_template_fitting_to_mask(self, 
                                     mask: np.ndarray, 
                                     params: Optional[dict] = None) -> np.ndarray:
        """
        对整个掩码应用模板拟合
        
        Args:
            mask: 输入掩码
            params: 拟合参数
            
        Returns:
            模板拟合后的掩码
        """
        if params is None:
            params = self.default_params
        
        logger.info("开始应用模板拟合")
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建输出掩码
        fitted_mask = np.zeros_like(mask)
        
        # 生成时间戳用于调试图像命名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for i, contour in enumerate(contours):
            # 检查轮廓是否合理
            area = cv2.contourArea(contour)
            if area < 100:  # 太小忽略
                continue
            
            # 为每个轮廓创建独立的调试参数
            contour_params = params.copy()
            if contour_params.get('debug_save', True):
                # 修改调试图像保存路径，包含时间戳和轮廓索引
                original_save_debug = self._save_debug_image
                
                def save_debug_with_timestamp(original_contour, template, fitted_contour, template_name):
                    try:
                        # 创建画布
                        canvas_size = 400
                        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
                        
                        # 计算偏移量使轮廓居中
                        all_points = np.vstack([original_contour, template, fitted_contour])
                        min_coords = np.min(all_points, axis=0)
                        max_coords = np.max(all_points, axis=0)
                        
                        scale = min(canvas_size * 0.8 / (max_coords[0] - min_coords[0]),
                                   canvas_size * 0.8 / (max_coords[1] - min_coords[1]))
                        
                        offset = np.array([canvas_size//2, canvas_size//2]) - \
                                scale * (min_coords + max_coords) / 2
                        
                        # 绘制原始轮廓（红色）
                        original_scaled = (original_contour * scale + offset).astype(np.int32)
                        cv2.polylines(canvas, [original_scaled], True, (0, 0, 255), 2)
                        
                        # 绘制模板轮廓（蓝色）
                        template_scaled = (template * scale + offset).astype(np.int32)
                        cv2.polylines(canvas, [template_scaled], True, (255, 0, 0), 2)
                        
                        # 绘制拟合轮廓（绿色）
                        fitted_scaled = (fitted_contour * scale + offset).astype(np.int32)
                        cv2.polylines(canvas, [fitted_scaled], True, (0, 255, 0), 2)
                        
                        # 添加文字
                        cv2.putText(canvas, f"Template: {template_name}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(canvas, f"Contour: {i+1}", (10, 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(canvas, "Red: Original", (10, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(canvas, "Blue: Template", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(canvas, "Green: Fitted", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # 确保目录存在
                        debug_dir = "data/output/template_fitting_debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # 保存图像到指定目录，包含时间戳和轮廓索引
                        debug_path = os.path.join(debug_dir, f'{timestamp}_contour_{i+1}_{template_name}.png')
                        cv2.imwrite(debug_path, canvas)
                        logger.info(f"调试图像已保存: {debug_path}")
                        
                    except Exception as e:
                        logger.warning(f"保存调试图像失败: {e}")
                
                # 临时替换保存方法
                self._save_debug_image = save_debug_with_timestamp
            
            # 应用模板拟合
            fitted_contour = self.fit_template_to_contour(contour, 'auto', contour_params)
            
            # 恢复原始保存方法
            if contour_params.get('debug_save', True):
                self._save_debug_image = original_save_debug
            
            # 绘制拟合后的轮廓
            cv2.drawContours(fitted_mask, [fitted_contour], -1, 255, -1)
        
        logger.info("模板拟合完成")
        return fitted_mask


def apply_template_fitting_simple(mask: np.ndarray, 
                                template_name: str = 'auto',
                                smooth_factor: float = 0.8) -> np.ndarray:
    """
    简化的模板拟合函数
    
    Args:
        mask: 输入掩码
        template_name: 模板名称
        smooth_factor: 平滑因子
        
    Returns:
        模板拟合后的掩码
    """
    fitter = NailTemplateFitter()
    params = {
        'fitting_method': 'affine',
        'smooth_factor': smooth_factor,
        'debug_save': True
    }
    
    return fitter.apply_template_fitting_to_mask(mask, params) 