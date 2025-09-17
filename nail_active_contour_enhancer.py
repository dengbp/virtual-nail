#!/usr/bin/env python3
"""
指甲掩码Active Contour增强模块 - 中期方案
专门针对指甲掩码进行Active Contour优化，提升边缘精度
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)

class NailActiveContourEnhancer:
    """
    指甲掩码Active Contour增强器
    使用Snake算法优化指甲轮廓边缘
    """
    
    def __init__(self):
        self.default_params = {
            'alpha': 0.1,           # 弹性系数
            'beta': 0.1,            # 刚性系数
            'gamma': 0.05,          # 步长
            'iterations': 50,       # 迭代次数
            'convergence': 0.1,     # 收敛阈值
            'window_size': 3,       # 搜索窗口大小
            'min_contour_length': 10, # 最小轮廓长度
            'edge_expansion': 3,    # 边缘扩展像素数（新增）
            'feather_width': 5,     # 羽化宽度（新增）
            'preserve_transition': True, # 是否保留过渡区域（新增）
        }
    
    def enhance_mask(self, 
                    mask: np.ndarray, 
                    image: np.ndarray,
                    params: Optional[dict] = None) -> np.ndarray:
        """
        使用Active Contour增强掩码边缘精度，并进行多源掩码智能融合和抗锯齿处理
        
        Args:
            mask: 原始掩码
            image: 原始图像
            params: 参数
            
        Returns:
            智能融合+抗锯齿优化后的掩码
        """
        if params is None:
            params = self.default_params
        
        logger.info("开始Active Contour增强")
        print("[LOG] enhance_mask: Active Contour增强流程被调用")
        
        # 1. 预处理掩码
        processed_mask = self._preprocess_mask(mask)
        cv2.imwrite('debug_processed_mask.png', processed_mask)
        
        # 2. Active Contour增强
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ac_mask = np.zeros_like(processed_mask)
        for i, contour in enumerate(contours):
            if len(contour) < params['min_contour_length']:
                continue
            refined_contour = self._apply_snake_algorithm(contour, image, params)
            cv2.drawContours(ac_mask, [refined_contour], -1, 255, -1)
        cv2.imwrite('debug_ac_mask.png', ac_mask)
        
        # 3. 轮廓平滑掩码（多边形近似+高斯）
        smooth_mask = self._smooth_poly_mask(processed_mask)
        cv2.imwrite('debug_smooth_mask.png', smooth_mask)
        
        # 4. 椭圆拟合掩码
        ellipse_mask = self._ellipse_mask(processed_mask)
        cv2.imwrite('debug_ellipse_mask.png', ellipse_mask)
        
        # 5. 多源融合
        fused_mask = self._smart_fuse_nail_mask(
            raw_mask=processed_mask,
            contour_mask=ac_mask,
            ellipse_mask=ellipse_mask,
            smooth_mask=smooth_mask,
            feather_width=params.get('feather_width', 5)
        )
        cv2.imwrite('debug_fused_mask.png', fused_mask)
        
        # 6. 超分辨率插值+高斯羽化+轮廓重建抗锯齿
        fused_mask = self._anti_alias_nail_mask(fused_mask, upscale=4, feather=7)
        cv2.imwrite('debug_optimized_mask.png', fused_mask)
        print("[LOG] enhance_mask: 已保存所有中间掩码图像 (debug_*.png)")
        
        logger.info("多源掩码融合+抗锯齿优化完成")
        return fused_mask
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """预处理掩码"""
        # 确保掩码是单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 灰度掩码直接返回，不做二值化
        return mask
    
    def _apply_snake_algorithm(self, contour: np.ndarray, image: np.ndarray, params: dict) -> np.ndarray:
        """
        应用Snake算法优化轮廓
        
        Args:
            contour: 输入轮廓
            image: 原始图像
            params: 参数
            
        Returns:
            优化后的轮廓
        """
        # 转换为浮点数
        snake = contour.astype(np.float32).reshape(-1, 2)
        
        # 创建能量图像
        energy_image = self._create_energy_image(image)
        
        # Snake算法迭代
        for iteration in range(params['iterations']):
            old_snake = snake.copy()
            
            # 对每个点进行优化
            for i in range(len(snake)):
                snake[i] = self._optimize_point(snake, i, energy_image, params)
            
            # 检查收敛
            if self._check_convergence(snake, old_snake, params['convergence']):
                logger.info(f"Snake算法在第{iteration+1}次迭代后收敛")
                break
        
        return snake.astype(np.int32).reshape(-1, 1, 2)
    
    def _create_energy_image(self, image: np.ndarray) -> np.ndarray:
        """
        创建能量图像
        
        Args:
            image: 原始图像
            
        Returns:
            能量图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算梯度
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 归一化梯度
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # 创建外部能量（负梯度，吸引到边缘）
        external_energy = -gradient_magnitude
        
        # 添加高斯平滑
        external_energy = cv2.GaussianBlur(external_energy, (5, 5), 0)
        
        return external_energy.astype(np.float32)
    
    def _optimize_point(self, snake: np.ndarray, point_idx: int, energy_image: np.ndarray, params: dict) -> np.ndarray:
        """
        优化单个点
        
        Args:
            snake: 蛇形轮廓
            point_idx: 点的索引
            energy_image: 能量图像
            params: 参数
            
        Returns:
            优化后的点位置
        """
        current_pos = snake[point_idx].copy()
        best_pos = current_pos.copy()
        best_energy = self._calculate_point_energy(snake, point_idx, energy_image, params)
        
        # 在搜索窗口内寻找最佳位置
        window_size = params['window_size']
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                new_pos = current_pos.copy()
                new_pos[0] += dx
                new_pos[1] += dy
                
                # 检查边界
                if (0 <= new_pos[0] < energy_image.shape[1] and 
                    0 <= new_pos[1] < energy_image.shape[0]):
                    
                    # 临时更新snake
                    temp_snake = snake.copy()
                    temp_snake[point_idx] = new_pos
                    
                    # 计算新能量
                    new_energy = self._calculate_point_energy(temp_snake, point_idx, energy_image, params)
                    
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pos = new_pos.copy()
        
        return best_pos
    
    def _calculate_point_energy(self, snake: np.ndarray, point_idx: int, 
                              energy_image: np.ndarray, params: dict) -> float:
        """
        计算点的总能量
        
        Args:
            snake: 蛇形轮廓
            point_idx: 点的索引
            energy_image: 能量图像
            params: 参数
            
        Returns:
            总能量
        """
        n_points = len(snake)
        
        # 获取当前点及其邻居
        current = snake[point_idx]
        prev_idx = (point_idx - 1) % n_points
        next_idx = (point_idx + 1) % n_points
        prev = snake[prev_idx]
        next_point = snake[next_idx]
        
        # 内部能量
        # 弹性项：控制点之间的距离
        elastic_energy = params['alpha'] * (
            np.linalg.norm(current - prev) + np.linalg.norm(current - next_point)
        )
        
        # 刚性项：控制曲率
        if n_points > 2:
            curvature = np.linalg.norm(prev - 2*current + next_point)
            rigid_energy = params['beta'] * curvature
        else:
            rigid_energy = 0
        
        # 外部能量（图像梯度）
        x, y = int(current[0]), int(current[1])
        if 0 <= x < energy_image.shape[1] and 0 <= y < energy_image.shape[0]:
            external_energy = energy_image[y, x]
        else:
            external_energy = 0
        
        # 总能量
        total_energy = elastic_energy + rigid_energy + external_energy
        
        return total_energy
    
    def _check_convergence(self, snake: np.ndarray, old_snake: np.ndarray, threshold: float) -> bool:
        """
        检查是否收敛
        
        Args:
            snake: 当前轮廓
            old_snake: 上一次轮廓
            threshold: 收敛阈值
            
        Returns:
            是否收敛
        """
        if len(snake) == 0:
            return True
        
        # 计算平均位移
        displacement = np.mean(np.linalg.norm(snake - old_snake, axis=1))
        return displacement < threshold
    
    def _smooth_poly_mask(self, mask: np.ndarray) -> np.ndarray:
        """多边形近似+高斯平滑掩码"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_mask = np.zeros_like(mask)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, epsilon=2, closed=True)
            temp = np.zeros_like(mask)
            cv2.drawContours(temp, [approx], -1, 255, -1)
            temp = cv2.GaussianBlur(temp, (5,5), 0)
            smooth_mask = cv2.bitwise_or(smooth_mask, temp)
        return smooth_mask

    def _ellipse_mask(self, mask: np.ndarray) -> np.ndarray:
        """椭圆拟合掩码"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse_mask = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if len(cnt) >= 5 and area > 100:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            else:
                cv2.drawContours(ellipse_mask, [cnt], -1, 255, -1)
        return ellipse_mask

    def _smart_fuse_nail_mask(self, raw_mask, contour_mask, ellipse_mask, smooth_mask, feather_width=6):
        """多源掩码智能融合+羽化"""
        # 1. 多源最大值融合
        fused = np.maximum.reduce([raw_mask, contour_mask, ellipse_mask, smooth_mask])
        # 2. 中心区域硬掩码，边缘羽化
        kernel = np.ones((feather_width, feather_width), np.uint8)
        center = cv2.erode(fused, kernel, iterations=1)
        feather = cv2.GaussianBlur(fused.astype(np.float32), (feather_width*2+1, feather_width*2+1), feather_width/3)
        mask_final = np.zeros_like(fused)
        mask_final[center > 0] = 255
        edge = (fused > 0) & (center == 0)
        mask_final[edge] = (feather[edge]).astype(np.uint8)
        return mask_final

    def _anti_alias_nail_mask(self, mask, upscale=4, feather=7):
        """超分辨率插值+高斯羽化+轮廓重建抗锯齿"""
        print("[LOG] _anti_alias_nail_mask: 抗锯齿优化被调用 (upscale={}, feather={})".format(upscale, feather))
        h, w = mask.shape
        mask_up = cv2.resize(mask, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC)
        mask_blur = cv2.GaussianBlur(mask_up, (feather*2+1, feather*2+1), feather)
        mask_down = cv2.resize(mask_blur, (w, h), interpolation=cv2.INTER_AREA)
        mask_final = np.clip(mask_down, 0, 255).astype(np.uint8)
        mask_final = self._smooth_poly_mask(mask_final)
        # 样条曲线平滑
        def spline_smooth_mask(mask, smooth_factor=0.01, points_per_contour=200):
            mask_bin = (mask > 128).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            smooth_mask = np.zeros_like(mask_bin)
            for cnt in contours:
                if len(cnt) < 8:
                    continue
                cnt = cnt.squeeze()
                if cnt.ndim != 2 or cnt.shape[0] < 8:
                    continue
                cnt = np.vstack([cnt, cnt[0]])
                x, y = cnt[:, 0], cnt[:, 1]
                tck, u = splprep([x, y], s=smooth_factor*len(cnt), per=True)
                u_fine = np.linspace(0, 1, points_per_contour)
                x_fine, y_fine = splev(u_fine, tck)
                smooth_cnt = np.stack([x_fine, y_fine], axis=1).astype(np.int32)
                cv2.drawContours(smooth_mask, [smooth_cnt], -1, 1, -1)
            return (smooth_mask * 255).astype(np.uint8)
        mask_final = spline_smooth_mask(mask_final, smooth_factor=0.01, points_per_contour=200)
        cv2.imwrite('debug_anti_alias_mask.png', mask_final)
        print("[LOG] _anti_alias_nail_mask: 已保存抗锯齿掩码 debug_anti_alias_mask.png (含样条平滑)")
        return mask_final


def enhance_with_active_contour_simple(mask: np.ndarray, 
                                     image: np.ndarray,
                                     iterations: int = 30,
                                     edge_expansion: int = 3,
                                     feather_width: int = 5) -> np.ndarray:
    """
    简化的Active Contour增强函数
    
    Args:
        mask: 原始掩码
        image: 原始图像
        iterations: Snake迭代次数
        edge_expansion: 边缘扩展像素数，用于保留皮肤-指甲过渡区域
        feather_width: 羽化宽度，用于创建自然的边缘过渡
        
    Returns:
        Active Contour增强后的掩码
        
    Note:
        - 输入图像通常已经通过resize_to_max1536处理，长边最大1536像素
        - Active Contour增强会保持输入图像的原始分辨率
        - 输出掩码与输入图像尺寸一致
        - edge_expansion和feather_width参数用于保留自然的皮肤-指甲过渡区域
    """
    enhancer = NailActiveContourEnhancer()
    params = enhancer.default_params.copy()
    params['iterations'] = iterations
    params['edge_expansion'] = edge_expansion
    params['feather_width'] = feather_width
    
    return enhancer.enhance_mask(mask, image, params)


# 测试函数
def test_active_contour_enhancement():
    """测试Active Contour增强功能"""
    import os
    
    # 创建测试图像和掩码
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    test_image[100:300, 200:400] = [255, 200, 150]  # 模拟手部区域
    
    # 创建模拟的指甲掩码
    test_mask = np.zeros((400, 600), dtype=np.uint8)
    test_mask[120:140, 220:280] = 255  # 水平指甲
    
    # 应用Active Contour增强
    enhanced_mask = enhance_with_active_contour_simple(test_mask, test_image, iterations=20)
    
    # 保存结果
    os.makedirs('test_output', exist_ok=True)
    cv2.imwrite('test_output/original_mask.png', test_mask)
    cv2.imwrite('test_output/active_contour_enhanced.png', enhanced_mask)
    cv2.imwrite('test_output/test_image.png', test_image)
    
    print("Active Contour增强测试完成，结果保存在 test_output 目录")


if __name__ == "__main__":
    test_active_contour_enhancement() 