import cv2
import numpy as np
import json
import random
from pathlib import Path
import time
import hashlib
from color_antialiased_highlight_visualizer import generate_antialiased_highlight_visualization
import argparse
from datetime import datetime
import threading

# ========== 配置 ==========
HIGHLIGHT_SHAPES_PATH = 'data/highlight/all_highlight_shapes_params.json'
DEBUG_DIR = Path('data/output/debug')

# ========== 全局缓存 ==========
_highlight_shapes_cache = None
_highlight_shapes_cache_lock = threading.Lock()

def get_highlight_shapes():
    """
    获取高光形状数据，使用全局缓存避免重复加载。
    """
    global _highlight_shapes_cache
    if _highlight_shapes_cache is None:
        with _highlight_shapes_cache_lock:
            if _highlight_shapes_cache is None:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载高光形状数据...")
                _highlight_shapes_cache = load_highlight_shapes(HIGHLIGHT_SHAPES_PATH)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 高光形状数据加载完成，共{len(_highlight_shapes_cache)}个形状")
    return _highlight_shapes_cache

# ========== 工具函数 ==========
def load_highlight_shapes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_nail_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 100]  # 过滤小噪点

def get_main_axis_angle(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    # OpenCV角度定义: [-90, 0)，主轴为rect的宽方向
    if rect[1][0] < rect[1][1]:
        angle = angle + 90
    return angle, rect

def get_contour_area(contour):
    return cv2.contourArea(contour)

def get_minarea_rect_points(rect):
    box = cv2.boxPoints(rect)
    return np.int0(box)

def get_nail_tip_center(contour, angle, rect):
    # 获取指甲前盖中心点（主轴正方向一侧的中点）
    box = get_minarea_rect_points(rect)
    center = np.array(rect[0])
    axis_vec = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    dists = [np.dot((pt-center), axis_vec) for pt in box]
    idx_max = int(np.argmax(dists))
    idx_min = int(np.argmin(dists))
    tip_center = (box[idx_max] + box[(idx_max+1)%4]) / 2
    base_center = (box[idx_min] + box[(idx_min+1)%4]) / 2
    return tip_center, base_center, axis_vec

def get_shape_main_axis_and_wide_end(points_norm):
    pts = np.array(points_norm)
    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    projections = np.dot(pts_centered, main_axis)
    idx_max = np.argmax(projections)
    idx_min = np.argmin(projections)
    wide_end = pts[idx_max]
    narrow_end = pts[idx_min]
    return main_axis, wide_end, narrow_end, pts_mean

def norm_shape_to_pixel_with_orient(points_norm, bbox, scale, angle, offset, flip):
    pts = np.array(points_norm, dtype=np.float32)
    cx, cy, w, h = bbox
    pts[:,0] = pts[:,0] * w + cx - w/2
    pts[:,1] = pts[:,1] * h + cy - h/2
    if scale != 1.0:
        pts = (pts - [cx, cy]) * scale + [cx, cy]
    if angle != 0.0:
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = np.dot(pts - [cx, cy], rot.T) + [cx, cy]
    if flip:
        # 以中心为轴，主轴方向翻转
        pts = (pts - [cx, cy]) * np.array([-1, 1]) + [cx, cy]
    pts = pts + np.array(offset)
    return pts.astype(np.int32)

def random_offset_within_mask(mask, shape_pts, max_attempts=50, optimization_level=1):
    h, w = mask.shape
    # 优化：只在第一次计算距离变换，避免重复计算
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 优化：预计算形状边界框
        minx, miny = shape_pts[:,0].min(), shape_pts[:,1].min()
        maxx, maxy = shape_pts[:,0].max(), shape_pts[:,1].max()
    shape_width = maxx - minx
    shape_height = maxy - miny
    
    # 优化：获取所有有效掩码点，避免重复计算
        mask_indices = np.argwhere(mask > 0)
        if len(mask_indices) == 0:
        return None
    
    # 根据优化级别决定检查策略
    if optimization_level == 0:
        # 原始版本：检查所有点
        check_all_points = True
    else:
        # 优化版本：只检查关键点
        check_all_points = False
    
    # 优化：减少尝试次数，提高成功率
    for _ in range(max_attempts):
        # 随机选择一个掩码点作为中心
        cy, cx = random.choice(mask_indices)
        
        # 计算偏移量
        offset = [cx - (minx + maxx)//2, cy - (miny + maxy)//2]
        moved = shape_pts + offset
        moved_int = moved.astype(np.int32)
        
        # 快速边界检查
        if (moved_int[:,1].min() < 0 or moved_int[:,1].max() >= h or 
            moved_int[:,0].min() < 0 or moved_int[:,0].max() >= w):
            continue
            
        # 检查所有点都在掩码内
        if not np.all(mask[moved_int[:,1], moved_int[:,0]] > 0):
            continue
            
        # 根据优化级别选择检查策略
        if check_all_points:
            # 原始版本：检查所有点
            points_to_check = moved_int
        else:
            # 优化版本：只检查关键点
            points_to_check = [
                moved_int[0],  # 第一个点
                moved_int[len(moved_int)//2],  # 中间点
                moved_int[-1]  # 最后一个点
            ]
        
                min_dist_to_edge = float('inf')
        for pt in points_to_check:
                    if 0 <= pt[1] < h and 0 <= pt[0] < w:
                        dist = dist_transform[pt[1], pt[0]]
                        min_dist_to_edge = min(min_dist_to_edge, dist)
        
        if min_dist_to_edge >= 6.0:  # 确保点到边缘≥6像素
            # 快速验证：检查形状是否完全在掩码内
                    test_mask = np.zeros_like(mask)
                    cv2.fillPoly(test_mask, [moved_int], 255)
                    if cv2.countNonZero(cv2.bitwise_and(test_mask, mask)) == cv2.countNonZero(test_mask):
                        return moved_int
    return None

def rotate_points(pts, angle_deg, center):
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] )
    return np.dot(pts - center, rot.T) + center

def mask_erode(mask, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)

def mask_erode_adaptive(mask, max_kernel=9, min_kernel=3):
    orig_area = np.count_nonzero(mask)
    for k in range(max_kernel, min_kernel-1, -2):
        eroded = mask_erode(mask, kernel_size=k)
        if np.count_nonzero(eroded) > orig_area * 0.3:
            return eroded
    return mask_erode(mask, kernel_size=min_kernel)

def poly_iou(poly1, poly2, shape):
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask1, [poly1], 1)
    cv2.fillPoly(mask2, [poly2], 1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def min_poly_distance(poly1, poly2):
    # poly1, poly2: Nx2 int32
    return np.min([np.linalg.norm(p1 - p2) for p1 in poly1 for p2 in poly2])

def place_highlight_on_nail(nail_mask, contour, chosen_shape, rng, img_shape=None, optimization_level=1):
    """
    为指甲放置高光碎片
    
    Args:
        optimization_level: 优化级别
            0: 原始版本（最高质量，最慢）
            1: 轻度优化（推荐，平衡质量和速度）
            2: 重度优化（最快，质量略有下降）
    """
    area = get_contour_area(contour)
    angle_base, rect = get_main_axis_angle(contour)
    cx, cy = rect[0]
    w, h = rect[1]
    bbox = (cx, cy, w, h)
    # 固定为1片高光碎片
    fragments = []
    points_norm = chosen_shape['points_norm']
    shape_np = np.array(points_norm)
    shape_area = 0.5 * np.abs(np.dot(shape_np[:,0], np.roll(shape_np[:,1], 1)) - np.dot(shape_np[:,1], np.roll(shape_np[:,0], 1)))
    shape_axis, wide_end, narrow_end, shape_center = get_shape_main_axis_and_wide_end(points_norm)
    theta_shape = np.rad2deg(np.arctan2(shape_axis[1], shape_axis[0]))
    tip_center, base_center, axis_vec = get_nail_tip_center(contour, angle_base, rect)
    main_axis_len = np.linalg.norm(tip_center - base_center)
    axis_perp = np.array([-axis_vec[1], axis_vec[0]])
    eroded_mask = mask_erode_adaptive(nail_mask, max_kernel=9, min_kernel=3)
    start_time = time.time()
    shape_norm_len = np.linalg.norm(wide_end - narrow_end)
    max_scale_by_length = (main_axis_len * 0.45) / shape_norm_len if shape_norm_len > 0 else 1.0
    min_ratio = 0.001  # 0.1%
    ratio = rng.uniform(0.001, 0.03)  # 0.1%~3%
    
    # 根据优化级别调整参数
    if optimization_level == 0:
        # 原始版本：最高质量
        area_attempts = 4
        attempts_per_area = 100
        timeout_seconds = 2.0
        max_offset_attempts = 100
    elif optimization_level == 1:
        # 轻度优化：推荐设置
        area_attempts = 3
        attempts_per_area = 60
        timeout_seconds = 1.8
        max_offset_attempts = 60
    else:
        # 重度优化：最快速度
        area_attempts = 2
        attempts_per_area = 40
        timeout_seconds = 1.5
        max_offset_attempts = 40
    
    for area_attempt in range(area_attempts):
        for attempt in range(attempts_per_area):
            if time.time() - start_time > timeout_seconds:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARN] 单片高光放置超时，跳过。")
                return fragments
            target_area = area * ratio
            scale_area = np.sqrt(target_area / (shape_area * w * h)) if shape_area > 0 else 1.0
            scale = min(scale_area, max_scale_by_length)
            angle = angle_base + rng.uniform(-10, 10)
            pts = np.array(points_norm, dtype=np.float32)
            pts[:,0] = pts[:,0] * w + cx - w/2
            pts[:,1] = pts[:,1] * h + cy - h/2
            if scale != 1.0:
                pts = (pts - [cx, cy]) * scale + [cx, cy]
            pts = rotate_points(pts, -theta_shape, [cx, cy])
            pts = rotate_points(pts, angle, [cx, cy])
            shape_center_pixel = np.array([shape_center[0]*w + cx - w/2, shape_center[1]*h + cy - h/2])
            shape_center_pixel = rotate_points(np.array([shape_center_pixel]), -theta_shape, [cx, cy])[0]
            shape_center_pixel = rotate_points(np.array([shape_center_pixel]), angle, [cx, cy])[0]
            t = rng.uniform(0.15, 0.20)
            s = rng.uniform(-0.05, 0.05)
            target_center = tip_center - axis_vec * (main_axis_len * t) + axis_perp * (w * s)
            move_vec = target_center - shape_center_pixel
            pts = pts + move_vec.astype(int)
            frag_pts_checked = random_offset_within_mask(eroded_mask, pts, max_attempts=max_offset_attempts)
            if frag_pts_checked is not None:
                fragments.append(frag_pts_checked)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 单片高光放置成功，面积比例{ratio:.3f}")
                return fragments
        ratio *= 0.7
        if ratio < min_ratio:
            break
    return fragments


def draw_antialiased_highlight(highlight_layer, frag_pts, img_shape, blur_ksize=7, scale_factor=2):
    h, w = img_shape[:2]
    big_shape = (w * scale_factor, h * scale_factor)
    big_mask = np.zeros(big_shape, dtype=np.uint8)
    big_frag_pts = (frag_pts * scale_factor).astype(np.int32)
    cv2.fillPoly(big_mask, [big_frag_pts], 255)
    big_mask_blur = cv2.GaussianBlur(big_mask, (blur_ksize, blur_ksize), 0)
    small_mask_blur = cv2.resize(big_mask_blur, (w, h), interpolation=cv2.INTER_AREA)
    highlight_layer = highlight_layer.astype(np.float32)
    for c in range(3):
        highlight_layer[:,:,c] += small_mask_blur * (255.0/255.0)
    highlight_layer = np.clip(highlight_layer, 0, 255).astype(np.uint8)
    return highlight_layer

def add_highlight_to_image(img_path, output_path=None, highlight_shapes=None, optimization_level=1):
    """
    为图像添加高光效果的核心函数，支持模块导入调用。
    
    Args:
        img_path: 输入图像路径
        output_path: 输出路径，如果为None则使用默认路径
        highlight_shapes: 高光形状数据，如果为None则使用全局缓存
        optimization_level: 优化级别
            0: 原始版本（最高质量，最慢）
            1: 轻度优化（推荐，平衡质量和速度）
            2: 重度优化（最快，质量略有下降）
    
    Returns:
        tuple: (基础高光输出路径, 抗锯齿高光输出路径)
    """
    img_path = Path(img_path)
    mask_path = DEBUG_DIR / f"{img_path.stem}_mask.png"
    
    if output_path is None:
    out_path = DEBUG_DIR / f"{img_path.stem}_with_highlight.png"
        out_path_aa = DEBUG_DIR / f"{img_path.stem}_with_antialiased_highlight.png"
    else:
        out_path = DEBUG_DIR / f"{img_path.stem}_with_highlight.png"
        out_path_aa = Path(output_path)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 输入图像: {img_path}\n掩码: {mask_path}")
    
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), 0)
    
    # 使用传入的highlight_shapes或全局缓存
    if highlight_shapes is None:
        highlight_shapes = get_highlight_shapes()
    
    rng = random.Random()
    seed = int(time.time()) + int(hashlib.md5(str(img_path).encode()).hexdigest(), 16) % 100000
    rng.seed(seed)

    # 本次只随机选一个碎片形状
    chosen_shape = rng.choice(highlight_shapes)

    nail_contours = get_nail_contours(mask)
    highlight_layer = np.zeros_like(img)
    all_fragments = []  # 收集所有碎片
    for idx, contour in enumerate(nail_contours):
        single_nail_mask = np.zeros_like(mask)
        cv2.drawContours(single_nail_mask, [contour], -1, 255, -1)
        fragments = place_highlight_on_nail(single_nail_mask, contour, chosen_shape, rng, img.shape, optimization_level)
        for frag_pts in fragments:
            cv2.fillPoly(highlight_layer, [frag_pts], (255,255,255))
            all_fragments.append(frag_pts)
    out_img = img.copy()
    out_img = cv2.addWeighted(out_img, 1.0, highlight_layer, 0.7, 0)
    # 用掩码裁剪高光层，杜绝溢出
    mask3 = np.repeat((mask>0)[:,:,None], 3, axis=2)
    out_img[~mask3] = img[~mask3]
    cv2.imwrite(str(out_path), out_img)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已保存: {out_path}")

    if all_fragments:
        # 1. 生成极限抗锯齿灰度掩码
        antialiased_mask_path = generate_antialiased_highlight_visualization(all_fragments, img.shape)
        # 2. 自动读取像素迁移效果图（与当前输入同名）
        pixel_transplant_path = img_path  # 假设输入就是迁移图
        pixel_img = cv2.imread(str(pixel_transplant_path))
        # 3. 读取抗锯齿灰度掩码
        highlight_mask_gray = cv2.imread(str(antialiased_mask_path), cv2.IMREAD_GRAYSCALE)
        # 4. 生成高光层（白色）
        highlight_color = np.array([255, 255, 255], dtype=np.uint8)
        highlight_layer = np.zeros_like(pixel_img)
        for c in range(3):
            highlight_layer[:, :, c] = highlight_color[c]
        # 5. 归一化掩码
        alpha = highlight_mask_gray.astype(np.float32) / 255.0
        # 6. 线性混合叠加
        out_img_antialiased = pixel_img.copy()
        for c in range(3):
            out_img_antialiased[:, :, c] = (pixel_img[:, :, c] * (1 - alpha) + highlight_layer[:, :, c] * alpha).astype(np.uint8)
        # 7. 保存最终效果图
        cv2.imwrite(str(out_path_aa), out_img_antialiased)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [主流程] 已保存抗锯齿高光叠加效果图: {out_path_aa}")
    
    return str(out_path), str(out_path_aa)

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="输入像素迁移图像路径")
    parser.add_argument("--output", help="抗锯齿高光输出路径", default=None)
    args = parser.parse_args()

    add_highlight_to_image(args.img_path, args.output)

if __name__ == '__main__':
    main()
