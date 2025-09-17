import cv2
import numpy as np
import os
import sys
import threading
from pathlib import Path
from nail_color_transfer import U2NetMasker
import time
from datetime import datetime

# --- 确保目录存在 ---
Path("data/debug").mkdir(parents=True, exist_ok=True)
Path("data/output/final").mkdir(parents=True, exist_ok=True)

# --- 全局U2Net实例 ---
_masker_instance = None
_masker_lock = threading.Lock()

def get_masker():
    global _masker_instance
    if _masker_instance is None:
        with _masker_lock:
            if _masker_instance is None:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 初始化U²-Net掩码生成器...")
                _masker_instance = U2NetMasker()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] U²-Net掩码生成器初始化完成。")
    return _masker_instance

def resize_image_long_edge(img, max_long_edge=1024):
    """
    将图像长边缩放到指定尺寸，短边等比缩放。
    """
    h, w = img.shape[:2]
    if max(h, w) > max_long_edge:
        if h > w:
            new_h = max_long_edge
            new_w = int(w * (max_long_edge / h))
        else:
            new_w = max_long_edge
            new_h = int(h * (max_long_edge / w))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img

def get_keypoints_from_mask(mask):
    """
    最终稳定版: 结合了边界框的稳定性和8点采样的塑形能力。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(main_contour)
    
    # 定义8个关键点: 4个角点和4个边的中点
    keypoints = np.array([
        [x, y],                             # Top-Left
        [x + w // 2, y],                    # Top-Mid
        [x + w, y],                         # Top-Right
        [x + w, y + h // 2],                # Right-Mid
        [x + w, y + h],                     # Bottom-Right
        [x + w // 2, y + h],                # Bottom-Mid
        [x, y + h],                         # Bottom-Left
        [x, y + h // 2]                     # Left-Mid
    ], dtype=np.float32)

    return keypoints.astype(int)

def process_one_pixel_transplant_auto(img_path, ref_path):
    """
    使用TPS变形和无缝融合实现全自动像素级内容搬运。
    """
    start_time = time.time()
    stem = Path(img_path).stem
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- 正在处理 (最终稳定版): {stem} ---")

    # 1. 读取图像
    img_orig = cv2.imread(str(img_path))
    ref_img_orig = cv2.imread(str(ref_path))

    if img_orig is None or ref_img_orig is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 无法读取图像文件")
        return None

    # 2. 统一尺寸 (遵循全局1024策略)
    img = resize_image_long_edge(img_orig)
    ref_img = resize_image_long_edge(ref_img_orig)
    
    # 3. 生成掩码
    masker = get_masker()
    mask_prob = masker.get_mask(img, str(img_path), disable_cache=True)
    u2net_mask_path = f"data/output/debug/{stem}_u2net_mask.png"
    cv2.imwrite(u2net_mask_path, mask_prob)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] U2Net原始掩码已保存: {u2net_mask_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] U2Net原始掩码 min/max/mean: {mask_prob.min()}/{mask_prob.max()}/{mask_prob.mean():.2f}")

    # 掩码二值化
    _, nail_mask_raw = cv2.threshold(mask_prob.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
    mask_save_path = f"data/output/debug/{stem}_pixel_transplant_mask.png"
    cv2.imwrite(mask_save_path, nail_mask_raw)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 像素迁移阶段掩码已保存: {mask_save_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 像素迁移阶段掩码 min/max/mean: {nail_mask_raw.min()}/{nail_mask_raw.max()}/{nail_mask_raw.mean():.2f}")

    # 4. 找到所有轮廓
    contours, _ = cv2.findContours(nail_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 无法在图像中找到任何指甲轮廓。")
        return None

    # 5. 提取参考图关键点
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    _, ref_mask = cv2.threshold(ref_gray, 240, 255, cv2.THRESH_BINARY_INV)
    src_points = get_keypoints_from_mask(ref_mask)
    if src_points is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 无法从参考图中提取关键点。")
        return None

    # 6. 循环处理
    output_image = img.copy()
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue
            
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理指甲 {i+1}/{len(contours)}...")

        # a. 创建单个指甲掩码
        single_nail_mask = np.zeros_like(nail_mask_raw)
        cv2.drawContours(single_nail_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # b. 提取指甲关键点
        dst_points = get_keypoints_from_mask(single_nail_mask)
        if dst_points is None:
            continue

        # c. 手动分片仿射变换 (100% OpenCV)
        # 基于8个关键点定义一个固定的三角剖分
        triangles = np.array([
            [0, 1, 7], [1, 2, 3], [1, 3, 7],
            [3, 4, 5], [3, 5, 7], [5, 6, 7]
        ], dtype=int)

        # 创建一个黑色画布，用于拼接所有变形后的三角形
        warped_canvas = np.zeros_like(img)

        for tri_indices in triangles:
            src_tri = src_points[tri_indices].astype(np.float32)
            dst_tri = dst_points[tri_indices].astype(np.float32)

            # 计算从源三角形到目标三角形的仿射变换
            M = cv2.getAffineTransform(src_tri, dst_tri)
            
            # 对整个参考图进行仿射变换
            warped_img_full = cv2.warpAffine(ref_img, M, (img.shape[1], img.shape[0]))

            # 创建一个只包含当前目标三角形的掩码
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [dst_tri.astype(int)], 0, 255, -1)
            
            # 使用掩码，将变形后图像的相应部分复制到画布上
            cv2.copyTo(warped_img_full, mask, warped_canvas)
        
        # 经过所有三角形的拼接，画布上已经有了完整的、精确变形的参考图
        warped_ref = warped_canvas

        # d. 最终方案: Alpha融合 (放弃不稳定且会导致崩溃的seamlessClone)
        #    1. 对指甲掩码进行腐蚀，收缩边缘，防止溢出
        kernel = np.ones((9,9), np.uint8)
        blend_mask = cv2.erode(single_nail_mask, kernel, iterations=2)

        #    2. 对收缩后的掩码进行高斯模糊，制造平滑的羽化边缘
        blend_mask = cv2.GaussianBlur(blend_mask, (15, 15), 0)

        #    3. 将掩码转换为浮点数，用于alpha通道计算 (范围 0.0 - 1.0)
        #       修正: 先转BGR，再转float，避免数据类型错误
        alpha_mask_bgr_uint8 = cv2.cvtColor(blend_mask, cv2.COLOR_GRAY2BGR)
        alpha_mask = alpha_mask_bgr_uint8.astype(float) / 255.0

        #    4. 定位融合区域
        x, y, w, h = cv2.boundingRect(contour)
        roi = output_image[y:y+h, x:x+w]
        warped_roi = warped_ref[y:y+h, x:x+w]
        alpha_roi = alpha_mask[y:y+h, x:x+w]

        #    5. 执行Alpha融合
        #       背景 * (1 - alpha) + 前景 * alpha
        blended_roi = cv2.multiply(roi.astype(float), 1.0 - alpha_roi)
        colored_roi = cv2.multiply(warped_roi.astype(float), alpha_roi)
        result_roi = cv2.add(blended_roi, colored_roi)

        #    6. 将结果放回输出图像
        output_image[y:y+h, x:x+w] = result_roi.astype(np.uint8)

    # 7. 保存结果
    debug_dir = "data/output/debug"
    os.makedirs(debug_dir, exist_ok=True)
    final_path = f"{debug_dir}/{stem}_pixel_transplant.png"
    cv2.imwrite(final_path, output_image)

    # 后续流程如AI精炼等请用 optimized_mask 或 optimized_mask_bin
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 完成! 结果已保存: {final_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 总耗时: {time.time() - start_time:.2f}秒")
    return final_path

def generate_mask_with_u2net(input_img_path, output_mask_path):
    """
    用U²-Net自动生成掩码并保存到output_mask_path，模型和推理方式与editor_image_server_optimized_1024.py一致。
    """
    masker = get_masker()  # 复用全局U2Net实例
    img = cv2.imread(input_img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_img_path}")
    mask = masker.get_mask(img, input_img_path, disable_cache=True)  # 返回单通道0-255掩码
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    cv2.imwrite(output_mask_path, mask)

def ensure_mask_exists(orig_img_path):
    img_stem = Path(orig_img_path).stem
    mask_path = f"data/test_masks/{img_stem}_mask.png"
    if not os.path.exists(mask_path):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 掩码不存在，自动生成: {mask_path}")
        generate_mask_with_u2net(orig_img_path, mask_path)
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 掩码已存在: {mask_path}")
    return mask_path

if __name__ == "__main__":
    input_dir = Path("data/test_images")
    ref_dir = Path("data/reference")

    if not input_dir.exists():
        sys.exit(f"错误: 输入目录 '{input_dir}' 不存在。")

    ref_paths = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.jpeg"))
    if not ref_paths:
        sys.exit(f"错误: 参考图目录 '{ref_dir}' 为空。")
    ref_path = ref_paths[0]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 使用参考图: {ref_path}")

    img_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not img_files:
        sys.exit(f"错误: 在 '{input_dir}' 目录下没有找到任何图片文件。")

    for img_path in img_files:
        mask_path = ensure_mask_exists(img_path)
        process_one_pixel_transplant_auto(img_path, ref_path)
        break # 只处理一张 