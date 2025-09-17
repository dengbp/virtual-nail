import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from generate_initial_masks import U2NET
import torch
from PIL import Image
from skimage.exposure import match_histograms

print("当前脚本路径：", __file__)

# 配置参数
MASK_DIR = 'data/test_masks'
OUTPUT_DIR = 'data/output'
U2NET_MODEL_PATH = 'models/u2net_nail_best.pth'
IMG_SIZE = (320, 320)

# Phong参数
PHONG_PARAMS = {
    "ambient": 0.3,
    "diffuse": 0.7,
    "specular": 0.5,
    "shininess": 32.0,
    "light_pos": (0.0, 0.0, 1.0),
    "view_pos": (0.0, 0.0, 1.0),
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class U2NetMasker:
    def __init__(self, model_path=U2NET_MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = U2NET(3, 1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"U2Net模型加载成功，设备：{self.device}")

    def get_mask(self, image: np.ndarray, image_path: str, disable_cache: bool = False) -> np.ndarray:
        mask_path = os.path.join(MASK_DIR, Path(image_path).stem + "_mask.png")
        
        # 如果禁用缓存或缓存文件不存在，则使用 U²-Net 生成掩码
        if disable_cache or not os.path.exists(mask_path):
            logger.info(f"使用 U²-Net 生成掩码: {image_path}")
            # 生成掩码
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img).resize(IMG_SIZE)
            arr = np.array(pil_img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                d0, *_ = self.model(tensor)
                pred = torch.sigmoid(d0)
            mask = pred.squeeze().cpu().numpy()
            # 只做一次高斯模糊的灰度掩码
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            # 只保留指甲区域的渐变，背景强制为0
            background_threshold = 10
            mask[mask < background_threshold] = 0
            ensure_dir(MASK_DIR)
            cv2.imwrite(mask_path, mask)
            logger.info(f"掩码已生成并保存到: {mask_path}")
            return mask
        else:
            # 使用缓存
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                logger.info(f"使用缓存的掩码: {mask_path}")
                return mask
            else:
                # 如果缓存文件损坏，重新生成
                logger.warning(f"缓存掩码文件损坏，重新生成: {mask_path}")
                return self.get_mask(image, image_path, disable_cache=True)

def apply_phong_shading(image: np.ndarray, mask: np.ndarray, phong_params=PHONG_PARAMS) -> np.ndarray:
    height, width = mask.shape
    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    normal_map[..., 2] = 1.0
    light_pos = np.array(phong_params["light_pos"])
    view_pos = np.array(phong_params["view_pos"])
    shaded = image.astype(np.float32)  # 用原图初始化，未被掩码区域自动保留原图
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                pixel_pos = np.array([x/width, y/height, 0])
                light_dir = light_pos - pixel_pos
                light_dir = light_dir / np.linalg.norm(light_dir)
                diffuse = np.maximum(np.dot(normal_map[y,x], light_dir), 0)
                view_dir = view_pos - pixel_pos
                view_dir = view_dir / np.linalg.norm(view_dir)
                reflect_dir = 2 * np.dot(normal_map[y,x], light_dir) * normal_map[y,x] - light_dir
                specular = np.power(np.maximum(np.dot(view_dir, reflect_dir), 0), phong_params["shininess"])
                ambient = phong_params["ambient"]
                diffuse = phong_params["diffuse"] * diffuse
                specular = phong_params["specular"] * specular
                color = image[y, x].astype(np.float32) / 255.0
                shaded[y, x] = (ambient + diffuse + specular) * color * 255.0
    return np.clip(shaded, 0, 255).astype(np.uint8)

def transfer_color_alpha_only(image, mask, ref_img=None, target_color=(180,105,255)):
    debug_dir = "data/output/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if ref_img is not None and ref_img.shape[:2] != image.shape[:2]:
        ref_img = cv2.resize(ref_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    print("掩码像素值统计：", np.min(mask), np.max(mask), np.mean(mask))
    # 保存原始U2Net灰度掩码
    cv2.imwrite(os.path.join(debug_dir, "debug_mask_raw.png"), mask)
    # 轻微高斯羽化
    mask_f = cv2.GaussianBlur(mask / 255.0, (5,5), 0)
    mask_f = np.clip(mask_f, 0, 1)[..., None]
    cv2.imwrite(os.path.join(debug_dir, "debug_maskf.png"), (mask_f*255).astype(np.uint8))
    if ref_img is not None:
        ref_resized = ref_img
    else:
        ref_resized = np.full_like(image, target_color, dtype=np.uint8)
    result = (image * (1 - mask_f) + ref_resized * mask_f).astype(np.uint8)
    return result

def render_nails(hand_path, mask_path, block_path, out_path, orig_shape=None):
    print(f"[DEBUG] render_nails called! hand_path={hand_path}, mask_path={mask_path}, block_path={block_path}, out_path={out_path}")
    hand = cv2.imread(hand_path) # BGR, uint8
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 0~255, uint8
    block = cv2.imread(block_path) # BGR 纯色块或纹理块
    result = hand.copy()
    # 1. 找出所有指甲轮廓
    m = cv2.GaussianBlur(mask, (3, 3), 0)
    _, m_bin = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] contours found: {len(contours)}")
    for idx, cnt in enumerate(contours):
        # 2. 提取指甲包围盒
        shape_mask = np.zeros_like(mask)
        cv2.drawContours(shape_mask, [cnt], -1, 255, -1)
        ys, xs = np.where(shape_mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h, w = y1 - y0 + 1, x1 - x0 + 1
        # 3. 指甲区域掩码（灰度，软融合，商用级羽化+gamma）
        nail_mask = mask[y0:y1+1, x0:x1+1]
        # 保存原始掩码
        cv2.imwrite(f'data/output/debug/nail_mask_{idx}.png', nail_mask)
        # 掩码自动收缩（腐蚀，极大核）
        kernel = np.ones((21, 21), np.uint8)
        nail_mask_eroded = cv2.erode(nail_mask, kernel, iterations=1)
        cv2.imwrite(f'data/output/debug/nail_mask_eroded_{idx}.png', nail_mask_eroded)
        # 保护带（再腐蚀一次，大核）
        protect_kernel = np.ones((11, 11), np.uint8)
        nail_mask_protect = cv2.erode(nail_mask_eroded, protect_kernel, iterations=1)
        cv2.imwrite(f'data/output/debug/nail_mask_protect_{idx}.png', nail_mask_protect)
        # 软融合用保护带掩码
        mask_f = cv2.GaussianBlur(nail_mask_protect / 255.0, (15,15), 0)[..., None]
        mask_f = mask_f ** 1.5
        # 4. 参考图resize到指甲区域大小
        block_resized = cv2.resize(block, (w, h), interpolation=cv2.INTER_CUBIC)
        # 5. 软融合贴色
        roi = result[y0:y1+1, x0:x1+1]
        result_patch = (roi * (1 - mask_f) + block_resized * mask_f).astype(np.uint8)
        # 6. 指甲边缘加淡阴影
        dist = cv2.distanceTransform((nail_mask > 128).astype(np.uint8), cv2.DIST_L2, 5)
        edge_band = ((dist > 0) & (dist < 8)).astype(np.uint8)
        shadow = np.zeros_like(result_patch)
        shadow[edge_band == 1] = (180, 170, 160)  # 可调肤色或淡灰色
        shadow = cv2.GaussianBlur(shadow, (7, 7), 0)
        result_patch = cv2.addWeighted(result_patch, 1, shadow, 0.10, 0)
        result[y0:y1+1, x0:x1+1] = result_patch
    # 贴色后还原分辨率（如果orig_shape存在）
    if orig_shape is not None:
        result = cv2.resize(result, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, result)

def process_image(image_path: str, ref_img_path: str = None, output_dir: str = OUTPUT_DIR, target_color=(180,105,255), blend_mode="seamless"):
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "masks"))
    ensure_dir(os.path.join(output_dir, "color_transfer"))
    ensure_dir(os.path.join(output_dir, "final"))
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return
    # 统一分辨率处理，降低内存占用
    h, w, _ = image.shape
    if h > 1024 or w > 1024:
        scale = min(1024 / h, 1024 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"原图缩放至 ({new_w}, {new_h}) 以降低内存占用")
    masker = U2NetMasker()
    mask = masker.get_mask(image, image_path)
    mask_path = os.path.join(output_dir, "masks", Path(image_path).stem + "_mask.png")
    cv2.imwrite(mask_path, mask)
    ref_img = None
    block_path = None
    if ref_img_path and os.path.exists(ref_img_path):
        ref_img = cv2.imread(ref_img_path)
        if ref_img is not None and ref_img.shape[:2] != image.shape[:2]:
            ref_img = cv2.resize(ref_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        block_path = os.path.join(output_dir, "color_transfer", Path(image_path).stem + "_block.png")
        cv2.imwrite(block_path, ref_img)
    else:
        ref_img = np.full_like(image, target_color, dtype=np.uint8)
        block_path = os.path.join(output_dir, "color_transfer", Path(image_path).stem + "_block.png")
        cv2.imwrite(block_path, ref_img)
    # 贴色中间效果图（只用Alpha混合）
    color_transfer = transfer_color_alpha_only(image, mask, ref_img, target_color)
    transfer_path = os.path.join(output_dir, "color_transfer", Path(image_path).stem + "_transfer.png")
    cv2.imwrite(transfer_path, color_transfer)
    # 高清美甲渲染（贴色后还原分辨率）
    out_path = os.path.join(output_dir, "final", Path(image_path).stem + "_final.png")
    render_nails(image_path, mask_path, block_path, out_path, orig_shape=(h, w))
    logger.info(f"处理完成: {image_path}")

def process_directory(input_dir: str, ref_img_path: str = None, output_dir: str = OUTPUT_DIR, target_color=(180,105,255), blend_mode="seamless"):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        logger.warning(f"在目录 {input_dir} 中未找到图像文件")
        return
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(input_dir, image_file)
        process_image(image_path, ref_img_path, output_dir, target_color, blend_mode)
    logger.success(f"目录处理完成: {input_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="美甲色彩/纹理迁移流水线（仅用OpenCV+NumPy+Phong+U2Net掩码）")
    parser.add_argument("--input_dir", type=str, default="data/test_images", help="输入图片目录")
    parser.add_argument("--ref_img", type=str, default=None, help="参考色块/纹理图路径，可选")
    parser.add_argument("--output_dir", type=str, default="data/output", help="输出目录")
    parser.add_argument("--target_color", type=str, default="180,105,255", help="纯色迁移的目标颜色（BGR，逗号分隔）")
    parser.add_argument("--blend_mode", type=str, choices=["seamless", "copy", "alpha"], default="alpha", help="融合方式")
    args = parser.parse_args()
    target_color = tuple(map(int, args.target_color.split(",")))
    process_directory(args.input_dir, args.ref_img, args.output_dir, target_color, args.blend_mode)

if __name__ == "__main__":
    main()

input_dir = "data/test_images"
ref_img_path = "data/reference/reference.jpg"
output_mask_dir = "data/output/masks"
output_color_dir = "data/output/color_transfer"
output_final_dir = "data/output/final" 