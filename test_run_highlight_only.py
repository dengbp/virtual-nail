import cv2
import numpy as np
from color_nail_highlight_shader import add_nail_highlight_with_shader
from color_transfer_pixel_level_transplant import get_masker
import os

# ========== 玻璃碎片贴图自动生成函数 ==========
def generate_fragment_maps(shape, nail_mask, n_fragments=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    h, w = shape
    points = np.stack([
        np.random.randint(0, w, n_fragments),
        np.random.randint(0, h, n_fragments)
    ], axis=1)
    yx = np.indices((h, w)).transpose(1,2,0)
    dists = np.linalg.norm(yx[None, ...] - points[:, None, None, :], axis=-1)
    labels = np.argmin(dists, axis=0)
    alpha_map = np.zeros((h, w), dtype='uint8')
    for i in range(n_fragments):
        mask = (labels==i) & (nail_mask>0)
        if np.sum(mask) == 0:
            continue
        brightness = np.random.uniform(0.5, 1.0)  # 每个碎片亮度随机
        alpha_map[mask] = int(255 * brightness)
    return alpha_map

if __name__ == "__main__":
    input_path = r"data/output/debug/120745432_pixel_transplant.png"
    output_path = r"data/output/debug/120745432_pixel_transplant_with_highlight.png"
    out_dir = "data/out/fragments"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    masker = get_masker()
    mask_prob = masker.get_mask(img, input_path, disable_cache=True)
    _, nail_mask = cv2.threshold(mask_prob.astype('uint8'), 128, 255, cv2.THRESH_BINARY)

    alpha_img = generate_fragment_maps(img.shape[:2], nail_mask, n_fragments=100, seed=42)
    alpha_path = os.path.join(out_dir, "alpha_mask.png")
    cv2.imwrite(alpha_path, alpha_img)
    alpha_img = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

    add_nail_highlight_with_shader(
        img,
        output_path,
        alpha_mask=alpha_img
    )
    print(f"高光/亮片效果已保存: {output_path}") 