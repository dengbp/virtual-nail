import cv2
import numpy as np
import os

def get_nail_mask(img):
    """自动提取指甲掩码（适合白底样板图）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 50, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask

def add_big_light_spot_glitter(
    img, mask, min_radius_ratio=0.13, max_radius_ratio=0.22, intensity=0.92, erode_px=18
):
    h, w = img.shape[:2]
    kernel = np.ones((erode_px, erode_px), np.uint8)
    safe_mask = cv2.erode(mask, kernel, iterations=1)
    ys, xs = np.where(safe_mask > 0)
    if len(ys) == 0:
        return img.copy()
    n_glitter = np.random.randint(1, 3)  # 1或2个亮片
    glitter = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_glitter):
        idx = np.random.choice(len(ys))
        y, x = ys[idx], xs[idx]
        # 随机椭圆参数
        radius1 = int(w * np.random.uniform(min_radius_ratio, max_radius_ratio))
        radius2 = int(h * np.random.uniform(min_radius_ratio, max_radius_ratio) * np.random.uniform(0.7, 1.0))
        angle = np.random.uniform(0, 180)
        spot = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(spot, (x, y), (radius1, radius2), angle, 0, 360, 255, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=radius1/2, sigmaY=radius2/2)
        spot = (spot.astype(np.float32) / 255.0) * intensity
        glitter = np.maximum(glitter, spot)
    # 只保留掩码内
    glitter = glitter * (safe_mask.astype(np.float32) / 255.0)
    img_float = img.astype(np.float32) / 255.0
    for c in range(3):
        img_float[..., c] = np.clip(img_float[..., c] + glitter, 0, 1)
    out = (img_float * 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    input_path = "data/reference/nail_input.png"
    output_dir = "data/output/debug"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nail_with_big_light_spot_glitter.png")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    mask = get_nail_mask(img)
    out = add_big_light_spot_glitter(img, mask)
    cv2.imwrite(output_path, out)
    print(f"已生成大亮片（灯光高光斑）效果：{output_path}") 