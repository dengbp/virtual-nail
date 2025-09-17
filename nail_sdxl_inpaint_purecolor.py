import cv2
import numpy as np
import os
from pathlib import Path
from nail_sdxl_inpaint_ip_adapter import NailSdxlInpaintIpAdapter
import torch
from PIL import Image
import matplotlib
# 新增：自动色名
try:
    import webcolors
except ImportError:
    webcolors = None
from nail_color_transfer import U2NetMasker

# --- 全局U2Net实例，避免重复加载 ---
_masker_instance = None
def get_masker():
    """获取U2NetMasker的单例，只在首次调用时初始化。"""
    global _masker_instance
    if _masker_instance is None:
        print("首次初始化 U2NetMasker...")
        _masker_instance = U2NetMasker()
        print("U2NetMasker 初始化完成。")
    return _masker_instance
# ------------------------------------

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def rgb_to_name(rgb):
    if webcolors is None:
        return ''
    try:
        return webcolors.rgb_to_name(tuple(rgb))
    except:
        # 最近色名
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

def rgb_to_desc(rgb):
    # 简单描述，可扩展
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return 'milky white'
    if r > 200 and g < 100 and b < 100:
        return 'vivid red'
    if r < 100 and g > 200 and b < 100:
        return 'vivid green'
    if r < 100 and g < 100 and b > 200:
        return 'vivid blue'
    if r > 150 and b > 150:
        return 'pinkish purple'
    if r > 200 and g > 150 and b < 100:
        return 'peach orange'
    if r > 150 and g > 150 and b < 100:
        return 'light yellow'
    if r > 100 and g < 100 and b > 100:
        return 'magenta'
    return 'custom color'

def get_pure_color_from_img(ref_img):
    # OpenCV读取是BGR，这里转为RGB再取均值
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    color_rgb = np.mean(ref_img_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    # 再转回BGR用于OpenCV后续处理
    color_bgr = color_rgb[::-1]
    print("参考图RGB均值:", color_rgb, "用于填充的BGR:", color_bgr)
    return color_bgr, color_rgb

def cie76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def closest_xkcd_color(rgb):
    min_dist = float('inf')
    closest_name = None
    for name, hex in matplotlib.colors.XKCD_COLORS.items():
        xkcd_rgb = matplotlib.colors.hex2color(hex)
        xkcd_rgb = tuple(int(255*x) for x in xkcd_rgb)
        dist = np.linalg.norm(np.array(rgb) - np.array(xkcd_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name.replace('xkcd:', '')
    return closest_name

def process_one(img_path, ref_path, progress_callback=None, nail_processor=None):
    """
    使用"空白画布"策略和"双重掩码"技术，对单张图片进行处理。
    """
    # 1. 初始化和加载
    if nail_processor is None:
        print("未提供预加载模型，正在函数内创建新实例...")
        nail = NailSdxlInpaintIpAdapter()
    else:
        nail = nail_processor
        
    stem = Path(img_path).stem
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"无法读取图片: {img_path}")
        return
    orig_h, orig_w = img_orig.shape[:2]
    print(f"原图尺寸: {orig_w}x{orig_h}")

    # 2. 实时生成基础掩码
    print("正在生成基础掩码...")
    masker = get_masker()
    mask_prob = masker.get_mask(img_orig, str(img_path), disable_cache=True) 
    mask_gray = (mask_prob * 255).astype(np.uint8)

    # 3. 创建双重掩码: 一个用于AI绘画(扩张)，一个用于最终融合(精确)
    print("正在创建双重掩码...")
    _, otsu_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 精确的融合掩码 (稍后使用)
    fusion_mask = cv2.bitwise_not(otsu_mask)
    
    # 供AI绘画用的扩张掩码
    kernel = np.ones((7, 7), np.uint8)
    inpainting_mask = cv2.dilate(fusion_mask, kernel, iterations=1)
    
    # 4. 统一缩放所有图像和掩码
    target_long_side = 1536
    long_side = max(orig_h, orig_w)
    if long_side > target_long_side:
        scale = target_long_side / long_side
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fusion_mask = cv2.resize(fusion_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        inpainting_mask = cv2.resize(inpainting_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"缩放到: {new_w}x{new_h}")
    else:
        new_h, new_w = orig_h, orig_w
        img = img_orig.copy()
        print(f"保持原图尺寸: {new_w}x{new_h}")

    # 5. "空白画布"策略：预填充指甲区域
    print("正在预填充指甲区域，为AI提供干净的画布...")
    style_ref_img = Image.open(ref_path).convert("RGB")
    style_ref_cv = cv2.cvtColor(np.array(style_ref_img), cv2.COLOR_RGB2BGR)
    avg_color = cv2.mean(style_ref_cv)[:3]
    
    img_prefilled = img.copy()
    img_prefilled[inpainting_mask > 0] = avg_color
    print("预填充完成。")
    
    # 6. 准备数据并调用AI
    pil_img_prefilled = Image.fromarray(cv2.cvtColor(img_prefilled, cv2.COLOR_BGR2RGB))
    pil_inpainting_mask = Image.fromarray(inpainting_mask).convert("L")

    # 使用我们之前调试好的最佳参数组合
    prompt = (
        "A photorealistic image of a hand with perfectly applied nail polish. "
        "The nails are glossy with natural highlights. "
        "The nail polish has a smooth, flawless finish, with soft, realistic light reflections. "
        "Visible nail thickness and a natural arc. Realistic shadows. "
        "The cuticle area is clean and the integration with the finger looks natural. "
        "High quality, detailed, studio lighting."
    )
    negative_prompt = (
        "plastic, sticker, fake, floating, unnatural edge, hard border, matte, dull, "
        "cartoon, painting, blurry, low detail, watermark, text, logo, "
        "unpainted nails, different colors, artifacts, distortion, overexposed, white spots"
    )
    
    print("开始使用IP-Adapter生成指甲效果...")
    ai_img_pil = nail.generate(
        image=pil_img_prefilled,
        mask_image=pil_inpainting_mask,
        ip_adapter_image=style_ref_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_scale=0.75,
        num_inference_steps=40,
        guidance_scale=5.0,
        callback=progress_callback
    )
    torch.cuda.empty_cache()

    # 7. AI结果后处理
    ai_img = cv2.cvtColor(np.array(ai_img_pil), cv2.COLOR_RGB2BGR)
    if ai_img.shape[:2] != (new_h, new_w):
        ai_img = cv2.resize(ai_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"data/debug/{stem}_debug_ip_adapter_result.png", ai_img)

    # 8. 使用精确的融合掩码进行最终合成
    print("使用精确、平滑的掩码进行最终融合...")
    maskf = fusion_mask / 255.0
    maskf = cv2.GaussianBlur(maskf, (9, 9), 0) # 边缘羽化
    
    final = img.copy()
    mask_3d = maskf[..., np.newaxis]
    final = img * (1 - mask_3d) + ai_img * mask_3d
    final = final.astype(np.uint8)

    final_path = f"data/output/final/{stem}_final.png"
    cv2.imwrite(final_path, final)
    print(f"完成: {final_path}")
    if progress_callback:
        progress_callback(1.0, 0, 0)

if __name__ == "__main__":
    # 定义输入输出目录
    input_dir = "data/test_images"
    # 参考图名称固定，与优化版和1024版的测试脚本保持一致
    # 用户可以将希望测试的风格图（如 517.png）重命名为 reference.png 并放到 data/reference 目录下
    ref_path = Path("data/reference/reference.png")

    # 检查固定的风格参考图是否存在
    if not ref_path.exists():
        print(f"错误: 固定的风格参考图 '{ref_path}' 不存在。")
        print("请将一个风格参考图（例如 517.png）放到 'data/reference/' 目录下，并重命名为 'reference.png'。")
    else:
        # 获取所有输入图片
    img_files = sorted(list(Path(input_dir).glob("*.*")))
        if not img_files:
            print(f"错误: 在 '{input_dir}' 目录下没有找到任何输入图片。")
        else:
            # 遍历所有输入图片，使用同一个风格参考图进行处理
            print(f"将使用固定的风格参考图 '{ref_path.name}' 处理 '{input_dir}' 中的所有图片。")
            for img_path in img_files:
                print(f"--- 开始处理: {img_path.name} ---")
        process_one(img_path, ref_path) 