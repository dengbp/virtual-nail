import cv2
import numpy as np
import os
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
import torch

input_dir = "data/test_images"
ref_dir = "data/reference"
output_mask_dir = "data/output/masks"
output_color_dir = "data/output/color_transfer"
output_final_dir = "data/output/final"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_color_dir, exist_ok=True)
os.makedirs(output_final_dir, exist_ok=True)

nail = NailSDXLInpaintOpenCV()

def get_pure_color_from_img(ref_img):
    # OpenCV读取是BGR，这里转为RGB再取均值
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    color_rgb = np.mean(ref_img_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    # 再转回BGR用于OpenCV后续处理
    color_bgr = color_rgb[::-1]
    print("参考图RGB均值:", color_rgb, "用于填充的BGR:", color_bgr)
    return color_bgr

def analyze_image_quality(img):
    """
    分析图像质量，用于智能调整推理步数
    """
    # 计算图像清晰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 计算图像复杂度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 计算颜色丰富度
    color_std = np.std(img, axis=(0, 1)).mean()
    
    # 综合质量评分 (0-1)
    clarity_score = min(laplacian_var / 500, 1.0)
    complexity_score = min(edge_density * 1000, 1.0)
    color_score = min(color_std / 50, 1.0)
    
    quality_score = (clarity_score + complexity_score + color_score) / 3
    
    print(f"图像质量分析: 清晰度={clarity_score:.2f}, 复杂度={complexity_score:.2f}, 颜色丰富度={color_score:.2f}, 综合评分={quality_score:.2f}")
    
    return quality_score

def adaptive_inference_steps(image_quality, user_preference="balanced", mask_complexity=0.5):
    """
    根据图像质量、用户偏好和掩码复杂度自适应调整推理步数
    """
    # 基础步数
    base_steps = 25
    
    # 根据图像质量调整
    if image_quality > 0.7:
        quality_adjustment = 5  # 高质量图像，增加步数
    elif image_quality > 0.4:
        quality_adjustment = 0  # 中等质量，保持基础步数
    else:
        quality_adjustment = -5  # 低质量图像，减少步数
    
    # 根据用户偏好调整
    if user_preference == "quality":
        preference_adjustment = 10
    elif user_preference == "speed":
        preference_adjustment = -10
    else:  # balanced
        preference_adjustment = 0
    
    # 根据掩码复杂度调整
    if mask_complexity > 0.7:
        mask_adjustment = 5  # 复杂掩码，增加步数
    elif mask_complexity > 0.3:
        mask_adjustment = 0  # 中等复杂度，保持基础步数
    else:
        mask_adjustment = -5  # 简单掩码，减少步数
    
    # 计算最终步数
    final_steps = base_steps + quality_adjustment + preference_adjustment + mask_adjustment
    
    # 限制在合理范围内
    final_steps = max(15, min(40, final_steps))
    
    print(f"自适应步数调整: 基础{base_steps} + 质量{quality_adjustment} + 偏好{preference_adjustment} + 掩码{mask_adjustment} = {final_steps}")
    
    return final_steps

def optimize_parameters_for_steps(steps):
    """
    根据推理步数优化其他参数以补偿质量损失
    """
    if steps < 25:
        # 低步数优化策略
        guidance_scale = 12.0  # 提高引导强度
        controlnet_weights = [1.5, 0.8]  # 增强ControlNet控制
        print(f"低步数优化: Guidance={guidance_scale}, ControlNet权重={controlnet_weights}")
    elif steps < 30:
        # 中等步数优化策略
        guidance_scale = 10.0
        controlnet_weights = [1.3, 0.75]
        print(f"中等步数优化: Guidance={guidance_scale}, ControlNet权重={controlnet_weights}")
    else:
        # 高步数标准策略
        guidance_scale = 9.0
        controlnet_weights = [1.2, 0.7]
        print(f"高步数标准: Guidance={guidance_scale}, ControlNet权重={controlnet_weights}")
    
    return guidance_scale, controlnet_weights

def process_one_optimized(img_path, ref_path, mask_path=None, progress_callback=None, user_preference="balanced"):
    """
    优化版本的process_one函数，支持智能步数调整
    """
    stem = Path(img_path).stem
    
    # 直接用data/test_masks下的掩码
    if mask_path is None:
        mask_path = f"data/test_masks/{stem}_mask_input_mask.png"
    
    # 进度回调函数（如果未提供则使用默认的）
    if progress_callback is None:
        def progress_callback(progress, current_step, total_steps):
            if total_steps > 0:
                print(f"AI生成进度: {progress:.1%} ({current_step}/{total_steps})")
            else:
                print(f"处理进度: {progress:.1%}")
    
    # 1. 读取原图
    img = cv2.imread(str(img_path))
    ref_img = cv2.imread(str(ref_path))
    if img is None or ref_img is None:
        print(f"无法读取图片: {img_path} 或 {ref_path}")
        return
    
    orig_h, orig_w = img.shape[:2]
    print(f"原图尺寸: {orig_w}x{orig_h}")
    
    # 2. 分析图像质量
    image_quality = analyze_image_quality(img)
    
    # 3. 读取掩码
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取掩码: {mask_path}")
        return
    
    if mask.shape[:2] != (orig_h, orig_w):
        print(f"掩码resize回原图尺寸: {mask.shape[:2]} -> {(orig_h, orig_w)}")
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # 分析掩码复杂度
    mask_complexity = np.sum(mask > 128) / (mask.shape[0] * mask.shape[1])
    print(f"掩码复杂度: {mask_complexity:.2f}")
    
    # 4. 智能调整推理步数
    inference_steps = adaptive_inference_steps(image_quality, user_preference, mask_complexity)
    
    # 5. 优化参数以补偿质量损失
    guidance_scale, controlnet_weights = optimize_parameters_for_steps(inference_steps)
    
    print("掩码归零前像素统计：", np.min(mask), np.max(mask), np.mean(mask))
    background_threshold = 170
    mask[mask < background_threshold] = 0
    mask[mask >= background_threshold] = 255
    print("掩码归零后像素统计：", np.min(mask), np.max(mask), np.mean(mask))
    
    # 可选：先用中值滤波去除毛刺
    mask = cv2.medianBlur(mask, 5)
    maskf = mask / 255.0
    maskf = cv2.GaussianBlur(maskf, (9, 9), 0)
    mask_for_contour = (maskf * 255).astype(np.uint8)
    
    # 调试：保存实际用于融合的掩码
    debug_dir = "data/debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(f"{debug_dir}/{stem}_debug_mask_used.png", (maskf * 255).astype(np.uint8))
    cv2.imwrite(f"{debug_dir}/{stem}_debug_mask_for_contour.png", mask_for_contour)
    
    print(f"maskf min/max/mean: {maskf.min():.3f}/{maskf.max():.3f}/{maskf.mean():.3f}")
    if maskf.mean() > 0.2:
        print("[警告] 掩码均值过高，可能导致全图染色！")
    
    # 6. 统一缩放到目标分辨率
    target_long_side = 1024  # 降低分辨率以提升速度
    long_side = max(orig_h, orig_w)
    if long_side > target_long_side:
        scale = target_long_side / long_side
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        maskf = cv2.resize(maskf, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_for_contour = cv2.resize(mask_for_contour, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ref_img = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"缩放到: {new_w}x{new_h}")
    else:
        new_h, new_w = orig_h, orig_w
        print(f"保持原图尺寸: {new_w}x{new_h}")
    
    # 7. AI高光/质感生成（使用优化参数）
    prompt = (
        "ultra realistic nail, natural nail shape, photorealistic, extremely glossy, mirror-like shine, "
        "strong highlight, glassy, crystal clear, jelly-like, wet look, 3D, curved, visible thickness, natural arc, "
        "photorealistic shadow, natural highlight, soft highlight, subtle gloss, natural reflection, transparent, glass-like, "
        "high gloss, high quality, detailed, studio lighting, cinematic lighting, rim light, specular highlight, reflection, "
        "refraction, translucent, luminous, radiant, thick front edge, visible nail edge, nail tip thickness"
    )
    negative_prompt = (
        "plastic, muddy, waxy, oily, draggy, patchy, sticker, fake, thick, rough, unnatural highlight, cartoon, matte, no shadow, no thickness, blur, low detail, painting, color shift, color change, extra pattern, distortion, artifacts, watermark, text, logo"
    )
    
    print(f"开始AI生成指甲效果 (步数: {inference_steps}, Guidance: {guidance_scale})...")
    ai_img = nail.sdxl_inpaint_controlnet_canny(
        image=img,
        mask=mask_for_contour,
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_strength=0.8,
        callback=progress_callback,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_weights
    )
    torch.cuda.empty_cache()
    
    if ai_img.shape[:2] != (new_h, new_w):
        print(f"调整AI生成图尺寸: {ai_img.shape[:2]} -> {(new_h, new_w)}")
        ai_img = cv2.resize(ai_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 8. Lab空间融合
    color = get_pure_color_from_img(ref_img)
    color_tile = np.ones((new_h, new_w, 3), dtype=np.uint8) * color
    lab_ai = cv2.cvtColor(ai_img, cv2.COLOR_BGR2LAB)
    lab_color = cv2.cvtColor(color_tile, cv2.COLOR_BGR2LAB)
    alpha = 0.3
    L = (lab_ai[..., 0].astype(np.float32) * alpha + lab_color[..., 0].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    ab = lab_color[..., 1:3]
    lab_final = np.dstack([L, ab]).astype(np.uint8)
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    # 9. 掩码融合到原图
    os.makedirs("data/debug", exist_ok=True)
    cv2.imwrite("data/debug/debug_maskf.png", (maskf * 255).astype(np.uint8))
    final = img.copy()
    for c in range(3):
        final[..., c] = img[..., c] * (1 - maskf) + result[..., c] * maskf
    cv2.imwrite("data/debug/debug_result.png", result)
    
    # 10. 最终输出
    final_path = f"{output_final_dir}/{stem}_final.png"
    cv2.imwrite(final_path, final)
    print(f"完成: {final_path}")
    
    # 最终完成时返回100%进度
    if progress_callback:
        progress_callback(1.0, 0, 0)
    
    return final_path

if __name__ == "__main__":
    # 假设每次只处理一对图片
    img_files = sorted(list(Path(input_dir).glob("*.*")))
    ref_files = sorted(list(Path(ref_dir).glob("*.*")))
    
    for img_path, ref_path in zip(img_files, ref_files):
        # 可以指定用户偏好: "speed", "balanced", "quality"
        process_one_optimized(img_path, ref_path, user_preference="balanced") 