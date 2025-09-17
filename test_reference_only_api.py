import requests
import base64
import os
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

# 使用 data/output/final/120745430_pixel_transplant_auto.png 作为主图和参考图
img_path = "data/output/final/120745430.png"
def img2b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 获取参考图尺寸
with Image.open(img_path) as im:
    width, height = im.size

init_image = img2b64(img_path)         # 主图
reference_image = img2b64(img_path)    # 参考图（和主图相同）

# 根据原图像名自动查找掩码（目录改为 data/test_masks/）
img_stem, _ = os.path.splitext(os.path.basename(img_path))
mask_path = os.path.join("data/test_masks", f"{img_stem}_mask.png")

def process_mask(mask_path, target_size, use_binary=False):
    """
    读取灰度掩码，可选择二值化或保持灰度，输出PIL格式，尺寸与目标一致。
    
    Args:
        mask_path: 掩码文件路径
        target_size: 目标尺寸 (width, height)
        use_binary: 是否二值化，False=保持灰度（推荐），True=二值化
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码: {mask_path}")
    
    if use_binary:
        # 二值化处理（传统方式）
        print("使用二值化掩码...")
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 面积过滤（去除小区域）
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask_bin)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # 羽化
        feathered = cv2.GaussianBlur(clean_mask, (15, 15), 0)
    else:
        # 灰度掩码处理（推荐方式，减少锯齿，融合更自然）
        print("使用灰度掩码...")
        # 轻微高斯模糊，减少噪点但保持渐变
        feathered = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 尺寸校正
    feathered = cv2.resize(feathered, target_size, interpolation=cv2.INTER_LINEAR)
    # 转为PIL
    pil_mask = Image.fromarray(feathered)
    return pil_mask

# 获取掩码并转base64（默认使用灰度掩码，效果更好）
mask_pil = process_mask(mask_path, (width, height), use_binary=False)  # 保持原有调用方式
buf = BytesIO()
mask_pil.save(buf, format="PNG")
mask_b64 = base64.b64encode(buf.getvalue()).decode()

# 恢复LoRA支持
lora1 = "Mastering_Manicure_A_Visual_Guide_to_Nail_Art_Techniques"
lora2 = "Stiletto_Nails"
lora1_weight = 1.0
lora2_weight = 1.0

prompt_str = (
    f"<lora:{lora1}:{lora1_weight}> <lora:{lora2}:{lora2_weight}> "
    "nail art, stiletto nails, long sharp tip, professional manicure, natural curved nail shape, arched nail surface, 3D, convex, "
    "realistic nail arch, natural nail arch, photorealistic, ultra glossy, mirror-like, highly polished, waxed finish, car wax shine, "
    "mirror finish, crystal clear reflection, smooth as glass, wet look, high gloss, shiny surface, reflective, high detail, "
    "natural highlights and shadows, no color shift, no pattern change, no sticker effect, no flatness, no plastic look, no AI artifacts, "
    "glossy 3D highlights and reflections"
)

negative_prompt = (
    "grainy, rough, artifact, noise, sandpaper, matte, dull, low detail, color shift, color change, overexposed, too bright, "
    "full-nail highlight, flat, 2d, no curvature, unnatural flatness, blocky, rigid, no arch, sticker, painting, cartoon, illustration, "
    "fake, ai, smooth, plastic, doll, no shadow, no highlight, unnatural, out of focus, overprocessed"
)

payload = {
    "init_images": [init_image],
    "mask": mask_b64,
    "mask_blur": 8,
    "inpainting_fill": 1,  # 1=原图保持，2=全白，3=全黑
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 32,
    "prompt": prompt_str,
    "negative_prompt": negative_prompt,
    "steps": 30,
    "width": width,
    "height": height,
    "sampler_name": "Euler a",
    "cfg_scale": 7,
    "denoising_strength": 0.3,  # 最大值，极限增强
    "seed": 123456789,  # 固定seed，保证结果可复现
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "input_image": reference_image,
                    "module": "reference_only",
                    "model": "None",
                    "weight": 1.0,
                    "resize_mode": "Just Resize",
                    "control_mode": "Balanced",
                }
            ]
        }
    },
    "script_name": None,  # 明确不使用额外脚本
    "batch_size": 1,
    "mode": "inpaint"
}

# ========== 第一阶段：reference_only生成中间图 ==========
response = requests.post(
    "http://127.0.0.1:7860/sdapi/v1/img2img",
    json=payload
)
try:
    result = response.json()
    output_dir = "data/output/debug"
    os.makedirs(output_dir, exist_ok=True)
    if 'images' in result and len(result['images']) > 0:
        out_path = os.path.join(output_dir, f"result_0.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result['images'][0]))
        print(f"reference_only生成成品图已保存到 {out_path}")
    else:
        print("未检测到图片，返回内容：", result)
except Exception as e:
    print("API返回内容无法解析为JSON，原始内容：")
    print(response.text)

# --- 新增：测试API服务/docs接口可用性 ---
# （本段已移除，不再打印/docs接口相关内容） 