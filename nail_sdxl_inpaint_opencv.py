import cv2
import numpy as np
from typing import Optional
import torch
from PIL import Image
from pathlib import Path
from nail_color_transfer import U2NetMasker  # 直接复用你项目里的掩码生成类
import torch, gc
import os
from controlnet_aux import MidasDetector

# 假设u2net_model.py里有U2NET定义
try:
    from u2net_model import U2NET
except ImportError:
    U2NET = None

class NailSDXLInpaintOpenCV:
    """
    综合美甲AI贴图与增强的自动化处理类。

    功能概述：
    1. 自动批量处理手部图片，实现商用级美甲贴图和AI增强。
    2. 掩码生成：集成U²-Net（U2NET）深度学习模型，自动分割手部图片中的指甲区域，输出高精度软掩码（概率图/灰度图），支持边缘羽化，便于后续自然融合。
    3. 颜色迁移/贴图：支持用OpenCV将参考色板/纹理迁移到指甲区域，保证主色调100%还原，可选tile平铺、无缝融合等多种方式。
    4. AI美甲增强：集成Stable Diffusion XL (SDXL) Inpainting模型，对指甲区域进行高质量AI上色、质感和高光增强。
    5. ControlNet结构控制：可选集成ControlNet，自动生成Canny边缘图和深度图，辅助AI模型精确还原指甲形状、结构和立体感。
    6. 掩码融合：支持软掩码（高斯羽化）融合AI生成结果与原图，边缘自然过渡，避免贴片感。
    7. 全流程可调参数：如掩码阈值、羽化宽度、AI增强权重、融合方式等，满足不同商用美甲贴图需求。
    8. 支持多种输出：可输出掩码、颜色迁移中间图、AI增强图、最终融合效果图，便于A/B测试和调优。

    主要用到的技术/模型：
    - U²-Net (U2NET) 指甲分割
    - OpenCV图像处理与无缝融合
    - Stable Diffusion XL (SDXL) Inpainting AI美甲
    - ControlNet（Canny边缘、深度图结构控制）
    - PIL、torch、diffusers、transformers等主流AI/图像处理库

    适用场景：
    - 高精度美甲贴图、AI美甲增强、商用级美甲效果图生成、批量自动化美甲渲染等。
    """
    def __init__(self, sdxl_model_path: str = None, device: str = 'cuda', model_path: str = 'models/u2net_nail_best.pth'):
        """
        初始化美甲渲染器，并集成U2NetMasker作为掩码生成后端。
        :param sdxl_model_path: SDXL Inpaint模型路径
        :param device: 推理设备
        :param model_path: U2Net模型路径
        """
        self.sdxl_model_path = sdxl_model_path
        self.device = device
        self.sdxl_model = self.load_sdxl_model(sdxl_model_path, device) if sdxl_model_path else None
        self.masker = U2NetMasker(model_path=model_path, device=device)

    def load_sdxl_model(self, model_path: str, device: str):
        """
        加载SDXL Inpaint模型（伪代码/接口，需用户实现）
        """
        # TODO: 实现SDXL模型加载
        return None

    def generate_mask_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        用OpenCV处理输入图像生成指甲掩码（示例：简单阈值/颜色分割）
        :param image: 输入BGR图像
        :return: 单通道掩码（0/255）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        # 可根据实际需求替换为更复杂的分割
        return mask

    def seamless_blend(self, src: np.ndarray, dst: np.ndarray, mask: np.ndarray, center: Optional[tuple]=None) -> np.ndarray:
        """
        用OpenCV无缝融合将src贴到dst的mask区域
        :param src: 源图像（如色块/纹理）
        :param dst: 目标图像
        :param mask: 掩码（0/255）
        :param center: 融合中心
        :return: 融合结果
        """
        if center is None:
            h, w = mask.shape
            center = (w//2, h//2)
        blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        return blended

    def sdxl_inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str = "nail art", callback=None) -> np.ndarray:
        from diffusers import StableDiffusionXLInpaintPipeline
        from PIL import Image
        import torch

        def progress_callback(step: int, timestep: int, latents: torch.FloatTensor):
            if callback:
                progress = (step + 1) / num_inference_steps
                callback(progress, step + 1, num_inference_steps)

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 掩码需为单通道PIL灰度，直接用灰度掩码
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_pil = Image.fromarray(mask)

        if callback:
            callback(0.1, 0, 0)

        if not hasattr(self, 'pipe') or self.pipe is None:
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16" if torch.cuda.is_available() else None,
                use_safetensors=True
            ).to(self.device)

        if callback:
            callback(0.2, 0, 0)

        num_inference_steps = 30
        result = self.pipe(
            prompt=prompt,
            negative_prompt="change color, color shift, extra pattern, cartoon, painting, distortion, blur, artifacts, watermark, text, logo, red, dark red, blood, stain, spot",
            image=img_pil,
            mask_image=mask_pil,
            guidance_scale=5.0,
            num_inference_steps=num_inference_steps,
            callback=progress_callback,
            callback_steps=1,
        ).images[0]

        if callback:
            callback(1.0, num_inference_steps, num_inference_steps)

        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result

    def generate_highlight_mask(self, nail_mask, angle=-30, width=0.18, strength=1.0, offset=0.0):
        """
        根据指甲掩码自动生成一条高光条带mask（灰度图）。
        angle: 高光条带方向（度），0为水平，正负为倾斜
        width: 高光条带宽度（归一化到-1~1）
        strength: 高光强度（0~1）
        offset: 高光中心偏移（归一化到-1~1）
        """
        h, w = nail_mask.shape
        yy, xx = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w), indexing='ij')
        angle_rad = np.deg2rad(angle)
        x_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
        band = np.exp(-((x_rot - offset)**2) / (2*width**2))
        band = cv2.GaussianBlur(band, (15,15), 0)
        band = (band / band.max() * 255 * strength).astype(np.uint8)
        # 只在掩码区域内保留
        highlight_mask = np.zeros_like(nail_mask, dtype=np.uint8)
        highlight_mask[nail_mask > 128] = band[nail_mask > 128]
        return highlight_mask

    def sdxl_inpaint_controlnet_canny(self, image: np.ndarray, mask: np.ndarray, prompt: str = None, negative_prompt: str = None, control_strength: float = 1.0, callback=None, target_color: tuple = None) -> np.ndarray:
        """
        全流程灰度掩码的SDXL ControlNet Inpaint实现
        支持颜色引导和颜色校正，确保颜色准确性
        """
        from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
        from PIL import Image
        import torch
        import os
        from controlnet_aux import MidasDetector

        # 进度回调函数 - 只有AI推理阶段才调用
        def progress_callback(step: int, timestep: int, latents: torch.FloatTensor):
            if callback:
                progress = (step + 1) / num_inference_steps
                callback(progress, step + 1, num_inference_steps)

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        sdxl_path = os.path.join(cache_dir, "models--stabilityai--stable-diffusion-xl-base-1.0")
        controlnet_canny_path = os.path.join(cache_dir, "models--diffusers--controlnet-canny-sdxl-1.0")
        controlnet_depth_path = os.path.join(cache_dir, "models--diffusers--controlnet-depth-sdxl-1.0")

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 全流程灰度掩码：直接使用灰度掩码，不进行二值化
        gray_mask = mask.copy()
        
        # 生成ControlNet结构图 - 使用灰度掩码生成更精确的结构图
        try:
            from nail_controlnet_structure_enhancer import create_controlnet_inputs
            controlnet_inputs = create_controlnet_inputs(
                image, gray_mask, 
                structure_methods=['canny', 'depth'],  # 使用Canny和深度结构图
                save_debug=True
            )
            print("已生成ControlNet结构图（基于灰度掩码）")
            
            # 使用生成的结构图
            canny_pil = Image.fromarray(controlnet_inputs['structure_maps']['canny'])
            depth_pil = Image.fromarray(controlnet_inputs['structure_maps']['depth'])
            
        except ImportError:
            print("ControlNet结构增强器不可用，使用传统方法生成边缘图")
            # 基于灰度掩码生成Canny边缘图
            blurred = cv2.GaussianBlur(gray_mask, (3, 3), 0)
            canny = cv2.Canny(blurred, 30, 100)
            kernel = np.ones((2,2), np.uint8)
            canny = cv2.dilate(canny, kernel, iterations=1)
            if canny.shape != image.shape[:2]:
                canny = cv2.resize(canny, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            canny_pil = Image.fromarray(canny)

            # 生成Depth图
            midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth_map = midas(img_rgb)
            if isinstance(depth_map, Image.Image):
                depth_pil = depth_map
            else:
                if depth_map.shape[:2] != image.shape[:2]:
                    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                depth_pil = Image.fromarray(depth_map)

        # 颜色引导的prompt优化
        if target_color is not None:
            # 将RGB颜色转换为颜色名称
            color_name = self._rgb_to_color_name(target_color)
            color_prompt = f"nail with {color_name} color, exact {color_name} shade, precise color matching, "
        else:
            color_prompt = ""
            
        # 强化prompt/negative prompt
        if prompt is None:
            prompt = (
                color_prompt +
                "ultra realistic nail, natural nail shape, photorealistic, glossy, smooth, with highlights and reflections, 3D, natural texture, glossy, highlight, high quality, detailed, "
                "preserve original glitter, keep all texture, keep all glitter details, do not blur, do not repaint, do not smooth, keep all details, ultra sharp, high detail, do not change pattern, no extra patterns, no distortion, "
                "perfect manicure, professional nail art, salon quality, perfect finish, flawless surface, mirror-like shine, crystal clear, transparent, translucent, glass-like finish, "
                "natural lighting, soft shadows, subtle depth, dimensional look, premium quality, luxury finish"
            )
        
        if negative_prompt is None:
            if target_color is not None:
                color_negative = f"wrong color, different color, color change, background color, full image color, {color_name}以外的颜色, "
            else:
                color_negative = ""
                
            negative_prompt = (
                color_negative +
                "blur, smooth, repaint, remove glitter, remove texture, low detail, cartoon, painting, fake, plastic, color shift, color change, extra pattern, distortion, artifacts, watermark, text, logo, red, dark red, blood, stain, spot, "
                "chipped, cracked, uneven, rough, bumpy, textured, grainy, noisy, dirty, dusty, scratched, damaged, imperfect, amateur, low quality, poor finish, dull, cloudy, opaque, milky, frosted, "
                "harsh shadows, strong contrast, overexposed, underexposed, unnatural lighting, artificial look"
            )

        # 全流程灰度掩码：直接使用灰度掩码，不二值化
        mask_pil = Image.fromarray(gray_mask)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not hasattr(self, 'pipe_controlnet') or self.pipe_controlnet is None:
            print("Loading ControlNet Canny+Depth model from local cache...")
            try:
                controlnet_canny = ControlNetModel.from_pretrained(
                    controlnet_canny_path,
                    local_files_only=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
                controlnet_depth = ControlNetModel.from_pretrained(
                    controlnet_depth_path,
                    local_files_only=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
                self.pipe_controlnet = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    sdxl_path,
                    controlnet=[controlnet_canny, controlnet_depth],
                    local_files_only=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
            except Exception as e:
                print(f"Local model loading failed: {e}")
                print("Falling back to downloading from HuggingFace...")
                controlnet_canny = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
                controlnet_depth = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )
                self.pipe_controlnet = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    controlnet=[controlnet_canny, controlnet_depth],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                )

            if torch.cuda.is_available():
                self.pipe_controlnet = self.pipe_controlnet.to("cuda")
            self.pipe_controlnet.safety_checker = None
            print("ControlNet Canny+Depth model loaded successfully!")

        num_inference_steps = 40
        # 开始AI推理 - 只有这个阶段才返回进度
        if callback:
            callback(0.0, 0, num_inference_steps)  # 0% - 开始AI推理

        with torch.no_grad():
            result = self.pipe_controlnet(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,  # 使用灰度掩码，不二值化
                control_image=[canny_pil, depth_pil],
                controlnet_conditioning_scale=[1.2, 0.7],  # Canny控制较强，深度控制适中
                num_inference_steps=num_inference_steps,
                guidance_scale=9.0,  # 提高引导强度确保颜色准确性
                callback=progress_callback,
                callback_steps=1,  # 每步都回调
            ).images[0]

        # AI推理完成
        if callback:
            callback(1.0, num_inference_steps, num_inference_steps)  # 100% - 完成

        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 颜色校正：如果指定了目标颜色，进行后处理颜色校正
        if target_color is not None:
            result_bgr = self._post_process_color_correction(result_bgr, target_color, gray_mask)
            
        return result_bgr

    def _rgb_to_color_name(self, rgb_color: tuple) -> str:
        """
        将RGB颜色转换为颜色名称，用于prompt优化
        """
        r, g, b = rgb_color
        
        # 简单的颜色映射
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 200 and g < 100 and b > 200:
            return "magenta"
        elif r < 100 and g > 200 and b > 200:
            return "cyan"
        elif r > 200 and g > 150 and b > 150:
            return "pink"
        elif r > 150 and g > 100 and b > 200:
            return "purple"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif r > 100 and g > 100 and b > 100:
            return "gray"
        else:
            return f"RGB({r},{g},{b})"

    def _post_process_color_correction(self, result_image: np.ndarray, target_color: tuple, gray_mask: np.ndarray) -> np.ndarray:
        """
        后处理颜色校正，确保颜色相似度达到95%以上
        """
        # 计算当前颜色和目标颜色的差异
        mask_3d = np.stack([gray_mask] * 3, axis=-1).astype(np.float32) / 255.0
        
        # 在掩码区域计算当前平均颜色
        masked_pixels = result_image[mask_3d[:, :, 0] > 0.5]
        if len(masked_pixels) == 0:
            return result_image
            
        current_color = np.mean(masked_pixels, axis=0)
        target_color_array = np.array(target_color)
        
        # 计算颜色差异
        color_diff = target_color_array - current_color
        
        # 应用颜色校正，保持灰度掩码的渐变
        correction_strength = 0.8  # 校正强度
        corrected = result_image.astype(np.float32) + (color_diff * mask_3d * correction_strength)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # 计算校正后的颜色相似度
        corrected_masked_pixels = corrected[mask_3d[:, :, 0] > 0.5]
        if len(corrected_masked_pixels) > 0:
            corrected_color = np.mean(corrected_masked_pixels, axis=0)
            similarity = self._calculate_color_similarity(corrected_color, target_color_array)
            print(f"颜色校正后相似度: {similarity:.2%}")
            
            # 如果相似度还不够，进行进一步校正
            if similarity < 0.95:
                print("进行二次颜色校正...")
                corrected = self._enhanced_color_correction(corrected, target_color_array, gray_mask)
        
        return corrected

    def _calculate_color_similarity(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """
        计算颜色相似度
        """
        color_distance = np.linalg.norm(color1 - color2)
        max_distance = np.sqrt(255**2 * 3)  # 最大可能距离
        similarity = 1.0 - (color_distance / max_distance)
        return similarity

    def _enhanced_color_correction(self, image: np.ndarray, target_color: np.ndarray, gray_mask: np.ndarray) -> np.ndarray:
        """
        增强颜色校正
        """
        mask_3d = np.stack([gray_mask] * 3, axis=-1).astype(np.float32) / 255.0
        
        # 使用更强的校正
        masked_pixels = image[mask_3d[:, :, 0] > 0.5]
        if len(masked_pixels) == 0:
            return image
            
        current_color = np.mean(masked_pixels, axis=0)
        color_diff = target_color - current_color
        
        # 更强的校正强度
        correction_strength = 1.2
        corrected = image.astype(np.float32) + (color_diff * mask_3d * correction_strength)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected

    def resize_to_max1536(self, image: np.ndarray) -> np.ndarray:
        """
        如果输入图像最大边大于1536，则等比例缩放到最长边为1536，短边等比例缩放。
        例如：
        - 原图为3024x4032（竖图），则缩放后为1152x1536。
        - 原图为4000x3000（横图），则缩放后为1536x1152。
        - 原图本身长边≤1536，则保持原图分辨率。
        """
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > 1536:
            scale = 1536 / max_side
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def process_image(self, image: np.ndarray, ref_img: Optional[np.ndarray]=None, prompt: str = "nail art") -> np.ndarray:
        """
        主流程：生成掩码→无缝融合→SDXL Inpaint，AI 处理部分降低分辨率到 512×512，最后用超分恢复清晰度。
        :param image: 输入BGR图像
        :param ref_img: 参考色块/纹理
        :param prompt: SDXL文本提示
        :return: 最终渲染结果
        """
        # 分辨率校验与统一
        image = self.resize_to_max1536(image)
        orig_h, orig_w = image.shape[:2]
        # 1. 生成掩码（改为U2Net）
        mask = self.generate_mask_u2net(image)
        # 掩码生成后，做一次适中高斯模糊
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        # 掩码二值化，保证背景为0
        _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask = (mask * (mask_bin // 255)).astype(np.uint8)
        # 新增：每根指甲左右两侧收窄，并前盖拉长（纵向膨胀）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_shrink = np.zeros_like(mask)
        for cnt in contours:
            single_nail = np.zeros_like(mask)
            cv2.drawContours(single_nail, [cnt], -1, 255, -1)
            # 1. 先做水平方向腐蚀（左右收窄）
            kernel_h = np.ones((1, 21), np.uint8)
            single_nail = cv2.erode(single_nail, kernel_h, iterations=1)
            # 2. 再做纵向膨胀（前端拉长）
            kernel_v = np.ones((15, 1), np.uint8)  # 15x1核，纵向拉长，15可调
            single_nail = cv2.dilate(single_nail, kernel_v, iterations=1)
            mask_shrink = cv2.bitwise_or(mask_shrink, single_nail)
        mask = mask_shrink
        # 最终掩码修正：二值化并裁剪，保证背景为0，指甲为255
        _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask = (mask * (mask_bin // 255)).astype(np.uint8)
        # 2. 贴色/纹理（可选）
        if ref_img is not None:
            ref_resized = cv2.resize(ref_img, (image.shape[1], image.shape[0]))
            blended = self.seamless_blend(ref_resized, image, mask)
        else:
            blended = image.copy()
        # 3. SDXL Inpaint高质量修复/上色（AI 处理部分，分辨率已降至 512×512）
        result = self.sdxl_inpaint(blended, mask, prompt)
        # 还原回原始分辨率（超分恢复清晰度）
        if (result.shape[0], result.shape[1]) != (orig_h, orig_w):
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)  # 超分恢复清晰度
        return result

    def apply_nail_shading(self, texture, mask, light_dir=(0,0,1), view_dir=(0,0,1), shininess=32, specular_strength=0.5):
        """
        对全局贴色区域做法线光照+高光叠加，增强三维感和镜面反射。
        :param texture: BGR贴图（全局）
        :param mask: 单通道掩码（0/255，全局）
        :return: BGR渲染结果（全局）
        """
        h, w = mask.shape
        yy, xx = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w), indexing='ij')
        r = np.sqrt(xx**2 + yy**2)
        z = np.sqrt(np.clip(1 - r**2, 0, 1))
        normals = np.stack([xx, yy, z], axis=2)
        normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
        mask_f = (mask/255.0)[...,None]
        normals = normals * mask_f + np.array([0,0,1]) * (1-mask_f)
        l = np.array(light_dir) / np.linalg.norm(light_dir)
        v = np.array(view_dir) / np.linalg.norm(view_dir)
        hvec = (l + v) / np.linalg.norm(l + v)
        ndotl = np.clip(np.sum(normals * l, axis=2), 0, 1)
        ndoth = np.clip(np.sum(normals * hvec, axis=2), 0, 1)
        color = texture.astype(np.float32) / 255.0
        ambient = 0.25
        diffuse = 0.7 * ndotl[...,None]
        # 镜面高光用纯白
        specular = specular_strength * (ndoth[...,None] ** shininess) * np.array([1,1,1])
        shaded = color * (ambient + diffuse) + specular
        # mask外还原原图
        result = color * (1-mask_f) + shaded * mask_f
        return (result*255).astype(np.uint8)

    def render_nail_curved(self, img, nail_mask, warped, 
                          a=0.7, b=0.3, 
                          highlight_params=[{'angle':-30, 'width':0.18, 'strength':60, 'offset':0.1},
                                           {'angle':30, 'width':0.12, 'strength':40, 'offset':-0.15}],
                          highlight_img_path=None, highlight_img_alpha=0.3):
        h_, w_ = nail_mask.shape
        yy, xx = np.meshgrid(np.linspace(-1,1,h_), np.linspace(-1,1,w_), indexing='ij')
        # 椭球面/多项式曲面
        z = (1 - (a * xx**2 + b * yy**4)).astype(np.float32)
        z = np.clip(z, 0, 1).astype(np.float32)
        # 法线
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=5)
        N = np.dstack((-gx, -gy, np.ones_like(z)))
        N /= np.linalg.norm(N, axis=2, keepdims=True)
        # 光照
        L_dir = np.array([0.5, -1.0, 1.0])
        L_dir = L_dir / np.linalg.norm(L_dir)
        D = np.clip((N @ L_dir), 0, 1)
        # 立体感增强L通道
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        L_base = warped_lab[...,0].astype(np.float32)
        L_shape = L_base.copy()  # 去掉结构性明暗，直接用原亮度
        # 多高光条带
        highlight_mask = np.zeros_like(z)
        for hp in highlight_params:
            angle = np.deg2rad(hp.get('angle', 0))
            width = hp.get('width', 0.15)
            strength = hp.get('strength', 60)
            offset = hp.get('offset', 0.0)
            x_rot = xx * np.cos(angle) + yy * np.sin(angle)
            band = np.exp(-((x_rot - offset)**2) / (2*width**2))
            band = cv2.GaussianBlur(band, (15,15), 0)
            highlight_mask += band * strength
        # 真实高光贴图
        if highlight_img_path:
            hi = cv2.imread(highlight_img_path, cv2.IMREAD_GRAYSCALE)
            if hi is not None:
                hi = cv2.resize(hi, (w_, h_))
                hi = (hi.astype(np.float32)/255.0) * highlight_img_alpha * 255
                highlight_mask += hi
        highlight_mask = highlight_mask * (nail_mask/255.0)
        # 高光加法+亮度基线
        L_highlight = L_shape + highlight_mask
        L_final = np.clip(np.maximum(L_base, L_highlight), 0, 255).astype(np.uint8)
        # ab通道100%还原
        lab_final = warped_lab.copy()
        lab_final[...,0] = L_final
        curved = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        # 羽化融合
        alpha = cv2.GaussianBlur(nail_mask.astype(np.float32), (21,21), 0)[...,None] / 255.0
        result = (img.astype(np.float32)*(1-alpha) + curved*alpha).astype(np.uint8)
        return result

    def generate_color_transfer(self, image: np.ndarray, ref_img_path: str, output_path: str):
        """
        在原图分辨率下进行颜色迁移，自动适配纯色/纹理参考图。
        每个指甲区域单独1:1贴图，中心区域100%硬贴色，边缘羽化融合。
        :param image: 输入BGR图像
        :param ref_img_path: 参考色块/纹理图路径
        :param output_path: 输出路径（可为 .png 或 .jpg）
        """
        h, w = image.shape[:2]
        # 1. 用U2Net生成掩码（原图分辨率）
        mask = self.generate_mask_u2net(image)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # 2. 掩码严格二值化
        _, mask_bin = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        result = image.copy()
        # 3. 读取参考图
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            raise FileNotFoundError(f"参考色块图片不存在: {ref_img_path}")
        # 4. 遍历每个指甲区域，单独1:1贴图
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box < 10 or h_box < 10:
                continue
            nail_mask = np.zeros_like(mask)
            cv2.drawContours(nail_mask, [cnt], -1, 255, -1)
            nail_mask_crop = nail_mask[y:y+h_box, x:x+w_box]
            nail_img_crop = image[y:y+h_box, x:x+w_box]
            # 1:1 resize参考图到指甲区域
            ref_crop = cv2.resize(ref_img, (w_box, h_box), interpolation=cv2.INTER_CUBIC)
            # 中心掩码（腐蚀）
            kernel = np.ones((7,7), np.uint8)
            center_mask = cv2.erode(nail_mask_crop, kernel, iterations=1)
            # 边缘掩码
            edge_mask = cv2.subtract(nail_mask_crop, center_mask)
            # 结果crop
            nail_result = nail_img_crop.copy()
            nail_result[center_mask == 255] = ref_crop[center_mask == 255]
            # 边缘羽化
            edge_alpha = cv2.GaussianBlur(edge_mask, (15,15), 0) / 255.0
            nail_result = (nail_result * (1 - edge_alpha[...,None]) + ref_crop * edge_alpha[...,None]).astype(np.uint8)
            # 合成回全图
            result[y:y+h_box, x:x+w_box][nail_mask_crop > 0] = nail_result[nail_mask_crop > 0]
        # 5. 保存
        ext = output_path.lower().split('.')[-1]
        if ext == 'jpg' or ext == 'jpeg':
            cv2.imwrite(output_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(output_path, result)

    def generate_mask_u2net(self, image: np.ndarray, image_path: str = "tmp.png") -> np.ndarray:
        """
        用U2NetMasker生成掩码，保证和权重、后处理完全兼容。
        直接输出灰度掩码（不做二值化），从而减少锯齿，指甲融合效果更自然。
        :param image: 输入BGR图像
        :param image_path: 用于掩码缓存/命名（可随意）
        :return: 单通道掩码（0～255，灰度掩码）
        """
        # 不再缩放，直接用原图分辨率，禁用缓存确保每次都重新生成
        mask = self.masker.get_mask(image, image_path, disable_cache=True)
        # 如果mask是概率掩码（0～1），则乘以255，得到灰度掩码（0～255）
        if mask.dtype != np.uint8 or mask.max() <= 1.0:
             mask = (mask * 255).astype(np.uint8)
        # 确保掩码尺寸与输入图像一致，用线性插值避免锯齿
        if mask.shape[:2] != image.shape[:2]:
             mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 中期方案：应用Active Contour增强来弥补U2Net模型分割的不足
        try:
            # 首先尝试使用Active Contour增强
            from nail_active_contour_enhancer import enhance_with_active_contour_simple
            enhanced_mask = enhance_with_active_contour_simple(
                mask, 
                image, 
                iterations=30,      # 适中的迭代次数，平衡效果和速度
                edge_expansion=4,   # 边缘扩展4像素，保留皮肤-指甲过渡区域
                feather_width=6     # 羽化宽度6像素，创建自然的边缘过渡
            )
            print("使用Active Contour增强掩码（中期方案）")
            
            # 新增：边缘引导增强
            try:
                from nail_edge_guided_enhancer import enhance_mask_with_edge_guidance
                edge_guided_mask = enhance_mask_with_edge_guidance(
                    enhanced_mask,
                    image,
                    method='combined',  # 使用Canny+Sobel组合方法
                    edge_weight=0.7,    # 边缘权重70%
                    smooth_factor=0.8,  # 平滑因子80%
                    save_debug=True
                )
                print("应用边缘引导增强（高级方案）")
                return edge_guided_mask
                
            except ImportError:
                print("边缘引导模块不可用，跳过边缘引导步骤")
                return enhanced_mask
            
        except ImportError:
            try:
                # 如果Active Contour不可用，尝试使用指甲形状优化器
                from nail_shape_optimizer import optimize_nail_mask_simple
                enhanced_mask = optimize_nail_mask_simple(
                    mask, 
                    image, 
                    edge_expansion=4,    # 边缘扩展4像素
                    front_extension=6,   # 前端延长6像素
                    feather_width=7      # 羽化宽度7像素
                )
                print("使用指甲形状优化器增强掩码（最快上手方案）")
                
                # 新增：边缘引导增强
                try:
                    from nail_edge_guided_enhancer import enhance_mask_with_edge_guidance
                    edge_guided_mask = enhance_mask_with_edge_guidance(
                        enhanced_mask,
                        image,
                        method='canny',   # 使用Canny边缘检测
                        edge_weight=0.6,  # 边缘权重60%
                        smooth_factor=0.7, # 平滑因子70%
                        save_debug=True
                    )
                    print("应用边缘引导增强（高级方案）")
                    return edge_guided_mask
                    
                except ImportError:
                    print("边缘引导模块不可用，跳过边缘引导步骤")
                    return enhanced_mask
                
            except ImportError:
                try:
                    # 如果指甲形状优化器不可用，尝试使用基础掩码增强
                    from nail_mask_enhancer import enhance_nail_mask_simple
                    enhanced_mask = enhance_nail_mask_simple(
                        mask, 
                        image, 
                        edge_expansion=3,  # 边缘扩展3像素
                        feather_width=5    # 羽化宽度5像素
                    )
                    print("使用基础掩码增强器增强掩码（基础方案）")
                    
                    # 新增：边缘引导增强
                    try:
                        from nail_edge_guided_enhancer import enhance_mask_with_edge_guidance
                        edge_guided_mask = enhance_mask_with_edge_guidance(
                            enhanced_mask,
                            image,
                            method='sobel',  # 使用Sobel边缘检测
                            edge_weight=0.5, # 边缘权重50%
                            smooth_factor=0.6, # 平滑因子60%
                            save_debug=True
                        )
                        print("应用边缘引导增强（高级方案）")
                        return edge_guided_mask
                        
                    except ImportError:
                        print("边缘引导模块不可用，跳过边缘引导步骤")
                        return enhanced_mask
                    
                except ImportError:
                    # 如果所有增强模块都不可用，使用原始掩码
                    print("警告：所有掩码增强模块都不可用，使用原始掩码")
                    return mask

    def clean_up(self):
        del self.sdxl_model
        del self.masker
        gc.collect()
        torch.cuda.empty_cache()

    def lab_l_from_ai_ab_from_color(self, ai_img, color_img):
        """
        用AI增强图的L通道（高光/立体）+ 颜色迁移图的ab通道（色彩/颗粒）融合，返回BGR图像。
        颜色100%锁定为迁移图，AI只负责高光和质感。
        """
        # 保证尺寸一致
        if ai_img.shape[:2] != color_img.shape[:2]:
            ai_img = cv2.resize(ai_img, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        lab_ai = cv2.cvtColor(ai_img, cv2.COLOR_BGR2LAB)
        lab_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        # 只用AI的L通道，ab通道100%用颜色迁移图
        lab_final = np.zeros_like(lab_ai)
        lab_final[..., 0] = lab_ai[..., 0]
        lab_final[..., 1] = lab_color[..., 1]
        lab_final[..., 2] = lab_color[..., 2]
        return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    def lab_l_add_highlight_from_ai(self, ai_img, color_img, highlight_weight=0.3):
        """
        只加AI增强图的高光部分，不漂色。原色迁移图100%保留，AI只负责高光和质感。
        :param ai_img: AI增强BGR图
        :param color_img: 颜色迁移BGR图
        :param highlight_weight: 高光权重（0~1）
        :return: BGR图像
        """
        if ai_img.shape[:2] != color_img.shape[:2]:
            ai_img = cv2.resize(ai_img, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        lab_ai = cv2.cvtColor(ai_img, cv2.COLOR_BGR2LAB)
        lab_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        l_ai = lab_ai[..., 0].astype(np.float32)
        l_color = lab_color[..., 0].astype(np.float32)
        # 只加高光部分
        highlight = np.clip(l_ai - l_color, 0, None)
        l_final = np.clip(l_color + highlight * highlight_weight, 0, 255).astype(np.uint8)
        lab_final = np.zeros_like(lab_ai)
        lab_final[..., 0] = l_final
        lab_final[..., 1] = lab_color[..., 1]
        lab_final[..., 2] = lab_color[..., 2]
        return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    def process_with_ai_fusion(self, color_transfer_img: np.ndarray, mask: np.ndarray, prompt: str = None, target_color: tuple = None) -> np.ndarray:
        """
        全流程灰度掩码的AI融合处理
        支持颜色引导，确保颜色准确性
        """
        # 保证AI inpaint输入分辨率为1536像素长边
        h, w = color_transfer_img.shape[:2]
        if max(h, w) != 1536:
            scale = 1536 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            color_transfer_img = cv2.resize(color_transfer_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 全流程灰度掩码：直接使用灰度掩码，不进行二值化
        gray_mask = mask.copy()
        
        # 强化prompt/negative prompt
        strong_prompt = (
            "ultra realistic nail, natural nail shape, photorealistic, glossy, smooth, with highlights and reflections, 3D, natural texture, glossy, highlight, high quality, detailed, "
            "preserve original glitter, keep all texture, keep all glitter details, do not blur, do not repaint, do not smooth, keep all details, ultra sharp, high detail, do not change pattern, no extra patterns, no distortion, "
            "perfect manicure, professional nail art, salon quality, perfect finish, flawless surface, mirror-like shine, crystal clear, transparent, translucent, glass-like finish, "
            "natural lighting, soft shadows, subtle depth, dimensional look, premium quality, luxury finish"
        )
        strong_negative = (
            "blur, smooth, repaint, remove glitter, remove texture, low detail, cartoon, painting, fake, plastic, color shift, color change, extra pattern, distortion, artifacts, watermark, text, logo, red, dark red, blood, stain, spot, "
            "chipped, cracked, uneven, rough, bumpy, textured, grainy, noisy, dirty, dusty, scratched, damaged, imperfect, amateur, low quality, poor finish, dull, cloudy, opaque, milky, frosted, "
            "harsh shadows, strong contrast, overexposed, underexposed, unnatural lighting, artificial look"
        )
        
        final = self._process_with_ai_fusion_core(color_transfer_img, gray_mask, strong_prompt, strong_negative, target_color)
        return final

    def _process_with_ai_fusion_core(self, color_transfer_img, gray_mask, prompt, negative_prompt, target_color=None):
        """
        全流程灰度掩码的AI融合核心流程
        """
        # 使用全流程灰度掩码进行AI inpaint
        ai_enhanced = self.sdxl_inpaint_controlnet_canny(
            image=color_transfer_img,
            mask=gray_mask,  # 直接使用灰度掩码
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_strength=0.8,
            target_color=target_color  # 传递目标颜色
        )
        
        # 全流程灰度掩码融合：使用灰度掩码进行融合，不二值化
        if hasattr(self, 'u2net_mask_for_clip'):
            u2net_mask = self.u2net_mask_for_clip
        else:
            u2net_mask = gray_mask
        
        # 使用灰度掩码进行融合，保持渐变效果
        u2net_mask_3c = cv2.merge([u2net_mask/255.0]*3)
        base = color_transfer_img.copy()
        
        # 全流程灰度掩码：直接使用灰度掩码进行融合，不进行二值化处理
        ai_mask_soft = gray_mask.astype(np.float32) / 255.0
        ai_mask_soft = cv2.GaussianBlur(ai_mask_soft, (15,15), 0)  # 轻微平滑
        ai_mask_soft = np.clip(ai_mask_soft ** 1.2, 0, 1)  # 轻微增强对比度
        ai_mask_soft_3c = np.repeat(ai_mask_soft[..., None], 3, axis=2)
        
        # 使用AI增强的高光效果
        ai_highlight = self.lab_l_add_highlight_from_ai(ai_enhanced, base, highlight_weight=0.3)
        
        # 全流程灰度掩码融合：使用灰度掩码的渐变进行自然融合
        final = (base * (1 - ai_mask_soft_3c) + ai_highlight * ai_mask_soft_3c).astype(np.uint8)
        
        # 后处理
        final = self.postprocess_nail_highlight_curvature(final, gray_mask)
        return final

    def postprocess_nail_highlight_curvature(self, img, mask):
        # 实现postprocess_nail_highlight_curvature方法
        # 这里需要根据你的具体需求实现这个方法
        return img

print(cv2.imread('data/output/color_transfer/11111.png') is not None)
print(cv2.imread('data/output/masks/11111_mask.png', 0) is not None)
print(cv2.imread('data/test_images/11111.png') is not None) 