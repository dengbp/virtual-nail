import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import os
# V44: 切换到IP-Adapter方案，并从本地加载模型
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
import time
import sys
import argparse
from diffusers import StableDiffusionXLPipeline
import requests
import base64
from io import BytesIO
import threading
from datetime import datetime

# WebUI 进度监控工具类
class WebUIProgressMonitor:
    """WebUI 进度监控工具类"""
    
    @staticmethod
    def get_progress_via_http():
        """通过 HTTP 接口获取进度"""
        try:
            response = requests.get("http://127.0.0.1:7860/sdapi/v1/progress", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"HTTP进度获取失败: {e}")
        return None
    
    @staticmethod
    def get_queue_status():
        """获取队列状态"""
        try:
            response = requests.get("http://127.0.0.1:7860/sdapi/v1/queue", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"队列状态获取失败: {e}")
        return None
    
    @staticmethod
    def get_task_info():
        """获取当前任务信息"""
        try:
            response = requests.get("http://127.0.0.1:7860/sdapi/v1/queue", timeout=5)
            if response.status_code == 200:
                queue_data = response.json()
                return {
                    'queue_running': queue_data.get('queue_running', []),
                    'queue_pending': queue_data.get('queue_pending', []),
                    'queue_history': queue_data.get('queue_history', [])
                }
        except Exception as e:
            print(f"任务信息获取失败: {e}")
        return None
    
    @staticmethod
    def find_task_by_prompt(task_info, target_prompt_keywords=None, task_id=None):
        """根据提示词关键词或任务ID查找特定任务"""
        if not task_info:
            return None
            
        # 搜索正在运行的任务
        for task in task_info.get('queue_running', []):
            if task_id and task.get('id') == task_id:
                return {'status': 'running', 'task': task}
            
            if target_prompt_keywords:
                prompt = task.get('prompt', '')
                if any(keyword.lower() in prompt.lower() for keyword in target_prompt_keywords):
                    return {'status': 'running', 'task': task}
        
        # 搜索等待中的任务
        for task in task_info.get('queue_pending', []):
            if task_id and task.get('id') == task_id:
                return {'status': 'pending', 'task': task}
            
            if target_prompt_keywords:
                prompt = task.get('prompt', '')
                if any(keyword.lower() in prompt.lower() for keyword in target_prompt_keywords):
                    return {'status': 'pending', 'task': task}
        
        # 搜索历史任务
        for task in task_info.get('queue_history', []):
            if task_id and task.get('id') == task_id:
                return {'status': 'completed', 'task': task}
            
            if target_prompt_keywords:
                prompt = task.get('prompt', '')
                if any(keyword.lower() in prompt.lower() for keyword in target_prompt_keywords):
                    return {'status': 'completed', 'task': task}
        
        return None
    
    @staticmethod
    def monitor_specific_task(task_identifier, callback=None, timeout=300):
        """
        监控特定任务的进度
        task_identifier: 可以是任务ID或提示词关键词列表
        """
        start_time = time.time()
        task_found = False
        
        while time.time() - start_time < timeout:
            try:
                # 获取任务信息
                task_info = WebUIProgressMonitor.get_task_info()
                if not task_info:
                    time.sleep(2)
                    continue
                
                # 查找特定任务
                if isinstance(task_identifier, str):
                    # 按任务ID查找
                    task_result = WebUIProgressMonitor.find_task_by_prompt(task_info, task_id=task_identifier)
                else:
                    # 按提示词关键词查找
                    task_result = WebUIProgressMonitor.find_task_by_prompt(task_info, target_prompt_keywords=task_identifier)
                
                if task_result:
                    task_found = True
                    status = task_result['status']
                    task = task_result['task']
                    
                    if status == 'running':
                        # 获取详细进度
                        progress_data = WebUIProgressMonitor.get_progress_via_http()
                        if progress_data:
                            state = progress_data.get('state', {})
                            current_step = state.get('sampling_step', 0)
                            total_steps = state.get('sampling_steps', 30)
                            
                            if total_steps > 0:
                                progress_percent = (current_step / total_steps) * 100
                                status_msg = f"任务进度: {current_step}/{total_steps} ({progress_percent:.1f}%)"
                                print(status_msg)
                                
                                if callback:
                                    callback(current_step, total_steps, progress_percent, task)
                                
                                # 检查是否完成
                                if current_step >= total_steps:
                                    print(f"任务 {task.get('id', 'unknown')} 已完成！")
                                    return True
                    
                    elif status == 'pending':
                        print(f"任务 {task.get('id', 'unknown')} 正在等待中...")
                    
                    elif status == 'completed':
                        print(f"任务 {task.get('id', 'unknown')} 已完成！")
                        return True
                
                else:
                    if task_found:
                        print("任务已完成或已从队列中移除")
                        return True
                    else:
                        print("未找到指定任务，继续搜索...")
                
                time.sleep(2)  # 每2秒检查一次
                
            except Exception as e:
                print(f"任务监控出错: {e}")
                time.sleep(2)
        
        print(f"任务监控超时 ({timeout}秒)")
        return False
    
    @staticmethod
    def monitor_progress_http(callback=None):
        """HTTP方式监控进度"""
        while True:
            progress_data = WebUIProgressMonitor.get_progress_via_http()
            if progress_data:
                state = progress_data.get('state', {})
                current_step = state.get('sampling_step', 0)
                total_steps = state.get('sampling_steps', 30)
                job = state.get('job', '')
                
                if current_step > 0 and total_steps > 0:
                    progress_percent = (current_step / total_steps) * 100
                    status_msg = f"AI精炼进度: {current_step}/{total_steps} ({progress_percent:.1f}%)"
                    print(status_msg)
                    
                    if callback:
                        callback(current_step, total_steps, progress_percent)
                    
                    # 如果处理完成，退出监控
                    if job == '':
                        print("AI精炼处理完成！")
                        break
                elif job == '':
                    # 没有任务在运行
                    break
            
            time.sleep(1)  # 每秒检查一次
    
    @staticmethod
    def monitor_progress_websocket(callback=None):
        """WebSocket方式监控进度（需要安装websocket-client）"""
        try:
            import websocket
            import json
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if 'type' in data and data['type'] == 'progress':
                        current_step = data.get('data', {}).get('value', 0)
                        total_steps = data.get('data', {}).get('max', 30)
                        if total_steps > 0:
                            progress_percent = (current_step / total_steps) * 100
                            status_msg = f"AI精炼进度: {current_step}/{total_steps} ({progress_percent:.1f}%)"
                            print(status_msg)
                            
                            if callback:
                                callback(current_step, total_steps, progress_percent)
                except Exception as e:
                    print(f"WebSocket消息解析失败: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket错误: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("WebSocket连接关闭")
            
            def on_open(ws):
                print("WebSocket连接已建立，开始监控进度...")
            
            # 连接WebUI的WebSocket
            ws = websocket.WebSocketApp(
                "ws://127.0.0.1:7860/queue/join",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
            
        except ImportError:
            print("WebSocket监控需要安装 websocket-client: pip install websocket-client")
            print("回退到HTTP监控方式...")
            WebUIProgressMonitor.monitor_progress_http(callback)
        except Exception as e:
            print(f"WebSocket监控失败: {e}")
            print("回退到HTTP监控方式...")
            WebUIProgressMonitor.monitor_progress_http(callback)

class SDXLRefiner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()

    def _load_models(self):
        print("开始加载SDXL + IP-Adapter精炼模型 (V52 - safetensors 优先)...")
        start_time = time.time()
        
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # V52: 优先加载 safetensors，并让库自动处理缓存
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)

        # V62: 降级 transformers 后，使用最终被证实的本地 .pth 文件路径
        local_snapshots_path = "C:/Users/Administrator/.cache/huggingface/hub/models--h94--IP-Adapter/snapshots"
        
        self.pipe.load_ip_adapter(local_snapshots_path, subfolder="local-xl", weight_name="ip-adapter_xl.pth")
        self.pipe.set_ip_adapter_scale(0.7)

        print(f"模型加载完成，使用降级策略，耗时: {time.time() - start_time:.2f}秒")

    def refine_image(self, image_path: str, mask_path: str):
        """
        (V44 - 本地IP-Adapter) - 使用我们手动下载的模型进行高保真风格迁移。
        """
        print(f"\n--- [AI精炼阶段 V44 - 本地IP-Adapter] 开始处理: {image_path} ---")
        
        pixel_transplant_img = cv2.imread(image_path)
        if pixel_transplant_img is None:
            print(f"错误: 无法读取像素迁移图像 {image_path}")
            return None, None
        
        # 统一分辨率
        h0, w0, _ = pixel_transplant_img.shape
        max_side = max(h0, w0)
        if max_side > 1024:
            new_h, new_w = (1024, int(w0 * 1024 / h0)) if h0 >= w0 else (int(h0 * 1024 / w0), 1024)
        else:
            new_h, new_w = h0, w0
        
        pixel_transplant_img_resized = cv2.resize(pixel_transplant_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        pixel_transplant_pil = Image.fromarray(cv2.cvtColor(pixel_transplant_img_resized, cv2.COLOR_BGR2RGB))
        
        debug_dir = Path("data/debug")
        debug_dir.mkdir(exist_ok=True)
        
        # 加载并处理掩码
        final_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if final_mask is None:
            print(f"错误: 无法从 {mask_path} 加载权威掩码")
            return None, None
        final_mask_resized = cv2.resize(final_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_pil = Image.fromarray(final_mask_resized)
        
        # 注意：在底层加载方式中，我们不再需要set_ip_adapter_scale

        prompt = (
            "Ultra-realistic, professional nail art photography. "
            "Extremely high-quality gel polish with a glossy, wet-look finish. "
            "Perfect 3D curvature, seamless cuticle integration, natural growth appearance. "
            "Sharp studio lighting, vibrant colors, intricate details preserved."
        )
        negative_prompt = "blurry, noisy, flat, matte, dull, deformed hand, bad lighting, color shift, pattern change, washed out, ugly"
        
        refined_pil = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pixel_transplant_pil,
            mask_image=mask_pil,
            ip_adapter_image=pixel_transplant_pil,
            strength=0.8,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        refined_bgr = cv2.cvtColor(np.array(refined_pil), cv2.COLOR_RGB2BGR)

        # 最终合成
        output_image = pixel_transplant_img_resized.copy()
        output_image[final_mask_resized == 255] = refined_bgr[final_mask_resized == 255]
        
        # 保存结果
        input_stem = Path(image_path).stem
        output_dir = Path("data/output/final_refined")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"{input_stem}_refined_v46.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), output_image)
        
        # 保存调试图
        comparison = np.hstack([pixel_transplant_img_resized, refined_bgr, output_image])
        cv2.imwrite(str(debug_dir / f"{Path(input_stem)}_v46_comparison.png"), comparison)
        
        print(f"\nIP-Adapter (底层加载) Control完成！结果已保存至: {output_path}")
        print("V46技术方案：底层IP-Adapter手动加载")
        return str(output_path), final_mask_resized

def process_refine(img_path, mask_path):
    """
    一个辅助函数，封装了精炼过程的主要逻辑。
    现在由 API 服务器直接调用。
    """
    start_time = time.time()
    
    # 注意：这里的 refiner 实例将由调用方（API服务器）传入或管理
    # 为了保持此文件的模块化，我们假设 refiner 已被正确初始化
    # 但在这个文件的独立版本中，我们需要自己初始化
    try:
        from refine_api_server import refiner_instance
    except (ImportError, NameError):
        print("警告: 无法从API服务器导入全局refiner实例。将创建一个新的实例。")
        refiner_instance = SDXLRefiner()

    refined_img_path, _ = refiner_instance.refine_image(img_path, mask_path)
    
    if refined_img_path:
        # 重命名文件，准备最终输出
        final_output_path = Path(refined_img_path).with_name(f"{Path(img_path).stem}_final.png")
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        os.rename(refined_img_path, final_output_path)
        
        print(f"\n--- 精炼流程结束 ---")
        print(f"结果已保存: {final_output_path}")
        print(f"耗时: {time.time() - start_time:.2f}秒")
        
        return str(final_output_path)
    else:
        print("AI精炼步骤失败，流程终止。")
        return None

# V49: 移除 __main__ 模块。
# 这个脚本现在作为一个模块被 API 服务器导入，不再需要独立执行。

# V63: 添加一个临时的 main 块，用于独立测试模型加载
if __name__ == '__main__':
    print("--- [独立测试] 开始执行模型加载 ---")
    try:
        # 确保调试时能看到所有日志
        import logging
        logging.basicConfig(level=logging.INFO)
        
        refiner = SDXLRefiner()
        print("--- [独立测试] 模型加载成功 ---")
    except Exception as e:
        print(f"--- [独立测试] 模型加载失败 ---")
        import traceback
        traceback.print_exc()

# ========== 主流程入口函数 ========== 
def refine_sdxl_pipeline(orig_img, orig_stem):
    """
    主流程入口：根据输入图片路径（可以是原图、像素迁移图或补光后图），自动推导掩码路径，完成精炼。
    参数：
        orig_img: PIL.Image.Image，输入PIL图片对象
        orig_stem: str，输入图片的原始文件名（不包括扩展名）
    返回：
        out_path: str，最终精炼成品图路径
    """
    from pathlib import Path
    import os
    import base64
    from PIL import Image
    import requests
    from io import BytesIO
    import numpy as np
    import cv2
    import time

    # 1. 读取输入图，确定目标尺寸
    if isinstance(orig_img, Image.Image):
        img = orig_img
        tmp_input_path = 'data/output/debug/_tmp_refine_input.png'
        img.save(tmp_input_path)
        orig_img_path = tmp_input_path
    else:
        img = Image.open(orig_img)
        orig_img_path = orig_img
    orig_w, orig_h = img.size
    max_side = max(orig_w, orig_h)
    if max_side > 1024:
        if orig_w >= orig_h:
            target_w, target_h = 1024, int(orig_h * 1024 / orig_w)
        else:
            target_w, target_h = int(orig_w * 1024 / orig_h), 1024
    else:
        target_w, target_h = orig_w, orig_h

    # 2. 用orig_stem拼掩码路径
    mask_path = f"data/test_masks/{orig_stem}_mask.png"
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
    def process_mask(mask_path, target_size, use_binary=False, mask_quality='high'):
        """
        处理掩码，支持多种质量模式
        
        Args:
            mask_path: 掩码文件路径
            target_size: 目标尺寸 (width, height)
            use_binary: 是否使用二值化
            mask_quality: 掩码质量模式 ('high', 'medium', 'low')
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {mask_path}")
        
        target_w, target_h = target_size
        
        if use_binary:
            # 二值化处理（不推荐，但保留选项）
            _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.OTSU)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(mask_bin)
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            # 二值化需要更强的模糊
            feathered = cv2.GaussianBlur(clean_mask, (15, 15), 0)
        else:
            # 灰度掩码处理（推荐）
            if mask_quality == 'high':
                # 高质量：保留更多细节，轻微模糊
                feathered = cv2.GaussianBlur(mask, (3, 3), 0)
            elif mask_quality == 'medium':
                # 中等质量：平衡细节和平滑
                feathered = cv2.GaussianBlur(mask, (5, 5), 0)
            else:  # 'low'
                # 低质量：更强模糊，更平滑
                feathered = cv2.GaussianBlur(mask, (7, 7), 0)
        
        # 调整到目标尺寸
        feathered = cv2.resize(feathered, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pil_mask = Image.fromarray(feathered)
        return pil_mask
    mask_pil = process_mask(mask_path, (target_w, target_h), use_binary=False, mask_quality='high')
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # 3. 组装prompt（保持原有逻辑）
    lora1 = "Mastering_Manicure_A_Visual_Guide_to_Nail_Art_Techniques"
    lora2 = "Stiletto_Nails"
    lora1_weight = 0.8  # 降低权重，减少过度风格化
    lora2_weight = 1.2  # 增强尖头指甲效果
    prompt_str = (
        f"<lora:{lora1}:{lora1_weight}> <lora:{lora2}:{lora2_weight}> "
        "nail art, stiletto nails, long sharp tip, professional manicure, natural curved nail shape, arched nail surface, 3D, convex, "
        "realistic nail arch, natural nail arch, photorealistic, ultra glossy, mirror-like, highly polished, waxed finish, car wax shine, "
        "mirror finish, crystal clear reflection, smooth as glass, wet look, high gloss, shiny surface, reflective, high detail, "
        "natural highlights and shadows, glossy, mirror, specular highlight, no color shift, no pattern change, no sticker effect, no flatness, no plastic look, no AI artifacts, "
        "glossy 3D highlights and reflections, "
        "perfect nail curvature, natural nail growth direction, proper nail length, realistic nail tip shape, "
        "natural nail bed shape, proper nail arch, realistic nail thickness, natural nail edge, "
        "professional nail salon quality, perfect nail form, natural nail structure"
    )
    negative_prompt = (
        "grainy, rough, artifact, noise, sandpaper, matte, dull, low detail, color shift, color change, overexposed, too bright, "
        "full-nail highlight, flat, 2d, no curvature, unnatural flatness, blocky, rigid, no arch, sticker, painting, cartoon, illustration, "
        "fake, ai, smooth, plastic, doll, no shadow, no highlight, unnatural, out of focus, overprocessed, "
        "wrong nail shape, deformed nail, unnatural nail length, flat nail surface, no nail arch, "
        "painted on effect, sticker effect, artificial nail shape, wrong nail direction"
    )
    def img2b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    init_image = img2b64(orig_img_path)
    reference_image = img2b64(orig_img_path)

    # 生成Canny边缘图用于形状控制
    def generate_canny_control(image_path, target_size):
        """生成Canny边缘图用于形状控制"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 调整尺寸
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # 转换为灰度
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        canny = cv2.Canny(blurred, 50, 150)
        
        # 轻微膨胀以增强边缘
        kernel = np.ones((2, 2), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=1)
        
        # 转换为PIL并编码
        canny_pil = Image.fromarray(canny)
        buf = BytesIO()
        canny_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # 生成Canny控制图
    canny_control = generate_canny_control(orig_img_path, (target_w, target_h))

    payload = {
        "init_images": [init_image],
        "mask": mask_b64,
        "mask_blur": 8,
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "prompt": prompt_str,
        "negative_prompt": negative_prompt,
        "steps": 35,
        "width": target_w,
        "height": target_h,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 8.5,
        "denoising_strength": 0.4,
        "seed": 123456789,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "input_image": reference_image,
                        "module": "reference_only",
                        "model": "None",
                        "weight": 0.8,
                        "resize_mode": "Just Resize",
                        "control_mode": "Balanced",
                    },
                    {
                        "enabled": True,
                        "input_image": canny_control,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "weight": 1.2,
                        "resize_mode": "Just Resize",
                        "control_mode": "Balanced",
                        "lowvram": False,
                        "processor_res": 512,
                        "threshold_a": 50,
                        "threshold_b": 150,
                    }
                ]
            }
        },
        "script_name": None,
        "batch_size": 1,
        "mode": "inpaint",
        "override_settings": {
            "sd_model_checkpoint": "sd_xl_base_1.0"
        }
    }

    # 直接调用API生成
    response = requests.post(
        "http://127.0.0.1:7860/sdapi/v1/img2img",
        json=payload
    )

    try:
        result = response.json()
        if 'images' in result and len(result['images']) > 0:
            output_dir = "data/output/final"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{orig_stem}_final.png")
            img_bytes = base64.b64decode(result['images'][0])
            tmp_path = os.path.join(output_dir, f"{orig_stem}_tmp.png")
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            ai_img = Image.open(tmp_path).convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
            ai_img.save(out_path)
            os.remove(tmp_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI精炼成品图已保存到 {out_path}")
            return out_path
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 未检测到图片，返回内容：", result)
            return None
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API返回内容无法解析为JSON，原始内容：")
        print(response.text)
        return None

# 用法示例：
# refine_sdxl_pipeline('data/test_images/120745430.png') 

# ========== 多任务管理实用函数 ==========
def list_all_tasks():
    """列出所有任务（运行中、等待中、已完成）"""
    task_info = WebUIProgressMonitor.get_task_info()
    if not task_info:
        print("无法获取任务信息")
        return
    
    print("\n=== 任务列表 ===")
    
    # 运行中的任务
    running_tasks = task_info.get('queue_running', [])
    if running_tasks:
        print(f"\n🟢 运行中的任务 ({len(running_tasks)}个):")
        for i, task in enumerate(running_tasks, 1):
            task_id = task.get('id', 'unknown')
            prompt = task.get('prompt', '')[:50] + '...' if len(task.get('prompt', '')) > 50 else task.get('prompt', '')
            print(f"  {i}. ID: {task_id} | 提示词: {prompt}")
    else:
        print("\n🟢 运行中的任务: 无")
    
    # 等待中的任务
    pending_tasks = task_info.get('queue_pending', [])
    if pending_tasks:
        print(f"\n🟡 等待中的任务 ({len(pending_tasks)}个):")
        for i, task in enumerate(pending_tasks, 1):
            task_id = task.get('id', 'unknown')
            prompt = task.get('prompt', '')[:50] + '...' if len(task.get('prompt', '')) > 50 else task.get('prompt', '')
            print(f"  {i}. ID: {task_id} | 提示词: {prompt}")
    else:
        print("\n🟡 等待中的任务: 无")
    
    # 已完成的任务
    completed_tasks = task_info.get('queue_history', [])
    if completed_tasks:
        print(f"\n🔵 已完成的任务 ({len(completed_tasks)}个):")
        for i, task in enumerate(completed_tasks[-5:], 1):  # 只显示最近5个
            task_id = task.get('id', 'unknown')
            prompt = task.get('prompt', '')[:50] + '...' if len(task.get('prompt', '')) > 50 else task.get('prompt', '')
            print(f"  {i}. ID: {task_id} | 提示词: {prompt}")
    else:
        print("\n🔵 已完成的任务: 无")

def find_task_by_keywords(keywords):
    """根据关键词查找任务"""
    task_info = WebUIProgressMonitor.get_task_info()
    if not task_info:
        print("无法获取任务信息")
        return None
    
    result = WebUIProgressMonitor.find_task_by_prompt(task_info, target_prompt_keywords=keywords)
    if result:
        status = result['status']
        task = result['task']
        task_id = task.get('id', 'unknown')
        prompt = task.get('prompt', '')[:100] + '...' if len(task.get('prompt', '')) > 100 else task.get('prompt', '')
        
        print(f"\n找到任务:")
        print(f"  ID: {task_id}")
        print(f"  状态: {status}")
        print(f"  提示词: {prompt}")
        return result
    else:
        print(f"未找到包含关键词 {keywords} 的任务")
        return None

def monitor_task_by_id(task_id):
    """根据任务ID监控特定任务"""
    print(f"开始监控任务 {task_id}...")
    success = WebUIProgressMonitor.monitor_specific_task(task_id)
    if success:
        print(f"任务 {task_id} 监控完成")
    else:
        print(f"任务 {task_id} 监控失败或超时")
    return success

def monitor_task_by_keywords(keywords):
    """根据关键词监控特定任务"""
    print(f"开始监控包含关键词 {keywords} 的任务...")
    success = WebUIProgressMonitor.monitor_specific_task(keywords)
    if success:
        print(f"关键词任务监控完成")
    else:
        print(f"关键词任务监控失败或超时")
    return success

# ========== 进度监控使用说明 ==========
"""
WebUI 进度监控功能说明：

1. HTTP 方式监控（默认）：
   - 使用 /sdapi/v1/progress 接口
   - 每秒轮询一次进度
   - 自动显示当前步骤和百分比

2. WebSocket 方式监控（可选）：
   - 需要安装：pip install websocket-client
   - 实时推送进度更新
   - 更低的延迟和资源消耗

3. 多任务管理功能：
   - 列出所有任务（运行中、等待中、已完成）
   - 根据关键词查找特定任务
   - 根据任务ID监控特定任务
   - 支持多任务并发监控

4. 进度信息包含：
   - 当前步骤 (sampling_step)
   - 总步骤数 (sampling_steps) 
   - 完成百分比
   - 任务状态

5. 使用示例：
   # 简单监控
   WebUIProgressMonitor.monitor_progress_http()
   
   # 带回调的监控
   def my_callback(current, total, percent):
       print(f"处理进度: {percent:.1f}%")
   
   WebUIProgressMonitor.monitor_progress_http(my_callback)
   
   # WebSocket监控
   WebUIProgressMonitor.monitor_progress_websocket(my_callback)
   
   # 多任务管理
   list_all_tasks()  # 列出所有任务
   find_task_by_keywords(['nail art', 'stiletto'])  # 查找特定任务
   monitor_task_by_id('task_123')  # 监控特定任务ID
   monitor_task_by_keywords(['nail art'])  # 监控包含关键词的任务

6. 队列状态查询：
   queue_status = WebUIProgressMonitor.get_queue_status()
   print(f"队列状态: {queue_status}")
   
7. 在 refine_sdxl_pipeline 中使用：
   # 启用进度监控（默认）
   refine_sdxl_pipeline(img, orig_stem)
   
   # 禁用进度监控
   refine_sdxl_pipeline(img, orig_stem, enable_progress_monitor=False)
   
   # 监控特定任务（根据关键词）
   refine_sdxl_pipeline(img, orig_stem, task_keywords=['nail art', 'stiletto'])
""" 