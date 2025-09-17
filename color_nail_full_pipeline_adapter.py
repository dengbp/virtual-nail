import sys
from pathlib import Path
from PIL import Image
import subprocess
import os
from color_transfer_pixel_level_transplant import process_one_pixel_transplant_auto, ensure_mask_exists
from color_transfer_pixel_level_refine_sdxl import refine_sdxl_pipeline
from color_nail_highlight_fill import add_highlight_to_image

def run_full_pipeline(img_path, ref_path, mask_path, task_id=None):
    """
    复用主流程：像素迁移+高光+AI精炼。
    img_path/ref_path/mask_path: 输入图片、参考色、掩码路径
    task_id: 任务ID，用于生成唯一的中间文件名
    返回: 精炼后图片路径
    """
    # 1. 像素迁移
    ensure_mask_exists(str(img_path))
    transplanted_img_path = process_one_pixel_transplant_auto(str(img_path), str(ref_path))
    if transplanted_img_path is None:
        raise Exception("像素迁移阶段失败")
    # 2. 高光叠加，强制指定输出文件名
    debug_dir = Path("data/output/debug")
    if task_id:
        highlight_out_path = debug_dir / f"{task_id}_with_antialiased_highlight.png"
    else:
        highlight_out_path = debug_dir / f"{Path(img_path).stem}_with_antialiased_highlight.png"
    
    # 使用新的模块化接口，避免重复加载
    # add_highlight_to_image 内部已经使用了缓存机制
    add_highlight_to_image(transplanted_img_path, str(highlight_out_path))
    
    # 3. AI精炼
    img = Image.open(highlight_out_path)
    orig_stem = Path(img_path).stem
    refined_img_path = refine_sdxl_pipeline(img, orig_stem)
    return refined_img_path