import sys
from pathlib import Path
import time
import os
import cv2
from PIL import Image

# 确保项目根目录在sys.path中，以便导入模块
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from color_transfer_pixel_level_transplant import process_one_pixel_transplant_auto, ensure_mask_exists
except ImportError as e:
    print(f"错误: 无法导入 color_transfer_pixel_level_transplant。\n详细错误: {e}")
    sys.exit(1)

try:
    from color_transfer_pixel_level_refine_sdxl import refine_sdxl_pipeline
except ImportError as e:
    print(f"错误: 无法导入 color_transfer_pixel_level_refine_sdxl。\n详细错误: {e}")
    sys.exit(1)

try:
    from color_nail_highlight_fill import add_highlight_to_image
except ImportError as e:
    print(f"错误: 无法导入 color_nail_highlight_fill。\n详细错误: {e}")
    sys.exit(1)

def run_pipeline(img_path: Path, ref_path: Path):
    print(f"--- [PIPELINE START] --- Processing: {img_path.name} ---")
    pipeline_start_time = time.time()

    # --- STAGE 1: Pixel Transplant ---
    print("\n--- [PIPELINE STAGE 1/2] Running Pixel Transplant ---")
    mask_path = ensure_mask_exists(str(img_path))
    transplanted_img_path = process_one_pixel_transplant_auto(str(img_path), str(ref_path))
    if transplanted_img_path is None:
        print("错误: 像素迁移阶段失败，流水线终止。")
        return
    print(f"--- [PIPELINE STAGE 1/2] DONE. Transplanted image at: {transplanted_img_path}")
    print(f"--- Authoritative mask saved at: {mask_path}")

    # --- STAGE 1.5: Antialiased Highlight Overlay ---
    print("\n--- [PIPELINE STAGE 1.5] Adding Antialiased Highlight ---")
    # 使用新的模块化接口，避免重复加载
    add_highlight_to_image(transplanted_img_path)
    print("--- [PIPELINE STAGE 1.5] DONE. Antialiased highlight overlay complete.")

    # --- STAGE 2: AI Refinement ---
    print("\n--- [PIPELINE STAGE 2/2] Running AI Refinement ---")
    debug_dir = Path("data/output/debug")
    out_path_aa = debug_dir / f"{Path(transplanted_img_path).stem}_with_antialiased_highlight.png"
    img = Image.open(out_path_aa)
    orig_stem = Path(img_path).stem
    refined_img = refine_sdxl_pipeline(img, orig_stem)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = debug_dir / f"{Path(transplanted_img_path).stem}_refined.png"
    if isinstance(refined_img, str):
        print(f"[AI精炼] 精炼后图片已保存: {refined_img}")
    elif hasattr(refined_img, 'save'):
        refined_img.save(str(out_img_path))
    else:
        import cv2
        import numpy as np
        if isinstance(refined_img, np.ndarray):
            cv2.imwrite(str(out_img_path), refined_img)
        else:
            raise TypeError("refine_sdxl_pipeline返回类型无法保存，请检查其输出类型")

    pipeline_end_time = time.time()
    print(f"\n--- [PIPELINE COMPLETE] ---")
    print(f"完整流水线总耗时: {pipeline_end_time - pipeline_start_time:.2f}秒")

if __name__ == '__main__':
    input_dir = Path("data/test_images")
    ref_dir = Path("data/reference")
    Path("data/masks").mkdir(exist_ok=True)
    Path("data/output/final").mkdir(exist_ok=True)
    Path("data/output/final_refined").mkdir(exist_ok=True)

    if not input_dir.exists():
        sys.exit(f"错误: 输入目录 '{input_dir}' 不存在。")

    ref_paths = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.jpeg"))
    if not ref_paths:
        sys.exit(f"错误: 参考图目录 '{ref_dir}' 为空。")
    ref_path = ref_paths[0]

    img_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not img_files:
        sys.exit(f"错误: 在 '{input_dir}' 目录下没有找到任何图片文件。")

    print(f"找到 {len(img_files)} 张待处理图片。")
    for img_path in img_files:
        run_pipeline(img_path, ref_path)
        print("\n" + "="*80 + "\n") 