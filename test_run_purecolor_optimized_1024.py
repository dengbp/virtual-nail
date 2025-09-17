import argparse
from pathlib import Path
from nail_sdxl_inpaint_purecolor_optimized_1024 import process_one_optimized_1024, preload_models_1024
import cv2
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
from nail_color_transfer import U2NetMasker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="调用 nail_sdxl_inpaint_purecolor_optimized_1024 处理纯色美甲渲染（1024分辨率版本）")
    parser.add_argument("--input_dir", type=str, default="data/test_images", help="手指原图目录，默认为 data/test_images")
    parser.add_argument("--output_mask_dir", type=str, default="data/output/masks", help="掩码输出目录，默认为 data/output/masks")
    parser.add_argument("--output_color_dir", type=str, default="data/output/color_transfer", help="颜色迁移输出目录，默认为 data/output/color_transfer")
    parser.add_argument("--output_final_dir", type=str, default="data/output/final", help="最终融合输出目录，默认为 data/output/final")
    parser.add_argument("--preload", action="store_true", help="是否预加载模型（推荐用于生产环境）")
    args = parser.parse_args()

    print("=" * 60)
    print("1024分辨率版本美甲渲染测试脚本")
    print("=" * 60)
    
    # 参考图路径写死为 'data/reference/reference.jpg'
    ref_path = Path("data/reference/reference.jpg")
    print(f"参考图路径: {ref_path}")
    
    # 检查参考图是否存在
    if not ref_path.exists():
        print(f"错误: 参考图不存在: {ref_path}")
        print("请确保 data/reference/reference.jpg 文件存在")
        exit(1)

    # 预加载模型（可选）
    if args.preload:
        print("开始预加载1024版本模型...")
        preload_models_1024()
        print("1024版本模型预加载完成")
    else:
        print("跳过模型预加载（首次处理时会有加载延迟）")

    # 初始化处理器（用于掩码生成）
    print("初始化掩码生成器...")
    masker = U2NetMasker()
    print("掩码生成器初始化完成")

    # 获取输入图像列表
    img_files = sorted(list(Path(args.input_dir).glob("*.*")))
    if not img_files:
        print(f"错误: 在目录 {args.input_dir} 中未找到图像文件")
        exit(1)
    
    print(f"找到 {len(img_files)} 个图像文件")
    
    # 处理每个图像
    for i, img_path in enumerate(img_files, 1):
        print(f"\n{'='*50}")
        print(f"处理第 {i}/{len(img_files)} 个图像: {img_path.name}")
        print(f"{'='*50}")
        
        stem = img_path.stem
        
        # 检查掩码是否存在
        mask_path = Path(f"data/test_masks/{stem}_mask_input_mask.png")
        if not mask_path.exists():
            print(f"未找到掩码: {mask_path}，使用 U²-Net 生成掩码")
            # 读取原图
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            # 使用 U²-Net 生成掩码
            mask = masker.get_mask(img, str(img_path), disable_cache=True)
            # 保存掩码
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
            print(f"掩码已生成并保存到: {mask_path}")
        else:
            print(f"使用现有掩码: {mask_path}")
        
        # 调用1024版本的处理函数
        try:
            print(f"开始处理图像: {img_path.name}")
            result = process_one_optimized_1024(
                img_path=str(img_path),
                ref_img_path=str(ref_path),
                output_mask_dir=args.output_mask_dir,
                output_color_dir=args.output_color_dir,
                output_final_dir=args.output_final_dir
            )
            if result and result.get('success'):
                print(f"✅ 处理完成: {result['final_path']}")
            else:
                print(f"❌ 处理失败: {img_path.name}")
        except Exception as e:
            print(f"❌ 处理异常: {img_path.name} - {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("所有图像处理完成！")
    print(f"最终结果保存在: {args.output_final_dir}")
    print(f"{'='*60}") 