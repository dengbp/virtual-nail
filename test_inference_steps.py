#!/usr/bin/env python3
"""
测试不同推理步数对美甲生成质量的影响
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from nail_sdxl_inpaint_opencv import NailSDXLInpaintOpenCV
import torch

class InferenceStepsTester:
    def __init__(self):
        self.nail_processor = NailSDXLInpaintOpenCV()
        
    def test_different_steps(self, image_path, mask_path, ref_path, test_steps=[15, 20, 25, 30, 35, 40]):
        """
        测试不同推理步数的效果
        """
        print("开始测试不同推理步数的效果...")
        
        # 读取测试图像
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path)
        
        if image is None or mask is None or ref_img is None:
            print("无法读取测试图像")
            return
            
        results = {}
        
        for steps in test_steps:
            print(f"\n测试 {steps} 步推理...")
            start_time = time.time()
            
            try:
                # 生成美甲效果
                result = self.nail_processor.sdxl_inpaint_controlnet_canny(
                    image=image,
                    mask=mask,
                    prompt="ultra realistic nail, natural nail shape, photorealistic, glossy, smooth, with highlights and reflections, 3D, natural texture, glossy, highlight, high quality, detailed",
                    negative_prompt="blur, smooth, repaint, remove glitter, remove texture, low detail, cartoon, painting, fake, plastic, color shift, color change, extra pattern, distortion, artifacts, watermark, text, logo",
                    control_strength=0.8,
                    num_inference_steps=steps,
                    guidance_scale=12.0 if steps < 30 else 9.0,  # 低步数时提高guidance
                    callback=None
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 保存结果
                output_path = f"test_results/steps_{steps}.png"
                os.makedirs("test_results", exist_ok=True)
                cv2.imwrite(output_path, result)
                
                # 计算质量指标
                quality_score = self.calculate_quality_score(result, image, mask)
                
                results[steps] = {
                    'time': generation_time,
                    'quality_score': quality_score,
                    'output_path': output_path
                }
                
                print(f"  - 生成时间: {generation_time:.1f}秒")
                print(f"  - 质量评分: {quality_score:.2f}")
                
            except Exception as e:
                print(f"  - 错误: {str(e)}")
                results[steps] = {'error': str(e)}
        
        return results
    
    def calculate_quality_score(self, result, original, mask):
        """
        计算生成结果的质量评分
        """
        # 简单的质量评估指标
        # 1. 边缘清晰度
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_result, cv2.CV_64F).var()
        
        # 2. 颜色一致性
        color_std = np.std(result, axis=(0, 1)).mean()
        
        # 3. 与原始图像的相似度（在非掩码区域）
        mask_bool = mask > 128
        non_mask_similarity = np.mean(np.abs(result[~mask_bool] - original[~mask_bool]))
        
        # 综合评分 (0-10分)
        edge_score = min(laplacian_var / 100, 3.0)  # 边缘清晰度
        color_score = max(0, 3.0 - color_std / 50)  # 颜色一致性
        similarity_score = max(0, 4.0 - non_mask_similarity / 50)  # 与原始图像相似度
        
        total_score = edge_score + color_score + similarity_score
        return min(10.0, total_score)
    
    def generate_comparison_report(self, results):
        """
        生成对比报告
        """
        print("\n" + "="*60)
        print("推理步数对比报告")
        print("="*60)
        
        print(f"{'步数':<6} {'时间(秒)':<10} {'质量评分':<10} {'效率比':<10}")
        print("-" * 40)
        
        best_efficiency = 0
        best_steps = 0
        
        for steps in sorted(results.keys()):
            if 'error' in results[steps]:
                print(f"{steps:<6} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
                continue
                
            time_taken = results[steps]['time']
            quality = results[steps]['quality_score']
            efficiency = quality / time_taken  # 质量/时间比
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_steps = steps
            
            print(f"{steps:<6} {time_taken:<10.1f} {quality:<10.2f} {efficiency:<10.3f}")
        
        print("-" * 40)
        print(f"推荐步数: {best_steps} (最佳效率比: {best_efficiency:.3f})")
        
        # 详细建议
        print("\n详细建议:")
        if best_steps <= 20:
            print("- 当前设置偏向速度，适合快速预览")
        elif best_steps <= 25:
            print("- 当前设置平衡质量和速度，适合一般使用")
        elif best_steps <= 30:
            print("- 当前设置偏向质量，适合高质量输出")
        else:
            print("- 当前设置追求最高质量，适合专业用途")
        
        return best_steps

def main():
    # 测试配置
    test_image = "data/test_images/test.jpg"
    test_mask = "data/test_masks/test_mask_input_mask.png"
    test_ref = "data/reference/test_reference.jpg"
    
    # 检查测试文件是否存在
    if not all(os.path.exists(f) for f in [test_image, test_mask, test_ref]):
        print("测试文件不存在，请先准备测试图像")
        return
    
    tester = InferenceStepsTester()
    
    # 测试不同步数
    results = tester.test_different_steps(
        test_image, 
        test_mask, 
        test_ref,
        test_steps=[15, 20, 25, 30, 35, 40]
    )
    
    # 生成报告
    recommended_steps = tester.generate_comparison_report(results)
    
    print(f"\n推荐配置:")
    print(f"- 推理步数: {recommended_steps}")
    print(f"- Guidance Scale: {12.0 if recommended_steps < 30 else 9.0}")
    print(f"- ControlNet权重: {[1.5, 0.8] if recommended_steps < 30 else [1.2, 0.7]}")

if __name__ == "__main__":
    main() 