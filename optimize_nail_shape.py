import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def monochrome_transfer(seamless_img, color_transfer_img, mask):
    """
    用Lab色彩空间，将seamlessClone结果的L通道替换为color_transfer_img的L通道，只在掩码区域内替换。
    """
    lab_seamless = cv2.cvtColor(seamless_img, cv2.COLOR_BGR2LAB)
    lab_color = cv2.cvtColor(color_transfer_img, cv2.COLOR_BGR2LAB)
    # 替换L通道
    lab_seamless[..., 0] = lab_color[..., 0]
    result = cv2.cvtColor(lab_seamless, cv2.COLOR_LAB2BGR)
    # 只在掩码区域内替换
    mask3 = cv2.merge([mask, mask, mask])
    final = np.where(mask3 > 128, result, seamless_img)
    return final

def optimize_nail_shape(original_img, color_transfer_img, mask, 
                       kernel_size=5, feather_width=3, use_seamless=True):
    """
    优化指甲形状，集成seamlessClone+monochrome_transfer，保留真实感和色彩还原。
    """
    # Convert mask to binary if not already
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of nails
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new mask for optimized shape
    optimized_mask = np.zeros_like(binary_mask)
    
    # Process each nail contour
    for contour in contours:
        hull = cv2.convexHull(contour)
        nail_mask = np.zeros_like(binary_mask)
        cv2.drawContours(nail_mask, [hull], 0, 255, -1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel)
        nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel)
        optimized_mask = cv2.bitwise_or(optimized_mask, nail_mask)

    # feathered mask for smooth blending
    feathered_mask = cv2.GaussianBlur(optimized_mask, (feather_width*2+1, feather_width*2+1), 0)
    feathered_mask = feathered_mask.astype(float) / 255.0
    feathered_mask = np.stack([feathered_mask] * 3, axis=-1)

    # --- 新增：seamlessClone + monochrome_transfer ---
    if use_seamless:
        # seamlessClone 需要3通道掩码
        mask_for_clone = optimized_mask.copy()
        if mask_for_clone.max() == 1:
            mask_for_clone = (mask_for_clone * 255).astype(np.uint8)
        if len(mask_for_clone.shape) == 2:
            mask_for_clone = mask_for_clone
        center = (mask_for_clone.shape[1]//2, mask_for_clone.shape[0]//2)
        # seamlessClone
        seamless = cv2.seamlessClone(color_transfer_img, original_img, mask_for_clone, center, cv2.NORMAL_CLONE)
        # monochrome_transfer
        result = monochrome_transfer(seamless, color_transfer_img, optimized_mask)
    else:
        # 仅羽化融合
        result = original_img * (1 - feathered_mask) + color_transfer_img * feathered_mask
        result = result.astype(np.uint8)

    return result, optimized_mask

def process_directory(input_dir, mask_dir, output_dir, original_dir):
    """
    Process all images in the input directory.
    
    Args:
        input_dir: Directory containing color transfer results
        mask_dir: Directory containing nail masks
        output_dir: Directory to save optimized results
        original_dir: Directory containing original hand images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        color_transfer_path = os.path.join(input_dir, img_file)
        name, ext = os.path.splitext(img_file)
        mask_path = os.path.join(mask_dir, f"{name}_mask.png")
        # 自动适配原图后缀
        possible_exts = [ext, '.png', '.jpg', '.jpeg']
        original_path = None
        for e in possible_exts:
            candidate = os.path.join(original_dir, f"{name}{e}")
            if os.path.exists(candidate):
                original_path = candidate
                break
        print(f"\n[INFO] Processing: {img_file}")
        print(f"  color_transfer_path: {color_transfer_path} exists: {os.path.exists(color_transfer_path)}")
        print(f"  mask_path: {mask_path} exists: {os.path.exists(mask_path)}")
        print(f"  original_path: {original_path} exists: {original_path is not None and os.path.exists(original_path)}")

        missing = []
        if not os.path.exists(color_transfer_path):
            missing.append('color_transfer')
        if not os.path.exists(mask_path):
            missing.append('mask')
        if original_path is None:
            print(f"  [WARNING] No original image found for {img_file} with tried extensions: {possible_exts}")
            missing.append('original')
        if missing:
            print(f"  [WARNING] Skipping {img_file} - missing files: {', '.join(missing)}")
            continue

        # Read images
        color_transfer_img = cv2.imread(color_transfer_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(original_path)

        if color_transfer_img is None:
            print(f"  [ERROR] Failed to read color_transfer_img: {color_transfer_path}")
        if mask is None:
            print(f"  [ERROR] Failed to read mask: {mask_path}")
        if original_img is None:
            print(f"  [ERROR] Failed to read original_img: {original_path}")
        if any(img is None for img in [color_transfer_img, mask, original_img]):
            print(f"  [ERROR] Skipping {img_file} due to read error.")
            continue

        # Optimize nail shape
        optimized_img, optimized_mask = optimize_nail_shape(
            original_img, color_transfer_img, mask
        )
        
        # Save results
        output_path = os.path.join(output_dir, img_file)
        mask_output_path = os.path.join(output_dir, f"{name}_optimized_mask.png")
        
        cv2.imwrite(output_path, optimized_img)
        cv2.imwrite(mask_output_path, optimized_mask)
        
        print(f"  [SUCCESS] Processed {img_file}, saved to {output_path}")

if __name__ == "__main__":
    # Define directories
    input_dir = "data/output/color_transfer"
    mask_dir = "data/output/masks"
    output_dir = "data/output/color_transfer_shape"
    original_dir = "data/test_images"
    
    # Process all images
    process_directory(input_dir, mask_dir, output_dir, original_dir)
    print("Nail shape optimization completed!") 