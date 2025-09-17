import numpy as np
import cv2
import moderngl
from PIL import Image
import os
from pathlib import Path

# ========== HSV自适应检测 + 局部对比度 + 形态学优化 + 后处理 ==========

class AdaptiveHighlightDetector:
    """自适应高光检测器"""
    
    def __init__(self):
        self.debug_mode = True
        self.debug_dir = "data/debug/highlight"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def hsv_adaptive_detection(self, nail_image, nail_mask):
        """第一步：HSV自适应检测"""
        print("[高光检测] 步骤1: HSV自适应检测")
        
        # 转换到HSV空间
        hsv = cv2.cvtColor(nail_image, cv2.COLOR_BGR2HSV)
        
        # 提取指甲区域HSV值
        nail_pixels = hsv[nail_mask > 0]
        
        if len(nail_pixels) == 0:
            print("[警告] 没有检测到指甲区域")
            return np.zeros_like(nail_mask)
        
        # 自适应阈值计算
        saturation_threshold = np.percentile(nail_pixels[:, 1], 80)  # 饱和度阈值
        brightness_threshold = np.percentile(nail_pixels[:, 2], 90)  # 亮度阈值
        
        print(f"[高光检测] HSV阈值 - 饱和度: {saturation_threshold:.1f}, 亮度: {brightness_threshold:.1f}")
        
        # 检测高光
        highlight_mask = np.zeros_like(nail_mask)
        highlight_mask[(hsv[:, :, 1] < saturation_threshold) & 
                       (hsv[:, :, 2] > brightness_threshold) & 
                       (nail_mask > 0)] = 255
        
        if self.debug_mode:
            cv2.imwrite(f"{self.debug_dir}/step1_hsv_detection.png", highlight_mask)
        
        return highlight_mask
    
    def local_contrast_detection(self, nail_image, nail_mask):
        """第二步：局部对比度检测"""
        print("[高光检测] 步骤2: 局部对比度检测")
        
        # 转换为灰度图
        gray = cv2.cvtColor(nail_image, cv2.COLOR_BGR2GRAY)
        
        # 计算局部统计
        local_mean = cv2.blur(gray, (15, 15))
        local_var = cv2.blur(gray.astype(np.float32)**2, (15, 15)) - local_mean.astype(np.float32)**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # 计算对比度
        contrast = np.zeros_like(gray, dtype=np.float32)
        valid_mask = local_std > 1e-6
        contrast[valid_mask] = (gray[valid_mask] - local_mean[valid_mask]) / local_std[valid_mask]
        
        # 自适应阈值
        nail_contrast = contrast[nail_mask > 0]
        if len(nail_contrast) > 0:
            contrast_threshold = np.percentile(nail_contrast, 95)
        else:
            contrast_threshold = 2.0
        
        print(f"[高光检测] 对比度阈值: {contrast_threshold:.2f}")
        
        # 检测高对比度区域
        contrast_mask = np.zeros_like(nail_mask)
        contrast_mask[(contrast > contrast_threshold) & (nail_mask > 0)] = 255
        
        if self.debug_mode:
            cv2.imwrite(f"{self.debug_dir}/step2_contrast_detection.png", contrast_mask)
        
        return contrast_mask
    
    def morphological_optimization(self, combined_mask):
        """第三步：形态学优化"""
        print("[高光检测] 步骤3: 形态学优化")
        
        # 去除小噪点
        kernel_small = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 连接断裂区域
        kernel_medium = np.ones((5, 5), np.uint8)
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # 平滑边缘
        kernel_smooth = np.ones((3, 3), np.uint8)
        smoothed = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_smooth)
        
        if self.debug_mode:
            cv2.imwrite(f"{self.debug_dir}/step3_morphological.png", smoothed)
        
        return smoothed
    
    def post_processing(self, highlight_mask, nail_mask=None):
        """第四步：后处理 - 生成纯白无羽化高光，并严格限制在指甲掩码内"""
        print("[高光检测] 步骤4: 后处理")
        
        # 提取轮廓
        contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("[警告] 没有检测到高光轮廓")
            return np.zeros_like(highlight_mask)
        
        # 过滤小轮廓
        min_area = 20
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            print("[警告] 所有高光轮廓都太小")
            return np.zeros_like(highlight_mask)
        
        # 形状优化（不做过度多边形化，保留自然边缘）
        optimized_contours = []
        for contour in valid_contours:
            # 轻微简化轮廓，避免锐利多边形
            epsilon = 0.005 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            optimized_contours.append(simplified)
        
        # 生成纯白填充
        pure_white_mask = np.zeros_like(highlight_mask)
        cv2.fillPoly(pure_white_mask, optimized_contours, 255)
        
        # 锐化边缘（确保无羽化）
        sharp_mask = self.create_sharp_edges(pure_white_mask)
        
        # 只保留指甲区域的高光
        if nail_mask is not None:
            final_result = cv2.bitwise_and(sharp_mask, sharp_mask, mask=nail_mask)
        else:
            final_result = sharp_mask
        
        if self.debug_mode:
            cv2.imwrite(f"{self.debug_dir}/step4_pure_white.png", final_result)
        
        return final_result
    
    def create_sharp_edges(self, mask):
        """创建锐利边缘，确保无羽化"""
        # 使用形态学操作确保边缘锐利
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        return dilated
    
    def detect_highlight(self, nail_image, nail_mask):
        """完整的高光检测流程"""
        print("[高光检测] 开始完整检测流程")
        
        # 严格二值化掩码，确保高光检测严格受限于指甲区域
        if nail_mask.dtype != np.uint8:
            nail_mask = nail_mask.astype(np.uint8)
        _, nail_mask_bin = cv2.threshold(nail_mask, 128, 255, cv2.THRESH_BINARY)
        
        # 步骤1：HSV自适应检测
        hsv_result = self.hsv_adaptive_detection(nail_image, nail_mask_bin)
        
        # 步骤2：局部对比度检测
        contrast_result = self.local_contrast_detection(nail_image, nail_mask_bin)
        
        # 融合检测结果
        combined_result = cv2.bitwise_or(hsv_result, contrast_result)
        
        if self.debug_mode:
            cv2.imwrite(f"{self.debug_dir}/combined_detection.png", combined_result)
        
        # 步骤3：形态学优化
        morphological_result = self.morphological_optimization(combined_result)
        
        # 步骤4：后处理（传入严格二值化的nail_mask_bin限制高光区域）
        final_result = self.post_processing(morphological_result, nail_mask=nail_mask_bin)
        
        print(f"[高光检测] 检测完成 - 高光像素数: {np.sum(final_result > 0)}")
        
        return final_result

# ========== 高光融合函数 ==========

def add_nail_highlight_with_adaptive_detection(
    input_img: np.ndarray,
    output_path: str,
    nail_mask: np.ndarray = None,
    debug_mode: bool = True
):
    """
    使用自适应检测方法添加指甲高光
    
    Args:
        input_img: 输入图像
        output_path: 输出路径
        nail_mask: 指甲掩码（如果为None，将使用U2Net生成）
        debug_mode: 是否保存调试图像
    """
    print(f"[高光] 开始自适应检测渲染: 输入图像尺寸={input_img.shape}, 输出={output_path}")
    
    # 如果没有提供指甲掩码，使用U2Net生成
    if nail_mask is None:
        print("[高光] 使用U2Net生成指甲掩码")
        nail_mask = generate_nail_mask_with_u2net(input_img)
    
    # 确保掩码是uint8类型
    if nail_mask.dtype != np.uint8:
        nail_mask = nail_mask.astype(np.uint8)
    # 严格二值化掩码
    _, nail_mask_bin = cv2.threshold(nail_mask, 128, 255, cv2.THRESH_BINARY)
    
    # 创建高光检测器
    detector = AdaptiveHighlightDetector()
    detector.debug_mode = debug_mode
    
    # 检测高光
    highlight_mask = detector.detect_highlight(input_img, nail_mask_bin)
    
    # 生成黑色高光可视化图像
    result = input_img.copy()
    # 将高光区域设为黑色，其余保持原图
    highlight_area = highlight_mask > 0
    result[highlight_area] = [0, 0, 0]
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"[高光] 自适应检测渲染完成，已保存: {output_path}")
    
    return result

def generate_nail_mask_with_u2net(image):
    """使用U2Net生成指甲掩码"""
    try:
        from nail_color_transfer import U2NetMasker
        masker = U2NetMasker()
        mask = masker.generate_mask(image)
        
        # 后处理掩码
        mask = post_process_mask(mask)
        
        return mask
    except ImportError:
        print("[警告] U2NetMasker不可用，使用简单阈值方法")
        return generate_simple_mask(image)

def post_process_mask(mask):
    """后处理掩码"""
    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    return processed

def generate_simple_mask(image):
    """简单的掩码生成方法（备用）"""
    # 转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 基于肤色范围生成掩码
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def analyze_nail_properties(image, mask):
    """分析指甲属性"""
    # 提取指甲区域
    nail_region = image.copy()
    nail_region[mask == 0] = 0
    
    # 转换为灰度
    gray = cv2.cvtColor(nail_region, cv2.COLOR_BGR2GRAY)
    nail_pixels = gray[mask > 0]
    
    if len(nail_pixels) == 0:
        return {'type': 'unknown', 'contrast': 0.5}
    
    # 计算对比度
    contrast = np.std(nail_pixels) / (np.mean(nail_pixels) + 1e-6)
    
    # 判断指甲类型
    if contrast > 0.3:
        nail_type = 'glossy'
    elif contrast > 0.15:
        nail_type = 'semi_glossy'
    else:
        nail_type = 'matte'
    
    return {
        'type': nail_type,
        'contrast': contrast,
        'mean_brightness': np.mean(nail_pixels)
    }

def adjust_highlight_intensity(highlight_mask, nail_properties):
    """根据指甲属性调整高光强度"""
    # 基础强度
    base_intensity = 0.7
    
    # 根据指甲类型调整
    if nail_properties['type'] == 'glossy':
        intensity_factor = 1.0
    elif nail_properties['type'] == 'semi_glossy':
        intensity_factor = 0.8
    else:  # matte
        intensity_factor = 0.6
    
    # 根据对比度调整
    contrast_factor = min(nail_properties['contrast'] * 2, 1.0)
    
    # 计算最终强度
    final_intensity = base_intensity * intensity_factor * contrast_factor
    
    # 应用强度
    adjusted_mask = (highlight_mask.astype(np.float32) / 255.0) * final_intensity
    
    return adjusted_mask

# ========== 保留原有的GLSL着色器代码（兼容性） ==========

HIGHLIGHT_SHADER = """
#version 330
uniform sampler2D u_image;
uniform sampler2D u_alpha_mask;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec3 color = texture(u_image, v_texcoord).rgb;
    float alpha = texture(u_alpha_mask, v_texcoord).r;
    // 在碎片区域内叠加亮度渐变高光
    if (alpha > 0.1) {
        color = mix(color, vec3(1.0), alpha * 0.7); // 0.7为高光强度
    }
    fragColor = vec4(color, 1.0);
}
"""

def add_nail_highlight_with_shader(
    input_img: np.ndarray,
    output_path: str,
    alpha_mask: np.ndarray = None,
):
    """保留原有的GLSL着色器方法（兼容性）"""
    print(f"[高光] 开始GLSL渲染: 输入图像尺寸={input_img.shape}, 输出={output_path}")
    h, w = input_img.shape[:2]
    ctx = moderngl.create_standalone_context()
    prog = ctx.program(
        vertex_shader='''
            #version 330
            in vec2 in_vert;
            in vec2 in_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
        ''',
        fragment_shader=HIGHLIGHT_SHADER
    )
    # 全屏quad
    vertices = np.array([
        -1, -1, 0, 0,
        +1, -1, 1, 0,
        -1, +1, 0, 1,
        +1, +1, 1, 1,
    ], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_texcoord')
    # 纹理
    img_tex = ctx.texture((w, h), 3, input_img[..., ::-1].astype('u1').tobytes())
    img_tex.build_mipmaps()
    img_tex.use(0)
    prog['u_image'] = 0
    if alpha_mask is not None:
        alpha_tex = ctx.texture((w, h), 1, alpha_mask.astype('u1').tobytes())
        alpha_tex.build_mipmaps()
        alpha_tex.use(1)
        prog['u_alpha_mask'] = 1
    # 离屏渲染
    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()
    vao.render(moderngl.TRIANGLE_STRIP)
    data = fbo.read(components=3, alignment=1)
    img_out = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
    img_out = img_out[..., ::-1]  # 转回BGR
    cv2.imwrite(output_path, img_out)
    print(f"[高光] GLSL渲染完成，已保存: {output_path}")
    return img_out

if __name__ == "__main__":
    input_path = r"data/output/debug/your_sample_nail.png"  # 改成你的样板图路径
    output_path = r"data/output/debug/your_sample_nail_with_adaptive_highlight.png"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    # 使用新的自适应检测方法
    add_nail_highlight_with_adaptive_detection(img, output_path)
    print(f"自适应高光检测效果已保存: {output_path}") 