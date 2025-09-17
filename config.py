"""
config.py

配置文件，包含所有配置项：
- API密钥
- 路径配置
- 模型参数
- 图像处理参数
"""

import os

# API配置
# 注意：请设置环境变量或在此处填入你的API密钥
# 如果遇到403错误，请访问 https://platform.openai.com/settings/organization/general 进行组织验证
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_API_URL = "https://api.openai.com/v1/images/edits"

# 路径配置
IMAGE_DIR = 'data/test_images'
OUTPUT_DIR = 'data/output'
MODEL_PATH = 'models/u2net_nail_best.pth'

# 模型参数
IMG_SIZE = (320, 320)
DEVICE = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') is not None else 'cpu'

# 图像处理参数
USE_GAUSSIAN_BLUR = True  # 是否使用高斯模糊羽化掩码
GAUSSIAN_KERNEL_SIZE = (5, 5)  # 高斯核大小，必须是奇数
GAUSSIAN_SIGMA = 2.0  # 高斯模糊的标准差

# 美甲效果参数
DEFAULT_PROMPT = "将指甲区域涂成粉红色指甲油，自然的美甲效果，边缘过渡自然，保持指甲形状完整，避免不自然、模糊、变形和颜色不均匀的效果"
NEGATIVE_PROMPT = "不自然, 模糊, 变形, 颜色不均匀"
BLEND_ALPHA = 0.7  # 融合强度

# 掩码处理参数
MASK_THRESHOLD = 128  # 掩码二值化阈值
MIN_REGION_SIZE = 100  # 最小区域大小（像素）
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11  # 自适应阈值邻域大小
ADAPTIVE_THRESHOLD_C = 2  # 自适应阈值常数差值 