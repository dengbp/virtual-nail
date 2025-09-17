# Nail Color Preview Application - Memory Bank

## 🎯 项目概述

这是一个基于AI的美甲颜色预览系统，集成了深度学习分割、图像处理、AI生成和Web服务等多个技术模块。系统能够自动识别指甲区域，应用不同颜色效果，并生成逼真的美甲预览图。

## 🏗️ 系统架构

### 核心功能模块

#### 1. **AI模型训练模块** (`train_u2net_pytorch.py`)
- **功能**: 训练U²-Net模型进行指甲分割
- **技术栈**: PyTorch, U²-Net, Albumentations
- **数据**: 1606张高质量训练图像
- **配置**: 
  - 分辨率: 1024px
  - 批次大小: 4
  - GPU: NVIDIA RTX A5000 (24GB)
  - 训练时间: 8-16小时

#### 2. **数据预处理模块** (`preprocess_training_data_precise.py`)
- **功能**: 精确预处理训练数据，确保与推理时完全一致
- **技术**: OpenCV, 精确缩放算法
- **特点**: 
  - 保持8的倍数对齐
  - 使用INTER_LANCZOS4高质量插值
  - 掩码使用INTER_NEAREST保持边缘清晰

#### 3. **掩码生成模块** (`generate_initial_masks.py`)
- **功能**: 使用U²-Net模型生成指甲分割掩码
- **模型**: U²-Net (U-shaped 2D Net)
- **架构**: 
  - 编码器-解码器结构
  - 多尺度特征融合
  - 7个输出层 (d0-d6)

#### 4. **颜色迁移模块** (`nail_color_transfer.py`)
- **功能**: 实现指甲颜色迁移和渲染
- **技术**: 
  - Phong光照模型
  - Lab颜色空间转换
  - 软融合算法
- **特点**: 支持多种混合模式

#### 5. **AI生成模块** (`nail_sdxl_inpaint_purecolor.py`)
- **功能**: 使用SDXL模型生成逼真指甲效果
- **技术**: 
  - Stable Diffusion XL
  - ControlNet
  - IP-Adapter
- **策略**: "空白画布"策略和"双重掩码"技术

#### 6. **高光渲染模块** (`color_nail_highlight_fill.py`)
- **功能**: 添加逼真的指甲高光效果
- **技术**: 
  - 物理光照系统
  - 抗锯齿高光渲染
  - 随机形状生成
- **特点**: 基于真实指甲形状的高光分布

#### 7. **Web服务模块** (`editor_image_server.py`)
- **功能**: 提供RESTful API服务
- **技术**: Flask, 异步处理
- **API**: 
  - POST /edit_nail - 美甲生成接口
  - 支持base64图像输入
  - 实时进度反馈

## 🤖 AI模型详解

### U²-Net模型架构
```python
# 核心组件
- REBNCONV: 残差块卷积
- RSU7/RSU6/RSU5/RSU4: 不同深度的U型结构
- 多尺度特征融合
- 7个输出层用于深度监督
```

### SDXL + ControlNet配置
```python
# 生成参数
- 分辨率: 1536px长边
- 推理步数: 40
- CFG比例: 5.0
- IP-Adapter强度: 0.75
```

## 📁 文件结构映射

### 训练相关
```
train_u2net_pytorch.py          # 主训练脚本
preprocess_training_data_precise.py  # 数据预处理
generate_initial_masks.py        # 掩码生成
训练环境安装指南.md              # 环境配置
```

### 推理相关
```
nail_color_transfer.py          # 颜色迁移核心
nail_sdxl_inpaint_purecolor.py # AI生成核心
color_nail_highlight_fill.py    # 高光渲染
editor_image_server.py          # Web服务
```

### 数据处理
```
data/
├── training_precise/           # 训练数据
│   ├── images/                # 1606张训练图像
│   └── masks/                 # 对应掩码
├── test_images/               # 测试图像
├── test_masks/               # 生成的掩码
├── reference/                # 参考色块
└── output/                   # 最终输出
```

## 🔧 依赖环境

### 训练环境
```bash
# Python包
torch==2.x.x+cu121          # PyTorch CUDA版本
torchvision, torchaudio      # PyTorch扩展
scikit-learn                # 机器学习
albumentations              # 数据增强
matplotlib, tqdm           # 可视化和进度
opencv-python              # 图像处理
kornia                     # 计算机视觉

# 硬件要求
GPU: NVIDIA RTX A5000 (24GB)
RAM: 32GB+
存储: 20GB可用空间
```

### 推理环境
```bash
# 基础依赖
torch, opencv-python, numpy
PIL, matplotlib, tqdm

# AI模型
diffusers                   # Stable Diffusion
transformers               # Hugging Face模型
controlnet-aux            # ControlNet扩展

# Web服务
flask                      # Web框架
```

## 🚀 核心算法

### 1. 指甲分割算法
```python
# U²-Net多输出损失函数
class StableU2NetLoss(nn.Module):
    def __init__(self, weights=[0.0, 0.0, 0.0, 1.0, 0.8, 0.6, 0.4]):
        # 跳过前3层有问题的输出
        # 主输出权重1.0，辅助输出递减权重
```

### 2. 颜色迁移算法
```python
# Lab空间颜色融合
def transfer_color_lab(image, mask, ref_color):
    # 转换到Lab空间
    # 保持AI生成的L通道
    # 使用参考色的a,b通道
    # 融合比例0.5
```

### 3. 高光渲染算法
```python
# 物理光照系统
def apply_phong_shading(image, mask, phong_params):
    # 环境光: 0.3
    # 漫反射: 0.7  
    # 镜面反射: 0.5
    # 光泽度: 32.0
```

## 📊 性能指标

### 训练性能
- **数据量**: 1606张高质量图像
- **训练时间**: 8-16小时
- **GPU利用率**: 90%
- **内存使用**: 24GB显存

### 推理性能
- **处理速度**: 30-60秒/张
- **分辨率支持**: 最高1536px
- **并发处理**: 支持多任务队列

## 🔄 处理流程

### 完整流水线
1. **图像输入** → 2. **U²-Net分割** → 3. **掩码增强** → 4. **颜色迁移** → 5. **AI生成** → 6. **高光渲染** → 7. **最终融合** → 8. **输出结果**

### 关键步骤详解
```python
# 1. 掩码生成
mask = U2NetMasker().get_mask(image, image_path)

# 2. 颜色迁移  
result = transfer_color_alpha_only(image, mask, ref_img)

# 3. AI生成
ai_result = nail.generate(image, mask, ref_image)

# 4. 高光添加
final = add_highlight_to_image(result, highlight_shapes)

# 5. 最终融合
output = blend_with_original(original, final, mask)
```

## 🛠️ 部署配置

### 开发环境
```bash
# 训练环境
python -m venv train_env
train_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 推理环境  
pip install flask diffusers transformers
```

### 生产部署
```bash
# Web服务启动
python editor_image_server.py
# 服务地址: http://0.0.0.0:80
# API端点: POST /edit_nail
```

## 📈 监控和日志

### 训练监控
- `train_u2net_detailed.log`: 详细训练日志
- `training_curves.png`: 训练曲线图
- TensorBoard: 实时训练监控

### 推理监控
- 实时进度反馈
- 错误日志记录
- 性能指标统计

## 🎨 技术特色

### 创新点
1. **双重掩码技术**: AI生成用扩张掩码，最终融合用精确掩码
2. **空白画布策略**: 预填充指甲区域，为AI提供干净画布
3. **物理光照系统**: 基于Phong模型的真实光照效果
4. **智能缓存机制**: 避免重复计算，提升性能

### 质量保证
- 8倍数尺寸对齐，优化AI处理
- 多尺度特征融合，提升分割精度
- 边缘羽化和平滑，确保自然过渡
- 自适应阈值调整，优化IoU指标

## Core Implementation Files

### nail_color_preview.py
- Main implementation file for the nail color preview application
- File size: 49KB
- Lines: 996

Let me examine the contents to provide more details about its functionality.

### Related Implementation Files
- `nail_colorizer.py` (13KB, 347 lines) - Handles color application logic
- `nail_detector.py` (27KB, 631 lines) - Core nail detection implementation
- `nail_detector_sam.py` (9.7KB, 264 lines) - SAM-based nail detection
- `nail_img2img.py` (9.7KB, 244 lines) - Image-to-image transformation for nails

## Model Training Files
- `train_nail_detection.py` (15KB, 413 lines) - Training script for nail detection
- `train_nail_segmentation.py` (3.9KB, 115 lines) - Training script for nail segmentation

## Utility Files
- `download_models.py` (6.7KB, 181 lines) - Script for downloading required models
- `system_info.py` (906B, 22 lines) - System information utilities
- `requirements.txt` (303B, 16 lines) - Project dependencies

# 项目结构和约定

## 目录结构
```
nail-color-preview/
├── images/              # 存放原始图片
├── masks/              # 存放生成的掩码图片
├── models/             # 存放模型文件
│   └── u2net.pth      # U2Net模型文件
├── output/            # 存放输出结果
└── venv/              # Python虚拟环境
```

## 核心文件说明
1. 模型相关：
   - `u2net_model.py`: U2Net模型定义
   - `models/u2net.pth`: 预训练模型文件

2. 掩码生成：
   - `generate_masks.py`: 生成图片掩码的主程序
   - `nail_detector.py`: 指甲检测器实现
   - `nail_detector_sam.py`: 基于SAM的指甲检测器

3. 颜色预览：
   - `nail_color_preview.py`: 指甲颜色预览主程序
   - `nail_colorizer.py`: 指甲颜色处理

4. 训练相关：
   - `train_nail_detection.py`: 指甲检测模型训练
   - `train_nail_segmentation.py`: 指甲分割模型训练
   - `train_u2net.py`: U2Net模型训练

5. 工具脚本：
   - `download_models.py`: 下载预训练模型
   - `download_sam_model.py`: 下载SAM模型
   - `convert_heic.py`: HEIC图片转换工具

## 命名约定
1. 图片文件：
   - 原始图片放在 `images/` 目录
   - 生成的掩码图片放在 `masks/` 目录
   - 掩码文件名格式：`{原文件名}_mask.png`

2. 模型文件：
   - 所有模型文件统一放在 `models/` 目录
   - 预训练模型保持原始文件名

3. 代码文件：
   - 使用小写字母和下划线
   - 文件名应清晰表达功能
   - 避免使用通用名称（如 `utils.py`）

## 功能模块
1. 图片处理：
   - 支持格式：jpg, jpeg, png, bmp
   - 图片预处理：调整大小、格式转换
   - 掩码生成：使用U2Net模型

2. 指甲检测：
   - 基于U2Net的语义分割
   - 支持批量处理
   - 输出二值化掩码

3. 颜色预览：
   - 支持多种颜色模式
   - 实时预览效果
   - 保存处理结果

## 开发规范
1. 代码风格：
   - 使用Python 3.8+
   - 遵循PEP 8规范
   - 添加适当的注释和文档字符串

2. 错误处理：
   - 使用try-except进行异常处理
   - 记录详细的错误日志
   - 提供清晰的错误信息

3. 日志记录：
   - 使用logging模块
   - 记录关键操作和错误
   - 保存日志到文件

## 依赖管理
1. 主要依赖：
   - PyTorch
   - OpenCV
   - Pillow
   - NumPy
   - Gradio (用于Web界面)

2. 版本控制：
   - 使用requirements.txt管理依赖
   - 指定具体的版本号
   - 定期更新依赖列表

## 注意事项
1. 目录结构：
   - 保持目录结构的一致性
   - 不要随意更改目录名称
   - 新增目录需要更新文档

2. 文件命名：
   - 保持命名规则的一致性
   - 避免使用特殊字符
   - 文件名应具有描述性

3. 代码维护：
   - 定期更新文档
   - 保持代码整洁
   - 及时修复问题

## Nail Color Preview Service API Documentation

### 接口基本信息
- **接口名称**: `/edit_nail`
- **请求方式**: POST
- **服务端口**: 80
- **服务地址**: `http://<server_ip>/edit_nail`

### 请求参数
请求参数以 form-data 形式提交：

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| img | string | 是 | 原图像的 base64 编码字符串 |
| ref_img | string | 是 | 参考色图像的 base64 编码字符串 |

### 响应参数
响应为 JSON 格式：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| status | int | 处理状态：0 表示成功，-1 表示失败 |
| msg | string | 处理结果说明，成功时为"成功"，失败时为具体错误原因 |
| result | string | 处理成功时返回最终效果图的 base64 编码字符串，失败时为空字符串 |

### 文件存储说明
服务会自动将处理过程中的文件保存到以下目录：
- 原图像：`data/test_images/<timestamp>.jpg`
- 掩码图：`data/test_masks/<timestamp>_mask.jpg`
- 参考图像：`data/reference/<timestamp>_reference.jpg`
- 最终效果图：`data/output/final/<timestamp>_final.png`

其中 `<timestamp>` 格式为：时分秒毫秒（例如：`143022123`）

### 调用示例
```python
import requests
import base64

# 读取图像文件并转为 base64
with open("original.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
with open("reference.jpg", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode("utf-8")

# 发送请求
response = requests.post(
    "http://<server_ip>/edit_nail",
    data={
        "img": img_b64,
        "ref_img": ref_b64
    }
)

# 处理响应
result = response.json()
if result["status"] == 0:
    # 将返回的 base64 字符串解码为图像
    final_img_data = base64.b64decode(result["result"])
    with open("final_result.png", "wb") as f:
        f.write(final_img_data)
else:
    print(f"处理失败：{result['msg']}")
```

### 注意事项
1. 确保服务器有足够的存储空间，因为会保存处理过程中的所有图像文件
2. 图像文件会按时间戳命名，便于追踪处理历史
3. 服务会自动创建必要的目录结构
4. 建议在生产环境中添加适当的错误处理和日志记录机制
5. 可以根据需要取消注释代码中的文件清理部分，以自动删除处理后的临时文件 