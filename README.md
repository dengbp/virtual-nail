# virtual-nail
System基于AI的美甲虚拟试戴系统 - 集成U²-Net分割、像素级颜色迁移、物理光照渲染、SDXL AI增强的完整美甲效果预览解决方案
# 🎨 AI美甲虚拟试戴系统 (Virtual-Nail System)

## 🌟 效果展示

### 💅 产品常规效果图

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="data/产品常规效果图/2623972fdc1c1567e6cdf25adf7d4fa2.JPG" width="180" alt="美甲效果图1" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/3b98f9a6735ee1723574f37be555b2ea.JPG" width="180" alt="美甲效果图2" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/473e5b33eadd99338de65d83d59d419f.JPG" width="180" alt="美甲效果图3" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/685c7a11bb16e20c3a68f762fab8db9f.JPG" width="180" alt="美甲效果图4" />
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="data/产品常规效果图/8ecf1799dd43a49a21296b7b3f435488.JPG" width="180" alt="美甲效果图5" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/a566b30aaa08d360f9ff6518682b4d72.JPG" width="180" alt="美甲效果图6" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/b07f0a85dda7ac1548ace782d5ea2cbf.JPG" width="180" alt="美甲效果图7" />
      </td>
      <td align="center">
        <img src="data/产品常规效果图/d302eaf0e5d6202d608eb7e9573e094a.JPG" width="180" alt="美甲效果图8" />
      </td>
    </tr>
    <tr>
      <td align="center" colspan="2">
        <img src="data/产品常规效果图/e9b1ba1895154a1a9033cd2a4dbe7662.JPG" width="180" alt="美甲效果图9" />
      </td>
      <td align="center" colspan="2">
        <img src="data/产品常规效果图/fc12e78b4e95c1807d585b05e470764b.JPG" width="180" alt="美甲效果图10" />
      </td>
    </tr>
  </table>
  <p><em>展示多种美甲颜色和风格的AI生成效果，色彩饱满，质感逼真</em></p>
</div>

### 🎨 广告宣传效果图

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="data/广告宣传效果图/1e0b3f5a4d76e2a656df918b561e545a.JPG" width="300" alt="广告宣传图1" />
        <br><em>专业级美甲展示效果</em>
      </td>
      <td align="center">
        <img src="data/广告宣传效果图/IMG_1525.jpg" width="300" alt="广告宣传图2" />
        <br><em>创意美甲设计展示</em>
      </td>
    </tr>
  </table>
  <p><em>高品质的宣传展示图，突出美甲艺术的精致与美感</em></p>
</div>

### ✨ 技术特色

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <strong>🎯 精准分割</strong>
        <br>U²-Net深度学习<br>IoU > 0.92
      </td>
      <td align="center" width="25%">
        <strong>🎨 颜色迁移</strong>
        <br>TPS变形算法<br>像素级精确
      </td>
      <td align="center" width="25%">
        <strong>💡 物理光照</strong>
        <br>Phong模型<br>逼真高光渲染
      </td>
      <td align="center" width="25%">
        <strong>🤖 AI增强</strong>
        <br>SDXL深度优化<br>专业级质感
      </td>
    </tr>
  </table>
</div>

## 🧠 核心模型介绍

### 🎯 U²-Net指甲分割模型 (`models/u2net_nail_best.pth`)

我们基于经典的 **U-2-Net** 模型架构，使用 `u2net.pth` 预训练权重进行了第一轮微调训练。目前模型在指甲位置识别方面**已达到80%的场景适应性**，能够准确处理大部分常见场景，但在复杂光照、极端角度等挑战性场景下**仍有20%的提升空间**。

#### 📊 当前训练状态
- ✅ **第一轮微调**: 已完成，基于U-2-Net预训练模型
- 🔄 **第二轮训练**: 正在进行数据标注和模型优化  
- 🎯 **训练目标**: 实现指甲位置的鲁棒性识别，适应各种光照、角度、手势场景

#### 🖼️ 训练数据展示

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>原始手部图像</strong></td>
      <td align="center"><strong>精确标注掩码</strong></td>
    </tr>
    <tr>
      <td align="center">
        <img src="data/images/微信图片_20250917174659_1_2.png" width="300" alt="手部原图" />
        <br><em>真实手部图像 - 训练样本</em>
      </td>
      <td align="center">
        <img src="data/masks/微信图片_20250917174752_2_2.png" width="300" alt="指甲分割掩码" />
        <br><em>精确标注的指甲分割掩码</em>
      </td>
    </tr>
  </table>
  <p><em>展示第二轮训练数据的标注质量：像素级精确的指甲区域分割</em></p>
</div>

#### 🔬 技术细节
- **模型架构**: U²-Net (U-square Network)
- **输入尺寸**: 长边1024像素，短边等比缩放；长边小于1024的按原图输入
- **输出**: 指甲区域分割掩码
- **训练策略**: 迁移学习 + 领域自适应微调
- **数据增强**: 旋转、缩放、光照变化、色彩扰动

#### 🚀 未来规划
我们正在扩展训练数据集，目标是在第二轮训练后实现：
- 📈 **IoU > 0.95**: 分割精度显著提升
- 🌟 **鲁棒性增强**: 适应各种复杂场景
- ⚡ **实时推理**: 优化模型推理速度
- 🎨 **多样性支持**: 支持不同肤色、指甲形状

## 📖 项目概述

这是一个基于人工智能的美甲虚拟试戴系统，通过深度学习技术实现从简单颜色替换到专业级美甲效果渲染的技术突破。系统采用**三阶段渐进式处理架构**，将传统图像处理与前沿AI技术完美融合，为美甲行业提供了革命性的数字化解决方案。

### 🎯 核心价值

- **🏪 商业应用**: 美甲店客户可实时预览美甲效果，提升服务体验
- **🛒 电商增强**: 美甲产品在线试戴，大幅提升转化率
- **📱 移动集成**: 可集成到美妆APP，提供个人美甲预览功能
- **🎨 创意设计**: 美甲师可快速验证设计效果，优化创作流程

## ✨ 核心特性

### 🧠 智能技术栈
- **深度学习分割**: 基于U²-Net的高精度指甲区域识别（IoU > 0.92）
- **像素级颜色迁移**: TPS变形算法实现精确颜色移植
- **物理光照渲染**: Phong模型生成逼真高光和阴影效果
- **AI质感增强**: Stable Diffusion XL深度优化细节和质感
- **智能边缘融合**: Active Contour算法优化分割边缘

### ⚡ 工程化特性
- **生产级API**: Flask RESTful接口，支持高并发部署
- **实时处理**: 30-60秒完成完整AI处理流程
- **智能缓存**: 避免重复计算，大幅提升性能
- **多级降级**: 确保在任何环境下都能稳定运行
- **完整监控**: 详细日志和进度回调机制

### 🔧 开发友好
- **完整工具链**: 从数据标注到模型训练的全流程工具
- **详细文档**: 包含API、训练、部署的完整技术文档
- **模块化设计**: 高内聚低耦合，便于二次开发

## 🏗️ 系统架构

### 三阶段处理流水线
<img width="600" alt="美甲系统架构流程图" src="https://github.com/user-attachments/assets/9f9060b0-f2ad-47a2-bdd4-9818954c5c37" />

### 技术创新点

1. **双重掩码策略**: AI生成使用扩张掩码提供足够空间，最终融合使用精确掩码确保边缘自然
2. **空白画布技术**: 预填充指甲区域为AI提供干净生成画布，避免原色干扰
3. **渐进式质量提升**: 每阶段专注特定效果，最终融合达到专业级质量
4. **智能降级机制**: Active Contour → 形状优化 → 基础增强 → 原始掩码的四级保障

## 📁 项目结构

```
nail-color-preview/
├── 🎯 核心处理模块
│   ├── editor_image_server.py              # 主服务器 - Flask API接口
│   ├── color_nail_full_pipeline_adapter.py # 完整流水线适配器
│   ├── color_transfer_pixel_level_transplant.py  # 第一阶段：像素迁移
│   ├── color_nail_highlight_fill.py        # 第二阶段：高光渲染
│   ├── color_transfer_pixel_level_refine_sdxl.py # 第三阶段：AI精炼
│   └── nail_color_transfer.py              # 颜色迁移核心算法
│
├── 🧠 AI模型相关
│   ├── u2net_model.py                      # U²-Net模型定义
│   ├── train_u2net_pytorch.py              # U²-Net训练脚本（主力）
│   ├── train_u2net_stable.py               # 稳定版训练脚本
│   ├── train_u2net_memory_optimized.py     # 内存优化训练脚本
│   ├── nail_sdxl_inpaint_opencv.py         # SDXL增强处理类
│   ├── nail_sdxl_inpaint_purecolor.py      # 纯色SDXL处理
│   └── nail_active_contour_enhancer.py     # Active Contour增强
│
├── 📊 数据处理工具
│   ├── convert_labelme_to_masks.py         # LabelMe标注转换器
│   ├── preprocess_training_data_precise.py # 精确数据预处理
│   ├── validate_labelme_data.py            # 标注数据验证
│   ├── quick_start_labelme.py              # 快速标注工具
│   └── generate_initial_masks.py           # 初始掩码生成
│
├── 🎨 高级渲染模块
│   ├── physical_lighting_system.py         # 物理光照系统
│   ├── color_nail_highlight_shader.py      # 高光着色器
│   ├── color_antialiased_highlight_visualizer.py # 抗锯齿可视化
│   └── nail_template_system.py             # 指甲模板系统
│
├── 🔧 工具脚本
│   ├── download_models.py                  # 模型下载工具
│   ├── system_info.py                      # 系统信息检查
│   ├── fix_encoding_issues.py              # 编码问题修复
│   └── security_middleware.py              # 安全中间件
│
├── 📚 文档
│   ├── API_FLOW_DOCUMENTATION.md           # API流程文档
│   ├── TESTING_GUIDE.md                    # 测试指南
│   ├── 训练环境安装指南.md                   # 训练环境指南
│   ├── MID_TERM_UPGRADE.md                 # 中期升级说明
│   └── memory_bank.md                      # 技术知识库
│
└── 📦 配置文件
    ├── requirements.txt                     # Python依赖
    ├── config.py                           # 系统配置
    └── logging_config.py                   # 日志配置
```

## 🚀 快速开始

### 环境要求

#### 硬件要求
- **GPU**: NVIDIA RTX 2080+ (8GB+ VRAM) 推荐RTX A5000 (24+GB)
- **内存**: 16GB+ RAM，训练时建议32GB+
- **存储**: 50GB+ 可用空间（包含模型文件）
- **CPU**: Intel i7+ 或 AMD Ryzen 7+

#### 软件环境
- **操作系统**: Ubuntu 18.04+ / macOS 10.15+ / Windows 10+
- **Python**: 3.8+ (推荐3.9)
- **CUDA**: 11.7+ (如使用GPU)
- **Git**: 2.20+

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/dengbp/virtual-nai.git
cd virtual-nai
```

#### 2. 创建虚拟环境
```bash
# 使用conda（推荐）
conda create -n nail-nai python=3.9
conda activate nail-nai

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
```

#### 3. 安装依赖
```bash
# 安装PyTorch（CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

#### 4. 下载模型文件
```bash
# 下载预训练模型
python download_models.py

# 验证模型完整性
python system_info.py
```

#### 5. 创建数据目录
```bash
mkdir -p data/{test_images,test_masks,reference,output/final,output/debug}
```

### 启动服务

#### 启动API服务
```bash
# 启动美甲生成服务器
python editor_image_server.py

# 服务器将在 http://0.0.0.0:80 启动
```

## 🎯 核心功能详解

### 1. 智能指甲分割

#### U²-Net深度学习模型
- **模型架构**: U-shaped 2D Network with 7-layer deep supervision
- **训练数据**: 1606张高质量手部图像 + 精确标注
- **分割精度**: IoU > 0.92, Dice > 0.95
- **边缘优化**: Active Contour算法进一步精化边缘

#### 关键训练文件
```python
# 模型定义
u2net_model.py              # U²-Net网络架构

# 训练脚本
train_u2net_pytorch.py      # 主训练脚本（超高鲁棒性）
train_u2net_stable.py       # 稳定版（适合调试）
train_u2net_memory_optimized.py  # 内存优化版

# 数据预处理
preprocess_training_data_precise.py  # 精确预处理（1024长边）
convert_labelme_to_masks.py          # LabelMe标注转换
validate_labelme_data.py             # 标注数据验证
```

### 2. 像素级颜色迁移

#### TPS变形算法
- **技术原理**: Thin Plate Spline非刚性变形
- **关键点检测**: 基于掩码轮廓的智能关键点提取
- **无缝融合**: Alpha混合 + 边缘羽化

#### 核心模块文件
```python
# 主处理模块
color_transfer_pixel_level_transplant.py

# 颜色迁移核心
nail_color_transfer.py

# 完整流水线适配器
color_nail_full_pipeline_adapter.py
```

### 3. 物理光照渲染

#### Phong光照模型
- **环境光**: 0.3强度基础照明
- **漫反射**: 0.7强度表面散射
- **镜面反射**: 可调强度高光效果
- **抗锯齿**: 多倍采样消除边缘锯齿

#### 高光渲染模块文件
```python
# 物理光照系统
physical_lighting_system.py

# 高光渲染
color_nail_highlight_fill.py

# 高光着色器
color_nail_highlight_shader.py

# 抗锯齿可视化
color_antialiased_highlight_visualizer.py
```

### 4. SDXL AI增强

#### Stable Diffusion XL集成
- **模型**: stabilityai/stable-diffusion-xl-base-1.0
- **ControlNet**: 结构控制和边缘保持
- **IP-Adapter**: 风格参考和质感迁移
- **优化策略**: 混合精度 + 梯度检查点

#### 核心模块文件
```python
# SDXL处理类
nail_sdxl_inpaint_opencv.py

# AI精炼管道
color_transfer_pixel_level_refine_sdxl.py

# 纯色处理版本
nail_sdxl_inpaint_purecolor.py
```


## 📊 训练数据流向图

<img width="500" height="600" alt="美甲训练数据流向图" src="https://github.com/user-attachments/assets/c9579c02-2274-492c-8396-c6f58184c261" />



## 🎯 核心处理数据流向图

<img width="500" height="600" alt="美甲核心处理数据流向图" src="https://github.com/user-attachments/assets/985f68dc-676a-42a2-b3d9-1e3fe8f620b7" />


## 🔧 技术栈详解

### 深度学习框架
- **torch>=2.0.0+cu121** - PyTorch CUDA版本
- **torchvision>=0.15.0** - 计算机视觉库
- **diffusers>=0.21.0** - Stable Diffusion管道
- **transformers>=4.25.0** - Transformer模型
- **controlnet-aux>=0.0.6** - ControlNet辅助工具

### 图像处理库
- **opencv-python>=4.5.0** - 计算机视觉处理
- **Pillow>=9.0.0** - 图像I/O和基础处理
- **albumentations>=1.3.0** - 数据增强
- **scikit-image>=0.19.0** - 科学图像处理
- **matplotlib>=3.5.0** - 可视化和绘图

### Web服务框架
- **Flask>=2.0.0** - Web服务框架
- **Flask-CORS>=3.0.0** - 跨域资源共享
- **requests>=2.28.0** - HTTP客户端
- **gunicorn>=20.1.0** - WSGI服务器

### 数据科学工具
- **numpy>=1.21.0** - 数值计算
- **scipy>=1.7.0** - 科学计算
- **pandas>=1.3.0** - 数据处理
- **scikit-learn>=1.0.0** - 机器学习
- **tqdm>=4.62.0** - 进度条

## 📊 API接口文档

### 主要接口

#### POST /edit_nail - 美甲生成接口

**请求格式**:
```http
POST /edit_nail HTTP/1.1
Content-Type: application/x-www-form-urlencoded

img=<base64_encoded_image>&ref_img=<base64_encoded_reference>
```

**请求参数**:
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| img | string | 是 | 原始手部图片的base64编码（不含前缀） |
| ref_img | string | 是 | 参考颜色图片的base64编码（不含前缀） |

**响应格式**:
```json
{
    "statusCode": 200,
    "message": "生成完成",
    "task_id": "143022123",
    "data": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**使用示例**:
```python
import requests
import base64

# 编码图像
with open("hand.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("color_ref.jpg", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode("utf-8")

# 发送请求
response = requests.post("http://localhost/edit_nail", data={
    "img": img_b64,
    "ref_img": ref_b64
})

result = response.json()
if result["statusCode"] == 200:
    # 解码并保存结果
    import re
    image_data = re.sub(r'^data:image/\w+;base64,', '', result["data"])
    with open("result.png", "wb") as f:
        f.write(base64.b64decode(image_data))
```

### 性能指标

- **响应时间**: 30-60秒（完整AI处理）
- **并发支持**: 支持多任务队列
- **文件大小**: 支持最大100MB图像
- **分辨率**: 最高支持1536px长边


## 🎓 训练指南

### 数据准备

#### 1. 图像采集
```bash
# 推荐图像规格
分辨率: 1024px-4096px
格式: JPG/PNG
质量: 无压缩或轻度压缩
光照: 均匀光照，避免强阴影
角度: 多角度手部姿态
```

#### 2. 数据标注
```bash
# 使用LabelMe进行标注
pip install labelme

# 启动标注工具
labelme

# 标注完成后转换为训练格式
python convert_labelme_to_masks.py
```

#### 3. 数据预处理
```bash
# 精确预处理（推荐）
python preprocess_training_data_precise.py

# 验证数据质量
python validate_labelme_data.py
```

### 模型训练

#### U²-Net分割模型训练
```bash
# 1. 准备训练数据
# 确保 data/training_precise/ 目录包含：
# - images/: 1606张预处理图像
# - masks/: 对应的掩码文件

# 2. 开始训练（主力脚本）
python train_u2net_pytorch.py

# 训练配置
EPOCHS = 120
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMAGE_SIZE = 1024  # 长边尺寸
```

#### 训练参数调优
```python
# train_u2net_pytorch.py 关键参数
class UltraRobustNailSegmentationDataset:
    def __init__(self, max_size=1024, is_train=True):
        self.max_size = max_size  # 1024长边
        
        # 数据增强策略
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=1024),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.7),
            # ... 更多增强
        ])
```


### 模型评估
```bash
# 生成验证掩码
python generate_initial_masks.py

# 质量验证
python verify_mask_quality.py

# 性能测试
python test_model_load.py
```

## 🚀 部署指南

### 本地部署

```bash
# 使用Gunicorn部署
pip install gunicorn

# 启动服务器
gunicorn -w 4 -b 0.0.0.0:80 --timeout 300 editor_image_server:app

# 配置文件
config.py:
    DEBUG = False
    PORT = 80
    WORKERS = 4
```

### 云平台部署

#### AWS部署
```bash
# EC2实例要求
实例类型: g4dn.xlarge (4 vCPU, 16GB RAM, 1x NVIDIA T4)
存储: 100GB EBS gp3
网络: 公网IP + 安全组配置

# ECS部署
aws ecs create-cluster --cluster-name nail-preview-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

#### 阿里云部署
```bash
# ECS GPU实例
实例规格: ecs.gn6i-c4g1.xlarge
GPU: NVIDIA T4 (16GB)
镜像: Ubuntu 20.04 + CUDA 11.7
```

### 性能优化

#### GPU优化
```python
# 启用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

#### 内存优化
```python
# 模型检查点
torch.utils.checkpoint.checkpoint(function, *args)

# 内存清理
import gc
import torch

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
```


## 🤝 贡献指南


### 贡献流程

1. **Fork项目** → 2. **创建特性分支** → 3. **开发功能** → 4. **提交PR**

```bash
# 1. Fork后克隆
git clone https://github.com/your-username/nail-color-preview.git

# 2. 创建分支
git checkout -b feature/new-highlight-algorithm

# 3. 开发功能
# ... 开发代码 ...

# 4. 提交更改
git add .
git commit -m "feat(highlight): 实现新的高光算法"

# 5. 推送并创建PR
git push origin feature/new-highlight-algorithm
```

### 开发环境设置

#### VSCode配置
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true
}
```

#### 预提交钩子
```bash
# 安装pre-commit
pip install pre-commit

# 配置钩子
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
EOF

# 安装钩子
pre-commit install
```

## 📊 性能基准

### 硬件配置对比

| 配置 | GPU | 处理时间 | 内存使用 | 推荐场景 |
|------|-----|---------|---------|----------|
| **入门级** | GTX 1660 (6GB) | 90-120秒 | 12GB RAM | 开发使用 |
| **标准级** | RTX 3080 (10GB) | 45-60秒 | 16GB RAM | 小规模生产 |
| **专业级** | RTX A5000 (24GB) | 30-45秒 | 32GB RAM | 大规模生产 |
| **企业级** | RTX A6000 (48GB) | 20-30秒 | 64GB RAM | 高并发服务 |

### 处理能力对比

| 并发数 | 硬件配置 | 平均响应时间 | 成功率 | 资源使用率 |
|--------|---------|-------------|--------|-----------|
| 1 | RTX 3080 | 45秒 | 99.5% | GPU 85% |
| 2 | RTX 3080 | 90秒 | 98.0% | GPU 95% |
| 4 | RTX A5000 | 60秒 | 99.0% | GPU 90% |
| 8 | RTX A6000 | 45秒 | 98.5% | GPU 95% |

## 📈 路线图

### 已完成功能 ✅
- [x] U²-Net指甲分割模型
- [x] 三阶段处理流水线
- [x] API服务接口
- [x] 完整技术文档
- [x] Active Contour边缘优化
- [x] SDXL AI增强集成

### 正在开发 🔄
- [ ] 实时进度推送（WebSocket）
- [ ] 批量处理接口
- [ ] 移动端SDK
- [ ] 性能监控面板

### 计划功能 📋
- [ ] 3D指甲渲染
- [ ] 视频美甲处理
- [ ] 多风格模型支持
- [ ] 边缘计算版本
- [ ] AR集成支持

## 🤝 项目合作与商业化

### 🎯 项目愿景与目标

这个开源项目不仅是技术展示，更是寻求**产业合作**和**技术商业化**的平台。我们致力于将AI美甲技术推向实际应用，为美甲行业带来数字化变革。

### 💼 寻求合作伙伴

#### 🔥 投资合作机会
- **技术成熟度**: 已完成核心算法开发，具备商业使用水准
- **市场潜力**: 美甲行业数字化转型需求巨大，市场前景广阔
- **资金需求**: 
  - **服务器部署**: 大模型推理需要高性能GPU服务器支持
  - **算力成本**: SDXL、ControlNet等大模型运行成本较高
  - **演示环境**: 目前受限于资金，暂无在线演示服务器
  - **数据扩展**: 需要更大规模的训练数据采集和标注

#### 🎯 寻求投资伙伴类型
- **AI/计算机视觉领域投资机构**
- **美妆/美甲行业投资方**
- **技术孵化器和加速器**
- **具有GPU算力资源的云服务商**

### 💡 商业化产品

#### 🚀 **高精度指甲分割模型**
- **产品**: 第二轮微调后的U²-Net指甲识别分割模型
- **性能**: 预期达到95%+的场景适应性和IoU>0.95的分割精度
- **独家优势**: 
  - 基于大规模专业标注数据训练
  - 针对美甲场景深度优化
  - 支持多种光照、角度、手势场景
  - **非开源商业模型** - 仅对合作伙伴和客户开放

#### 🏢 **目标客户群体**
- **美甲连锁店**: 提升客户体验，实现美甲效果预览
- **美妆电商平台**: 在线试戴功能，提高转化率
- **美妆APP开发商**: 集成AI美甲功能模块
- **美甲设备制造商**: 智能美甲设备的核心算法支持

### 🎨 技术发展路线图

#### 🚀 **美甲形状专业化训练计划**

我们正在制定全面的**美甲形态识别与生成**技术路线，致力于打造业界领先的美甲AI设计平台：

##### 📐 **多样化指甲形状建模**
- **经典形状**: 圆形、方形、椭圆形、杏仁形等传统美甲形状的精准识别与渲染
- **流行趋势**: 芭蕾舞形、棺材形、尖锐形等时尚美甲造型的智能生成
- **个性定制**: 基于手指特征的个性化形状推荐与适配算法

##### 🎭 **创意美甲设计引擎**
- **图案生成**: 法式美甲、渐变色、几何图案、花纹装饰的自动化设计
- **艺术风格**: 抽象艺术、卡通动漫、节日主题等多元化创意表达
- **智能搭配**: 基于肤色、服装、场合的美甲风格智能推荐系统

##### 🔬 **深度学习优化策略**
- **多模态训练**: 融合形状、颜色、纹理、光影的综合建模
- **生成对抗网络**: 利用GANs技术实现更逼真的美甲效果生成
- **风格迁移**: 将艺术作品、自然景观等风格迁移到美甲设计中

##### 🎯 **商业应用前景**
- **设计师工具**: 为美甲师提供AI辅助设计平台，提升创作效率
- **消费者体验**: 让用户轻松尝试各种创意美甲，降低试错成本
- **行业标准**: 建立美甲AI设计的技术标准和评估体系

### 📞 商务合作联系

如果您是：
- 🎯 **投资人** - 对AI+美妆领域感兴趣
- 🏢 **企业客户** - 需要美甲AI技术解决方案  
- 🤝 **技术合作方** - 希望技术合作或授权
- 💻 **云服务商** - 可提供GPU算力支持

**欢迎联系我们洽谈合作！**

### 📱 微信联系方式

<div align="center">
  <img src="data/verification/cn.png" width="200" alt="商务合作微信二维码" />
  <p><em>扫描二维码添加微信，洽谈商务合作</em></p>
</div>

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

### 协议要点
- ✅ 商业使用
- ✅ 修改和分发
- ✅ 私人使用
- ✅ 专利使用
- ❌ 无责任保证
- ❌ 无担保

## 🙏 致谢

### 开源项目
- [Stable Diffusion](https://github.com/Stability-AI/generative-models) - SDXL AI生成模型
- [U²-Net](https://github.com/xuebinqin/U-2-Net) - 图像分割网络
- [ControlNet](https://github.com/lllyasviel/ControlNet) - 结构控制模型
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) - 图像提示适配器
- [PyTorch](https://pytorch.org) - 深度学习框架
- [OpenCV](https://opencv.org) - 计算机视觉库

### 研究参考
- Qin, X., et al. "U²-Net: Going deeper with nested U-structure for salient object detection"
- Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
- Zhang, L., et al. "Adding Conditional Control to Text-to-Image Diffusion Models"

### 数据集贡献
感谢所有为训练数据标注做出贡献的志愿者们！

## 📞 联系方式

- **项目主页**: https://github.com/dengbp/virtual-nail
- **问题反馈**: https://github.com/dengbp/virtual-nail/issues
- **功能建议**: https://github.com/dengbp/virtual-nail/discussions
- **邮箱**: dengbangpang@gmail.com

## 📊 项目统计

![GitHub stars](https://img.shields.io/github/stars/dengbp/virtual-nail)
![GitHub forks](https://img.shields.io/github/forks/dengbp/virtual-nail)
![GitHub issues](https://img.shields.io/github/issues/dengbp/virtual-nail)
![GitHub license](https://img.shields.io/github/license/dengbp/virtual-nail)

---

⭐ **如果这个项目对您有帮助，请给我们一个Star！您的支持是我们持续改进的动力。**

💬 **有问题？欢迎在Issues中讨论，我们会尽快回复！**

🚀 **想要贡献代码？查看我们的贡献指南，一起让这个项目变得更好！**

---

**最后更新**: 2025年9月17日  
**当前版本**: v2.0  
**维护状态**: 🟢 积极维护中
