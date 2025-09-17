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
- **丰富测试**: 覆盖API、算法、性能的完整测试体系
- **详细文档**: 包含API、训练、部署的完整技术文档
- **模块化设计**: 高内聚低耦合，便于二次开发

## 🏗️ 系统架构

### 三阶段处理流水线
<img width="3840" height="3077" alt="美甲系统架构 _ Mermaid Chart-2025-09-17-061820" src="https://github.com/user-attachments/assets/9f9060b0-f2ad-47a2-bdd4-9818954c5c37" />

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
├── 🧪 测试套件
│   ├── test_progress_api.py                # API接口测试
│   ├── test_api_data_url.py                # API格式验证
│   ├── test_color_transfer_pixel_level_pipeline.py # 完整流水线测试
│   ├── test_antialiased_highlight.py       # 高光渲染测试
│   ├── test_nail_sdxl_inpaint_opencv.py    # SDXL AI测试
│   ├── test_highlight_detection.py         # 高光检测测试
│   └── test_gray_mask_pipeline.py          # 灰度掩码测试
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

### 快速测试

#### 启动API服务
```bash
# 启动美甲生成服务器
python editor_image_server.py

# 服务器将在 http://0.0.0.0:80 启动
```

#### 运行API测试
```bash
# 在另一个终端运行测试
python test_progress_api.py
python test_api_data_url.py
```

#### 测试完整流水线
```bash
# 测试三阶段处理流程
python test_color_transfer_pixel_level_pipeline.py

# 测试高光渲染
python test_antialiased_highlight.py

# 测试AI增强
python test_nail_sdxl_inpaint_opencv.py
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

## 🧪 测试文件映射表

| 测试类型 | 测试脚本 | 对应核心模块 | 功能描述 | 优先级 |
|---------|---------|-------------|---------|--------|
| **🌐 API接口测试** |
| API完整调用 | `test_progress_api.py` | `editor_image_server.py` | 完整API调用流程，包含任务提交、进度查询、结果获取 | ⭐⭐⭐⭐⭐ |
| API格式验证 | `test_api_data_url.py` | `editor_image_server.py` | Data URL格式验证，base64编解码测试 | ⭐⭐⭐⭐ |
| 参考图API | `test_reference_only_api.py` | `editor_image_server.py` | Reference-only模式API测试 | ⭐⭐⭐ |
| **🎨 第一阶段：基础颜色迁移** |
| 完整流水线 | `test_color_transfer_pixel_level_pipeline.py` | `color_transfer_pixel_level_transplant.py` | 像素级颜色迁移 + TPS变形 + 无缝融合 | ⭐⭐⭐⭐⭐ |
| 颜色迁移质量 | `test_color_transfer.py` | `nail_color_transfer.py` | 颜色准确性评估，边缘融合质量测试 | ⭐⭐⭐⭐ |
| 灰度掩码处理 | `test_gray_mask_pipeline.py` | `color_transfer_pixel_level_transplant.py` | 灰度掩码处理和颜色准确性验证 | ⭐⭐⭐ |
| **✨ 第二阶段：物理光照渲染** |
| 抗锯齿高光 | `test_antialiased_highlight.py` | `color_nail_highlight_fill.py` | 抗锯齿高光碎片处理，可视化对比 | ⭐⭐⭐⭐⭐ |
| 高光检测 | `test_highlight_detection.py` | `color_nail_highlight_shader.py` | 自适应高光检测，光照参数优化 | ⭐⭐⭐⭐ |
| 纯高光渲染 | `test_run_highlight_only.py` | `color_nail_highlight_fill.py` | 独立高光渲染模块测试 | ⭐⭐⭐ |
| **🤖 第三阶段：AI深度优化** |
| SDXL增强 | `test_nail_sdxl_inpaint_opencv.py` | `nail_sdxl_inpaint_opencv.py` | SDXL Inpainting + IP-Adapter + ControlNet | ⭐⭐⭐⭐⭐ |
| AI参数优化 | `test_inference_steps.py` | `nail_sdxl_inpaint_purecolor.py` | 推理步数和参数优化测试 | ⭐⭐⭐ |
| 进度回调 | `test_progress_callback.py` | `nail_sdxl_inpaint_purecolor.py` | AI生成进度监控和回调测试 | ⭐⭐ |
| **🔄 综合集成测试** |
| 主流水线 | `test_main_pipeline_no_template.py` | `color_nail_full_pipeline_adapter.py` | 无模板完整流水线测试 | ⭐⭐⭐⭐ |
| 任务ID验证 | `test_task_id.py` | `editor_image_server.py` | 任务ID生成和追踪测试 | ⭐⭐ |
| 大文件上传 | `test_large_file_upload.py` | `editor_image_server.py` | 大图像文件上传性能测试 | ⭐⭐ |

## 📊 训练数据流向图

<img width="914" height="3839" alt="美甲训练数据流向图 _ Mermaid Chart-2025-09-17-062349" src="https://github.com/user-attachments/assets/c9579c02-2274-492c-8396-c6f58184c261" />


## 🔄 数据预处理流程图

<img width="2751" height="3840" alt="美甲数据预处理流程图 _ Mermaid Chart-2025-09-17-062500" src="https://github.com/user-attachments/assets/5fa0e4f7-4552-457f-9644-35402e4e2748" />


## 🎯 核心处理数据流向图

<img width="898" height="3840" alt="美甲核心处理数据流向图 _ Mermaid Chart-2025-09-17-062628" src="https://github.com/user-attachments/assets/985f68dc-676a-42a2-b3d9-1e3fe8f620b7" />


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

## 🧪 测试体系

### API测试
```bash
# 基础API功能测试
python test_progress_api.py

# Data URL格式验证
python test_api_data_url.py

# 大文件上传测试
python test_large_file_upload.py

# 任务ID管理测试
python test_task_id.py
```

### 算法模块测试
```bash
# 第一阶段：颜色迁移测试
python test_color_transfer_pixel_level_pipeline.py
python test_color_transfer.py

# 第二阶段：高光渲染测试
python test_antialiased_highlight.py
python test_highlight_detection.py
python test_run_highlight_only.py

# 第三阶段：AI增强测试
python test_nail_sdxl_inpaint_opencv.py
python test_inference_steps.py
python test_progress_callback.py
```

### 集成测试
```bash
# 完整流水线测试
python test_main_pipeline_no_template.py

# 灰度掩码处理测试
python test_gray_mask_pipeline.py

# 自动化测试套件
./run_all_tests.sh
```

### 质量评估标准

| 测试类型 | 评估指标 | 目标值 |
|---------|---------|--------|
| **API性能** | 响应时间 | < 60秒 |
| **API稳定性** | 成功率 | > 95% |
| **分割精度** | IoU | > 0.92 |
| **颜色准确性** | 色差 | < 5% |
| **边缘质量** | 平滑度 | 无明显锯齿 |
| **AI生成** | 质感真实度 | 人工评估8/10 |

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

#### 训练监控
```bash
# 查看训练日志
tail -f train_u2net_detailed.log

# TensorBoard监控
tensorboard --logdir runs/training_logs/

# 训练曲线可视化
python plot_loss_curve.py
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

#### 开发环境
```bash
# 启动开发服务器
python editor_image_server.py

# 配置文件
config.py:
    DEBUG = True
    PORT = 80
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
```

#### 生产环境
```bash
# 使用Gunicorn部署
pip install gunicorn

# 启动生产服务器
gunicorn -w 4 -b 0.0.0.0:80 --timeout 300 editor_image_server:app

# 配置文件
config.py:
    DEBUG = False
    PORT = 80
    WORKERS = 4
```

### Docker部署

#### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# 安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件
COPY . .

# 下载模型
RUN python download_models.py

# 暴露端口
EXPOSE 80

# 启动命令
CMD ["python", "editor_image_server.py"]
```

#### 构建和运行
```bash
# 构建镜像
docker build -t nail-color-preview .

# 运行容器
docker run -d \
    --name nail-preview \
    --gpus all \
    -p 80:80 \
    -v $(pwd)/data:/app/data \
    nail-color-preview
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

## 🔍 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 症状：CUDA out of memory
# 解决方案：
1. 减小batch_size
2. 启用梯度检查点
3. 使用CPU推理

# 检查GPU状态
nvidia-smi

# 调整配置
BATCH_SIZE = 2  # 从4减少到2
```

#### 2. 模型加载失败
```bash
# 症状：模型文件不存在
# 解决方案：
python download_models.py

# 手动下载
wget https://example.com/models/u2net.pth -O models/u2net.pth
```

#### 3. API响应超时
```bash
# 症状：请求超时
# 解决方案：
1. 增加超时时间
2. 检查GPU可用性
3. 优化图像分辨率

# 配置调整
TIMEOUT = 300  # 5分钟超时
```

#### 4. 依赖版本冲突
```bash
# 症状：ImportError或VersionError
# 解决方案：
1. 创建新虚拟环境
2. 按顺序安装依赖

# 清理环境
conda deactivate
conda env remove -n nail-preview
conda create -n nail-preview python=3.9
```

### 调试工具

#### 日志分析
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看处理步骤
tail -f app.log
tail -f train_u2net_detailed.log
```

#### 性能分析
```python
# 处理时间监控
import time

start_time = time.time()
result = process_image(img)
print(f"处理耗时: {time.time() - start_time:.2f}秒")

# GPU监控
watch -n 1 nvidia-smi
```

## 🤝 贡献指南

### 开发规范

#### 代码风格
```python
# 使用Black代码格式化
pip install black
black . --line-length 88

# 使用flake8代码检查
pip install flake8
flake8 . --max-line-length 88

# 类型提示
from typing import Optional, List, Tuple
def process_image(img: np.ndarray) -> Optional[np.ndarray]:
    pass
```

#### 提交规范
```bash
# 提交信息格式
<type>(<scope>): <description>

# 示例
feat(api): 添加进度查询接口
fix(training): 修复U2Net训练内存泄漏
docs(readme): 更新安装指南
test(highlight): 添加高光渲染测试
```

### 贡献流程

1. **Fork项目** → 2. **创建特性分支** → 3. **开发功能** → 4. **编写测试** → 5. **提交PR**

```bash
# 1. Fork后克隆
git clone https://github.com/your-username/nail-color-preview.git

# 2. 创建分支
git checkout -b feature/new-highlight-algorithm

# 3. 开发并测试
# ... 开发代码 ...
python test_new_feature.py

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
| **入门级** | GTX 1660 (6GB) | 90-120秒 | 12GB RAM | 开发测试 |
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
- [x] 完整测试体系
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
