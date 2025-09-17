# Reference-Only 美甲迁移方案设置指南

## 概述

本方案基于 sd-webui-controlnet v1.1.400+ 的 Reference-Only 预处理器，实现高质量的美甲颜色迁移。相比传统方案，具有以下优势：

- **简化部署**：无需额外下载 Reference-Only 权重文件
- **更好的兼容性**：与最新版本完全集成
- **更自然的色调保持**：直接参考样板图的颜色和纹理
- **减少后处理步骤**：AI直接生成高质量结果

## 系统要求

### 硬件要求
- **显卡**：NVIDIA GPU，显存 ≥ 6GB（推荐 8GB+）
- **内存**：≥ 16GB RAM
- **存储**：≥ 10GB 可用空间

### 软件要求
- **操作系统**：Windows 10/11, Linux, macOS
- **Python**：3.8+
- **CUDA**：11.8+（如果使用NVIDIA显卡）

## 安装步骤

### 1. 安装 Stable Diffusion WebUI

```bash
# 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 安装依赖
pip install -r requirements.txt

# 启动WebUI
python launch.py
```

### 2. 安装 ControlNet 扩展

```bash
# 在 stable-diffusion-webui 目录下
cd extensions

# 克隆 ControlNet 扩展
git clone https://github.com/Mikubill/sd-webui-controlnet.git

# 更新到最新版本（确保 v1.1.400+）
cd sd-webui-controlnet
git pull origin master
```

### 3. 下载必要的模型

#### 基础模型
- **SDXL 1.0**：下载 `sd_xl_base_1.0.safetensors`
- **SD 1.5**：下载 `v1-5-pruned.safetensors`

#### ControlNet 模型
- **Canny 边缘检测**：`control_v11p_sd15_canny.pth`
- **SoftEdge**：`control_v11p_sd15_softedge.pth`

### 4. 安装 Python 依赖

```bash
pip install opencv-python numpy pillow requests
```

## 配置说明

### 1. 模型文件放置

```
stable-diffusion-webui/
├── models/
│   ├── Stable-diffusion/
│   │   ├── sd_xl_base_1.0.safetensors
│   │   └── v1-5-pruned.safetensors
│   └── ControlNet/
│       ├── control_v11p_sd15_canny.pth
│       └── control_v11p_sd15_softedge.pth
```

### 2. WebUI 启动参数

```bash
python launch.py --api --listen --port 7860 --enable-insecure-extension-access
```

### 3. 验证安装

启动 WebUI 后，在 ControlNet 面板中应该能看到：
- **Preprocessor** 列表包含 `reference_only`
- **Model** 列表包含已下载的 ControlNet 模型

## 使用方法

### 1. 基本使用

```python
from nail_reference_only_transfer import NailReferenceOnlyTransfer

# 初始化迁移器
transfer = NailReferenceOnlyTransfer()

# 执行迁移
result = transfer.process_nail_transfer(
    hand_image_path="path/to/hand.jpg",
    nail_sample_path="path/to/nail_sample.jpg",
    output_path="path/to/output.jpg"
)
```

### 2. 参数调优

```python
# 自定义参数
result = transfer.transfer_with_reference_only(
    hand_image=hand_image,
    nail_sample=nail_sample,
    mask=mask,
    strength=0.8,        # Reference-Only 强度
    cfg_scale=7.0,       # CFG 比例
    steps=20             # 推理步数
)
```

### 3. 批量处理

```python
import os

# 批量处理多张图片
input_dir = "input_images"
output_dir = "output_images"
nail_sample = "nail_sample.jpg"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"result_{filename}")
        
        result = transfer.process_nail_transfer(
            hand_image_path=input_path,
            nail_sample_path=nail_sample,
            output_path=output_path
        )
```

## 参数说明

### Reference-Only 参数
- **strength**：参考强度，范围 0.0-1.0，推荐 0.6-0.9
- **controlnet_conditioning_scale**：控制网络条件缩放，推荐 0.8-1.2

### 生成参数
- **cfg_scale**：CFG 比例，推荐 7.0-8.0
- **steps**：推理步数，推荐 20-30
- **sampler_name**：采样器，推荐 "DPM++ 2M Karras"

### 后处理参数
- **shadow_intensity**：阴影强度，默认 0.2
- **highlight_intensity**：高光强度，默认 0.3

## 故障排除

### 1. API 连接失败
```
错误：API服务不可用
解决：
- 确保 WebUI 正在运行
- 检查端口 7860 是否被占用
- 确认启动参数包含 --api
```

### 2. Reference-Only 预处理器不可用
```
错误：找不到 reference_only 预处理器
解决：
- 更新 sd-webui-controlnet 到 v1.1.400+
- 重启 WebUI
- 检查扩展是否正确安装
```

### 3. 显存不足
```
错误：CUDA out of memory
解决：
- 降低图像分辨率
- 减少 batch_size
- 使用 --lowvram 启动参数
```

### 4. 模型文件缺失
```
错误：找不到 ControlNet 模型
解决：
- 下载对应的 .pth 文件
- 放置到 models/ControlNet/ 目录
- 重启 WebUI
```

## 性能优化

### 1. 显存优化
- 使用较小的图像分辨率（512x512）
- 启用 `--lowvram` 模式
- 减少 ControlNet 模型数量

### 2. 速度优化
- 使用较少的推理步数（15-20）
- 选择更快的采样器
- 批量处理时复用模型加载

### 3. 质量优化
- 使用高质量的指甲样板图
- 调整 Reference-Only 强度
- 优化掩码生成质量

## 测试验证

运行测试脚本验证安装：

```bash
python test_reference_only_transfer.py
```

测试脚本会：
1. 检查 API 连接
2. 测试掩码生成
3. 验证生长效果
4. 执行完整流程测试
5. 进行参数调优测试

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 支持 Reference-Only 预处理器
- 集成边缘控制和后处理增强
- 提供完整的测试套件

## 技术支持

如遇到问题，请：
1. 检查本文档的故障排除部分
2. 运行测试脚本诊断问题
3. 查看日志文件获取详细错误信息
4. 提交 Issue 到项目仓库

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。 