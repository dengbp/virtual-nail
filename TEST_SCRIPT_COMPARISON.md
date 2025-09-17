# 测试脚本对比说明

## 脚本文件

| 版本 | 文件名 | 描述 |
|------|--------|------|
| 原始版本 | `test_run_purecolor.py` | 测试生产版本的处理函数 |
| 优化版本 | `test_run_purecolor_optimized.py` | 测试优化版本的处理函数 |

## 主要差异对比

### 1. 导入模块差异

#### 原始版本
```python
from nail_sdxl_inpaint_purecolor import process_one, input_dir, output_mask_dir, output_color_dir, output_final_dir
```

#### 优化版本
```python
from nail_sdxl_inpaint_purecolor_optimized_v2 import process_one_optimized_v2, preload_models
```

### 2. 新增功能

#### 优化版本新增参数
```python
parser.add_argument("--preload", action="store_true", help="是否预加载模型（推荐用于生产环境）")
```

#### 优化版本新增功能
- ✅ **模型预加载**：可选的模型预加载功能
- ✅ **更详细的进度显示**：显示处理进度和状态
- ✅ **错误处理增强**：更好的异常处理和错误提示
- ✅ **文件存在性检查**：检查参考图是否存在
- ✅ **处理状态反馈**：显示成功/失败状态

### 3. 使用方式对比

#### 原始版本
```bash
# 基本使用
python test_run_purecolor.py

# 指定输入目录
python test_run_purecolor.py --input_dir data/test_images

# 指定输出目录
python test_run_purecolor.py --output_final_dir data/output/final
```

#### 优化版本
```bash
# 基本使用（不预加载模型）
python test_run_purecolor_optimized.py

# 预加载模型（推荐用于生产环境）
python test_run_purecolor_optimized.py --preload

# 指定输入目录
python test_run_purecolor_optimized.py --input_dir data/test_images

# 指定输出目录
python test_run_purecolor_optimized.py --output_final_dir data/output/final

# 完整参数示例
python test_run_purecolor_optimized.py --preload --input_dir data/test_images --output_final_dir data/output/final
```

### 4. 输出信息对比

#### 原始版本输出
```
参考图路径: data/reference/reference.jpg
未找到掩码: data/test_masks/test_mask_input_mask.png，使用 U²-Net 生成掩码
掩码已生成并保存到: data/test_masks/test_mask_input_mask.png
```

#### 优化版本输出
```
============================================================
美甲渲染优化版本测试脚本
============================================================
参考图路径: data/reference/reference.jpg
跳过模型预加载（首次处理时会有加载延迟）
初始化掩码生成器...
掩码生成器初始化完成
找到 1 个图像文件

==================================================
处理第 1/1 个图像: test.jpg
==================================================
使用现有掩码: data/test_masks/test_mask_input_mask.png
开始处理图像: test.jpg
✅ 处理完成: data/output/final/test_final.png

============================================================
所有图像处理完成！
最终结果保存在: data/output/final
============================================================
```

### 5. 功能特性对比

| 功能 | 原始版本 | 优化版本 |
|------|----------|----------|
| 基本图像处理 | ✅ | ✅ |
| 掩码自动生成 | ✅ | ✅ |
| 模型预加载 | ❌ | ✅ |
| 详细进度显示 | ❌ | ✅ |
| 错误处理增强 | ❌ | ✅ |
| 文件存在性检查 | ❌ | ✅ |
| 处理状态反馈 | ❌ | ✅ |
| 美观的输出格式 | ❌ | ✅ |

### 6. 性能对比

#### 首次运行
- **原始版本**：首次处理时会有模型加载延迟
- **优化版本**：使用 `--preload` 参数可避免首次延迟

#### 连续处理
- **原始版本**：每次处理都使用相同的模型实例
- **优化版本**：使用全局模型实例，避免重复加载

### 7. 使用建议

#### 开发测试
```bash
# 快速测试，不预加载模型
python test_run_purecolor_optimized.py
```

#### 生产环境
```bash
# 预加载模型，避免首次请求延迟
python test_run_purecolor_optimized.py --preload
```

#### 批量处理
```bash
# 处理大量图像时使用预加载
python test_run_purecolor_optimized.py --preload --input_dir data/batch_images
```

## 总结

优化版本的测试脚本在保持原有功能的基础上，增加了：

1. **🚀 性能优化**：模型预加载功能
2. **📊 更好的用户体验**：详细的进度显示和状态反馈
3. **🛡️ 更强的稳定性**：增强的错误处理和文件检查
4. **🎨 更美观的输出**：格式化的输出信息

两个脚本可以独立使用，互不干扰，满足不同的使用场景需求。 