# LabelMe指甲模板生成指南

## 概述

这个工具可以从LabelMe标注的指甲图像数据中自动提取指甲轮廓，通过聚类分析生成主流指甲形状模板，并集成到指甲颜色预览系统中。

## 数据要求

### LabelMe标注格式
- 标注文件：`.json`格式
- 标注标签：`nail`、`指甲`、`nail_region`（不区分大小写）
- 标注类型：多边形（polygon）或矩形（rectangle）
- 数据量：建议至少50个标注样本，越多越好

### 标注示例
```json
{
  "shapes": [
    {
      "label": "nail",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "shape_type": "polygon"
    }
  ]
}
```

## 使用方法

### 1. 生成模板

```bash
# 基本用法
python generate_templates_from_labelme.py --labelme_dir /path/to/labelme/data

# 指定输出目录和聚类数量
python generate_templates_from_labelme.py \
    --labelme_dir /path/to/labelme/data \
    --output_dir my_templates \
    --n_clusters 10
```

### 2. 测试模板

```bash
python example_use_labelme_templates.py
```

### 3. 集成到主流水线

模板会自动集成到现有的指甲颜色预览系统中，无需额外配置。

## 输出文件

生成的文件包括：

- `generated_templates.npz` - 模板数据（numpy格式）
- `generated_templates_visualization.png` - 模板可视化图像
- `generated_templates_code.py` - 可直接导入的Python代码
- `cluster_info.json` - 聚类信息（样本分布等）

## 技术原理

### 1. 轮廓提取
- 从LabelMe JSON文件中提取多边形点
- 自动闭合轮廓
- 过滤无效轮廓（点数过少等）

### 2. 轮廓归一化
- 中心化：将轮廓中心移到原点
- 缩放：统一到单位大小
- 重采样：每个轮廓统一为100个点

### 3. 特征提取
- 几何特征：长宽比、实心度、曲率、对称性
- 形状特征：轮廓点坐标
- 组合特征：几何特征 + 形状特征

### 4. 聚类分析
- 使用K-means算法
- 特征标准化
- 自动确定最佳聚类数

### 5. 模板生成
- 计算每个聚类的平均轮廓
- 生成代表性模板
- 保存多种格式

## 参数调优

### 聚类数量
- 默认：8个模板
- 建议：根据数据量和多样性调整
- 原则：每个聚类至少5个样本

### 轮廓点数
- 默认：100个点
- 影响：点数越多精度越高，但计算量越大
- 建议：50-200之间

### 特征权重
可以在代码中调整几何特征和形状特征的权重：
```python
# 在 _extract_contour_features 方法中
basic_features = [aspect_ratio, solidity, curvature, symmetry]
all_features = np.concatenate([basic_features, contour_flat])
```

## 质量评估

### 聚类质量指标
- 轮廓数量分布：每个聚类的样本数
- 轮廓相似度：聚类内轮廓的相似程度
- 聚类间距离：不同聚类的区分度

### 模板质量检查
- 轮廓平滑度：无锯齿、无自交
- 形状合理性：符合指甲自然形状
- 覆盖度：能覆盖大部分真实指甲形状

## 故障排除

### 常见问题

1. **数据不足**
   ```
   错误：数据不足，需要至少 X 个轮廓，当前只有 Y 个
   解决：增加标注数据或减少聚类数量
   ```

2. **标注格式错误**
   ```
   错误：处理文件 xxx.json 时出错
   解决：检查JSON格式和标注标签
   ```

3. **轮廓质量差**
   ```
   问题：生成的模板形状异常
   解决：检查原始标注质量，过滤异常轮廓
   ```

### 调试技巧

1. **查看中间结果**
   ```python
   # 在代码中添加调试输出
   print(f"提取了 {len(contours)} 个轮廓")
   print(f"聚类结果: {np.bincount(cluster_labels)}")
   ```

2. **可视化检查**
   - 查看 `generated_templates_visualization.png`
   - 检查每个聚类的样本分布
   - 验证模板形状合理性

3. **参数实验**
   - 尝试不同的聚类数量
   - 调整特征提取方法
   - 修改归一化参数

## 高级功能

### 自定义特征提取
可以扩展特征提取方法：
```python
def custom_feature_extraction(contour):
    # 添加自定义特征
    # 例如：指甲弧度、边缘平滑度等
    pass
```

### 多级聚类
对于大量数据，可以使用层次聚类：
```python
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=n_clusters)
```

### 模板验证
添加模板质量验证：
```python
def validate_template(template):
    # 检查模板质量
    # 返回质量分数
    pass
```

## 集成说明

生成的模板会自动替换 `nail_template_fitter.py` 中的默认模板，影响：

1. **批量处理脚本** (`test_run_purecolor.py`)
2. **服务器脚本** (`editor_image_server.py`)
3. **所有使用模板拟合的功能**

原始模板文件会备份为 `.backup` 文件，可以随时恢复。

## 最佳实践

1. **数据准备**
   - 标注多样化的指甲形状
   - 确保标注精度
   - 包含不同角度和光照条件

2. **参数选择**
   - 根据数据量调整聚类数
   - 平衡精度和计算效率
   - 定期验证模板质量

3. **持续优化**
   - 收集更多标注数据
   - 定期重新生成模板
   - 根据使用效果调整参数

## 联系支持

如果遇到问题，请提供：
- LabelMe数据示例
- 错误信息
- 系统环境信息
- 期望的模板数量和质量 