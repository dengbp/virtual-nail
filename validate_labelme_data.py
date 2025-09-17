#!/usr/bin/env python3
"""
验证LabelMe标注数据格式和内容
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelMeValidator:
    """LabelMe数据验证器"""
    
    def __init__(self, labelme_dir: str):
        self.labelme_dir = Path(labelme_dir)
        
    def validate_directory(self):
        """验证目录结构"""
        logger.info(f"验证目录: {self.labelme_dir}")
        
        if not self.labelme_dir.exists():
            logger.error(f"目录不存在: {self.labelme_dir}")
            return False
        
        if not self.labelme_dir.is_dir():
            logger.error(f"不是目录: {self.labelme_dir}")
            return False
        
        # 查找JSON文件
        json_files = list(self.labelme_dir.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        
        if len(json_files) == 0:
            logger.warning("未找到JSON文件")
            return False
        
        return True
    
    def validate_json_format(self, json_file):
        """验证单个JSON文件格式"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必需字段
            required_fields = ['shapes', 'imagePath', 'imageData']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"{json_file.name}: 缺少字段 '{field}'")
                    return False
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{json_file.name}: JSON格式错误 - {e}")
            return False
        except Exception as e:
            logger.error(f"{json_file.name}: 读取错误 - {e}")
            return False
    
    def extract_nail_annotations(self, json_file):
        """提取指甲标注"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            nail_shapes = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                
                # 检查是否是指甲标注
                if label in ['nail', '指甲', 'nail_region']:
                    points = shape.get('points', [])
                    shape_type = shape.get('shape_type', '')
                    
                    if len(points) >= 3:
                        nail_shapes.append({
                            'label': label,
                            'points': points,
                            'shape_type': shape_type,
                            'file': json_file.name
                        })
                    else:
                        logger.warning(f"{json_file.name}: 标注 '{label}' 点数不足 ({len(points)})")
            
            return nail_shapes
            
        except Exception as e:
            logger.error(f"{json_file.name}: 提取标注失败 - {e}")
            return []
    
    def analyze_annotations(self):
        """分析所有标注"""
        logger.info("分析标注数据...")
        
        json_files = list(self.labelme_dir.glob("*.json"))
        all_nail_shapes = []
        
        valid_files = 0
        total_annotations = 0
        
        for json_file in json_files:
            # 验证JSON格式
            if not self.validate_json_format(json_file):
                continue
            
            valid_files += 1
            
            # 提取指甲标注
            nail_shapes = self.extract_nail_annotations(json_file)
            all_nail_shapes.extend(nail_shapes)
            total_annotations += len(nail_shapes)
        
        # 统计信息
        logger.info(f"有效文件: {valid_files}/{len(json_files)}")
        logger.info(f"总标注数: {total_annotations}")
        
        # 按标签统计
        label_counts = {}
        shape_type_counts = {}
        
        for shape in all_nail_shapes:
            label = shape['label']
            shape_type = shape['shape_type']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            shape_type_counts[shape_type] = shape_type_counts.get(shape_type, 0) + 1
        
        logger.info("标签统计:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count}")
        
        logger.info("形状类型统计:")
        for shape_type, count in shape_type_counts.items():
            logger.info(f"  {shape_type}: {count}")
        
        return all_nail_shapes
    
    def visualize_annotations(self, nail_shapes, max_samples=10):
        """可视化标注样本"""
        logger.info(f"可视化前 {min(max_samples, len(nail_shapes))} 个标注样本...")
        
        # 创建可视化图像
        canvas_width = 800
        canvas_height = 200 * ((len(nail_shapes) + 4) // 5)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        for i, shape in enumerate(nail_shapes[:max_samples]):
            row = i // 5
            col = i % 5
            
            # 计算位置
            x_offset = col * (canvas_width // 5) + (canvas_width // 5) // 2
            y_offset = row * 200 + 100
            
            # 转换点坐标
            points = np.array(shape['points'], dtype=np.float32)
            
            # 计算边界框
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            # 缩放以适应显示区域
            scale = min(80 / (x_max - x_min), 40 / (y_max - y_min)) if (x_max - x_min) > 0 and (y_max - y_min) > 0 else 1
            
            # 缩放和移动
            scaled_points = (points - np.array([x_min, y_min])) * scale
            scaled_points[:, 0] += x_offset - (x_max - x_min) * scale / 2
            scaled_points[:, 1] += y_offset - (y_max - y_min) * scale / 2
            
            # 绘制轮廓
            scaled_points = scaled_points.astype(np.int32)
            cv2.polylines(canvas, [scaled_points], True, (0, 0, 255), 2)
            
            # 添加文字
            text_x = col * (canvas_width // 5) + 10
            text_y = row * 200 + 30
            label_text = f"{shape['file'][:10]}...{shape['label']}"
            cv2.putText(canvas, label_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 保存图像
        vis_file = self.labelme_dir / "annotation_validation.png"
        cv2.imwrite(str(vis_file), canvas)
        logger.info(f"可视化结果已保存: {vis_file}")
        
        return vis_file
    
    def check_data_quality(self, nail_shapes):
        """检查数据质量"""
        logger.info("检查数据质量...")
        
        if len(nail_shapes) == 0:
            logger.warning("没有找到有效的指甲标注")
            return False
        
        # 检查点数分布
        point_counts = [len(shape['points']) for shape in nail_shapes]
        min_points = min(point_counts)
        max_points = max(point_counts)
        avg_points = np.mean(point_counts)
        
        logger.info(f"点数统计: 最小={min_points}, 最大={max_points}, 平均={avg_points:.1f}")
        
        # 检查轮廓质量
        quality_issues = []
        
        for shape in nail_shapes:
            points = np.array(shape['points'])
            
            # 检查点数
            if len(points) < 5:
                quality_issues.append(f"{shape['file']}: 点数过少 ({len(points)})")
            
            # 检查自交
            if self._check_self_intersection(points):
                quality_issues.append(f"{shape['file']}: 轮廓自交")
            
            # 检查面积
            area = cv2.contourArea(points.astype(np.float32))
            if area < 10:
                quality_issues.append(f"{shape['file']}: 面积过小 ({area:.1f})")
        
        if quality_issues:
            logger.warning("发现质量问题:")
            for issue in quality_issues[:10]:  # 只显示前10个
                logger.warning(f"  {issue}")
            if len(quality_issues) > 10:
                logger.warning(f"  ... 还有 {len(quality_issues) - 10} 个问题")
        else:
            logger.info("数据质量良好")
        
        return len(quality_issues) == 0
    
    def _check_self_intersection(self, points):
        """检查轮廓是否自交（简化版本）"""
        if len(points) < 4:
            return False
        
        # 简单的自交检查：检查相邻线段是否相交
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            for j in range(i + 2, len(points)):
                p3 = points[j]
                p4 = points[(j + 1) % len(points)]
                
                # 跳过相邻线段
                if (j + 1) % len(points) == i:
                    continue
                
                # 检查线段相交
                if self._line_intersection(p1, p2, p3, p4):
                    return True
        
        return False
    
    def _line_intersection(self, p1, p2, p3, p4):
        """检查两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def generate_report(self):
        """生成验证报告"""
        logger.info("生成验证报告...")
        
        # 验证目录
        if not self.validate_directory():
            return False
        
        # 分析标注
        nail_shapes = self.analyze_annotations()
        
        if len(nail_shapes) == 0:
            logger.error("没有找到有效的指甲标注")
            return False
        
        # 检查数据质量
        quality_ok = self.check_data_quality(nail_shapes)
        
        # 可视化样本
        vis_file = self.visualize_annotations(nail_shapes)
        
        # 生成报告
        report_file = self.labelme_dir / "validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("LabelMe数据验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"数据目录: {self.labelme_dir}\n")
            f.write(f"有效文件数: {len(list(self.labelme_dir.glob('*.json')))}\n")
            f.write(f"指甲标注数: {len(nail_shapes)}\n")
            f.write(f"数据质量: {'良好' if quality_ok else '需要改进'}\n\n")
            
            f.write("标签分布:\n")
            label_counts = {}
            for shape in nail_shapes:
                label = shape['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            for label, count in label_counts.items():
                f.write(f"  {label}: {count}\n")
            
            f.write(f"\n可视化文件: {vis_file}\n")
        
        logger.info(f"验证报告已保存: {report_file}")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证LabelMe标注数据")
    parser.add_argument("--labelme_dir", type=str, required=True,
                       help="LabelMe数据目录")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = LabelMeValidator(args.labelme_dir)
    
    # 运行验证
    success = validator.generate_report()
    
    if success:
        print("\n✅ 数据验证完成！")
        print("可以运行 generate_templates_from_labelme.py 生成模板")
    else:
        print("\n❌ 数据验证失败，请检查数据格式")

if __name__ == "__main__":
    main() 