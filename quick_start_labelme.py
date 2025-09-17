#!/usr/bin/env python3
"""
LabelMe模板生成快速启动脚本
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'opencv-python',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def run_validation(labelme_dir):
    """运行数据验证"""
    print("\n=== 步骤1: 验证LabelMe数据 ===")
    
    cmd = [sys.executable, "validate_labelme_data.py", "--labelme_dir", labelme_dir]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 数据验证成功")
            return True
        else:
            print("❌ 数据验证失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行验证脚本失败: {e}")
        return False

def run_template_generation(labelme_dir, output_dir, n_clusters):
    """运行模板生成"""
    print(f"\n=== 步骤2: 生成模板 (聚类数: {n_clusters}) ===")
    
    cmd = [
        sys.executable, "generate_templates_from_labelme.py",
        "--labelme_dir", labelme_dir,
        "--output_dir", output_dir,
        "--n_clusters", str(n_clusters)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 模板生成成功")
            print("输出信息:")
            print(result.stdout)
            return True
        else:
            print("❌ 模板生成失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行模板生成脚本失败: {e}")
        return False

def run_template_testing(output_dir):
    """运行模板测试"""
    print("\n=== 步骤3: 测试生成的模板 ===")
    
    # 修改脚本中的输出目录
    test_script = "example_use_labelme_templates.py"
    
    try:
        # 读取测试脚本
        with open(test_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换输出目录
        content = content.replace(
            'Path("generated_templates/generated_templates.npz")',
            f'Path("{output_dir}/generated_templates.npz")'
        )
        
        # 写入临时文件
        temp_script = "temp_test_script.py"
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 运行测试
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True)
        
        # 清理临时文件
        os.remove(temp_script)
        
        if result.returncode == 0:
            print("✅ 模板测试成功")
            return True
        else:
            print("❌ 模板测试失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行模板测试失败: {e}")
        return False

def show_results(output_dir):
    """显示结果"""
    print(f"\n=== 生成结果 ===")
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    print(f"输出目录: {output_dir}")
    print("\n生成的文件:")
    
    files = list(output_path.glob("*"))
    for file in files:
        if file.is_file():
            size = file.stat().st_size
            print(f"  📄 {file.name} ({size} bytes)")
        else:
            print(f"  📁 {file.name}/")
    
    print(f"\n🎉 模板生成完成！")
    print(f"现在可以运行以下命令使用新模板:")
    print(f"  python test_run_purecolor.py")
    print(f"  python editor_image_server.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LabelMe模板生成快速启动")
    parser.add_argument("--labelme_dir", type=str, required=True,
                       help="LabelMe标注数据目录")
    parser.add_argument("--output_dir", type=str, default="generated_templates",
                       help="输出目录")
    parser.add_argument("--n_clusters", type=int, default=8,
                       help="聚类数量")
    parser.add_argument("--skip_validation", action="store_true",
                       help="跳过数据验证步骤")
    parser.add_argument("--skip_testing", action="store_true",
                       help="跳过模板测试步骤")
    
    args = parser.parse_args()
    
    print("🚀 LabelMe模板生成快速启动")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 验证数据
    if not args.skip_validation:
        if not run_validation(args.labelme_dir):
            print("\n❌ 数据验证失败，请检查LabelMe数据格式")
            return
    else:
        print("\n⏭️  跳过数据验证")
    
    # 生成模板
    if not run_template_generation(args.labelme_dir, args.output_dir, args.n_clusters):
        print("\n❌ 模板生成失败")
        return
    
    # 测试模板
    if not args.skip_testing:
        if not run_template_testing(args.output_dir):
            print("\n⚠️  模板测试失败，但模板已生成")
    else:
        print("\n⏭️  跳过模板测试")
    
    # 显示结果
    show_results(args.output_dir)

if __name__ == "__main__":
    main() 