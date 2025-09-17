#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键修复编码问题脚本
使用方法: python fix_encoding_issues.py
"""

import os
import sys
import re
import glob
from pathlib import Path

def fix_logging_config_in_file(file_path):
    """
    修复单个文件中的日志配置
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复 logging.basicConfig 调用
        patterns = [
            # 修复 FileHandler 缺少编码参数
            (
                r'logging\.FileHandler\(([^)]+)\)',
                r'logging.FileHandler(\1, encoding="utf-8")'
            ),
            # 修复 StreamHandler 使用 stderr
            (
                r'logging\.StreamHandler\(\)',
                r'logging.StreamHandler(sys.stdout)'
            ),
            # 添加 sys 导入（如果需要）
            (
                r'import logging',
                r'import logging\nimport sys'
            )
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 已修复: {file_path}")
            return True
        else:
            print(f"- 无需修复: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ 修复失败: {file_path} - {str(e)}")
        return False

def find_python_files(directory="."):
    """
    查找所有Python文件
    """
    python_files = []
    
    # 查找所有 .py 文件
    for pattern in ["*.py", "**/*.py"]:
        python_files.extend(glob.glob(pattern, recursive=True))
    
    # 排除虚拟环境目录
    python_files = [f for f in python_files if not any(exclude in f for exclude in [
        'venv/', 'venv_new/', '__pycache__/', '.git/'
    ])]
    
    return python_files

def check_logging_usage(file_path):
    """
    检查文件是否使用了日志功能
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含日志相关代码
        logging_patterns = [
            r'logging\.basicConfig',
            r'logging\.FileHandler',
            r'logging\.StreamHandler',
            r'logging\.getLogger',
            r'logger\.',
        ]
        
        for pattern in logging_patterns:
            if re.search(pattern, content):
                return True
        
        return False
        
    except Exception:
        return False

def setup_environment():
    """
    设置环境变量
    """
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 重新配置标准输出（如果可能）
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.7 以下版本不支持 reconfigure
        pass

def create_backup(file_path):
    """
    创建文件备份
    """
    backup_path = f"{file_path}.backup"
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"警告: 无法创建备份 {file_path}: {str(e)}")
        return None

def main():
    """
    主函数
    """
    print("=" * 60)
    print("编码问题一键修复工具")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    print("✓ 环境变量已设置")
    
    # 查找Python文件
    python_files = find_python_files()
    print(f"✓ 找到 {len(python_files)} 个Python文件")
    
    # 筛选包含日志的文件
    logging_files = [f for f in python_files if check_logging_usage(f)]
    print(f"✓ 其中 {len(logging_files)} 个文件包含日志代码")
    
    if not logging_files:
        print("未找到需要修复的文件")
        return
    
    # 修复文件
    fixed_count = 0
    for file_path in logging_files:
        print(f"\n处理: {file_path}")
        
        # 创建备份
        backup_path = create_backup(file_path)
        if backup_path:
            print(f"  备份: {backup_path}")
        
        # 修复文件
        if fix_logging_config_in_file(file_path):
            fixed_count += 1
    
    print("\n" + "=" * 60)
    print(f"修复完成！共修复了 {fixed_count} 个文件")
    print("=" * 60)
    
    # 提供后续建议
    print("\n后续建议:")
    print("1. 测试您的应用是否正常运行")
    print("2. 检查日志文件是否正确生成")
    print("3. 如果遇到问题，可以使用 .backup 文件恢复")
    print("4. 考虑使用 logging_config.py 进行更安全的日志配置")
    
    # 创建测试脚本
    test_script = """
# 测试脚本
import logging
import sys

# 测试日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_fix.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 测试各种字符
logging.info("测试中文日志")
logging.info("测试特殊字符: À\\x13À")
logging.info("测试正常英文: Hello World")

print("测试完成，请检查 test_fix.log 文件")
"""
    
    with open('test_encoding_fix.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("\n已创建测试脚本: test_encoding_fix.py")
    print("运行 'python test_encoding_fix.py' 来验证修复效果")

if __name__ == "__main__":
    main() 