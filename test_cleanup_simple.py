#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的清理功能测试，使用已存在的任务文件
"""

import os
import requests

def test_cleanup_with_existing_files():
    """使用已存在的任务文件测试清理功能"""
    print("使用已存在的任务文件测试清理功能")
    print("=" * 60)
    
    # 使用一个已存在的任务ID
    task_id = "175647868"
    
    # API配置
    base_url = "http://localhost:80"
    
    # 定义文件路径
    img_path = os.path.join("data/test_images", task_id + ".jpg")
    ref_path = os.path.join("data/reference", task_id + "_reference.jpg")
    mask_path = os.path.join("data/test_masks", task_id + "_mask.jpg")
    final_path = os.path.join("data/output/final", task_id + "_final.png")
    
    print(f"测试任务ID: {task_id}")
    print("\n1. 检查清理前的文件状态:")
    
    def check_file_exists(file_path, description):
        exists = os.path.exists(file_path)
        print(f"  {description}: {'✅ 存在' if exists else '❌ 不存在'} - {file_path}")
        return exists
    
    original_exists = check_file_exists(img_path, "原图")
    reference_exists = check_file_exists(ref_path, "参考色块图")
    mask_exists = check_file_exists(mask_path, "掩码文件")
    final_exists = check_file_exists(final_path, "最终效果图")
    
    print("\n2. 执行清理操作...")
    response = requests.get(f"{base_url}/cleanup_task/{task_id}")
    
    if response.status_code != 200:
        print(f"❌ 清理任务失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False
    
    result = response.json()
    if result['statusCode'] != 200:
        print(f"❌ 清理任务失败: {result['message']}")
        return False
    
    print(f"✅ 清理结果: {result['message']}")
    print(f"清理的文件: {result.get('cleaned_files', [])}")
    print(f"保留的文件: {result.get('preserved_files', [])}")
    
    print("\n3. 检查清理后的文件状态:")
    original_after = check_file_exists(img_path, "原图")
    reference_after = check_file_exists(ref_path, "参考色块图")
    mask_after = check_file_exists(mask_path, "掩码文件")
    final_after = check_file_exists(final_path, "最终效果图")
    
    print("\n4. 验证清理结果...")
    
    # 验证重要文件是否被保留
    important_files_preserved = True
    if not original_after:
        print("❌ 原图文件被意外删除")
        important_files_preserved = False
    if not reference_after:
        print("❌ 参考色块图文件被意外删除")
        important_files_preserved = False
    if not final_after:
        print("❌ 最终效果图文件被意外删除")
        important_files_preserved = False
    
    # 验证中间文件是否被清理
    intermediate_files_cleaned = True
    if mask_after:
        print("❌ 掩码文件未被清理")
        intermediate_files_cleaned = False
    
    if important_files_preserved and intermediate_files_cleaned:
        print("✅ 清理功能验证通过：")
        print("  - 重要文件（原图、参考色块图、最终效果图）已正确保留")
        print("  - 中间处理文件（掩码）已正确清理")
        return True
    else:
        print("❌ 清理功能验证失败")
        return False

if __name__ == "__main__":
    print("简单清理功能测试")
    print("=" * 60)
    
    try:
        success = test_cleanup_with_existing_files()
        if success:
            print("\n✅ 测试完成！")
        else:
            print("\n❌ 测试失败！")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc() 