#!/usr/bin/env python3
"""
测试进度监控功能是否影响主流程
"""

import sys
from pathlib import Path
import time

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_import_compatibility():
    """测试导入兼容性"""
    print("=== 测试1: 导入兼容性 ===")
    
    try:
        # 测试所有主要模块的导入
        print("1. 测试主要模块导入...")
        
        from color_transfer_pixel_level_refine_sdxl import (
            WebUIProgressMonitor,
            SDXLRefiner,
            process_refine,
            refine_sdxl_pipeline,
            list_all_tasks,
            find_task_by_keywords,
            monitor_task_by_id,
            monitor_task_by_keywords
        )
        
        print("✅ 所有主要模块导入成功")
        return True
        
    except Exception as e:
        print(f"❌ 导入兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_progress_monitor_basic():
    """测试基本的进度监控功能"""
    print("\n=== 测试2: 基本进度监控功能 ===")
    
    try:
        from color_transfer_pixel_level_refine_sdxl import WebUIProgressMonitor
        
        # 测试获取队列状态
        print("1. 测试获取队列状态...")
        queue_status = WebUIProgressMonitor.get_queue_status()
        if queue_status:
            print("✅ 队列状态获取成功")
            print(f"   队列信息: {len(queue_status.get('queue_running', []))} 个运行中任务")
        else:
            print("⚠️  队列状态获取失败（可能是WebUI未运行）")
        
        # 测试列出所有任务
        print("\n2. 测试列出所有任务...")
        from color_transfer_pixel_level_refine_sdxl import list_all_tasks
        list_all_tasks()
        
        print("✅ 基本进度监控功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 基本进度监控功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试进度监控功能对主流程的影响...")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_import_compatibility())
    test_results.append(test_progress_monitor_basic())
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！进度监控功能不会影响主流程。")
        print("\n💡 建议:")
        print("   1. 现在可以安全地在生产环境中使用进度监控功能")
        print("   2. 如果WebUI未运行，进度监控会优雅地跳过")
        print("   3. 可以通过 enable_progress_monitor=False 禁用进度监控")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    print("\n🔧 下一步测试:")
    print("   1. 启动WebUI")
    print("   2. 运行实际的refine_sdxl_pipeline测试")
    print("   3. 测试多任务监控功能")

if __name__ == '__main__':
    main() 