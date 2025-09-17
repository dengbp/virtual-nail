
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
logging.info("测试特殊字符: À\x13À")
logging.info("测试正常英文: Hello World")

print("测试完成，请检查 test_fix.log 文件")
