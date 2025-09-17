import platform
import psutil
import torch

def get_system_info():
    print("系统信息：")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    print(f"物理内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Python 版本: {platform.python_version()}")
    
    print("\nPyTorch 信息：")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

if __name__ == "__main__":
    get_system_info() 