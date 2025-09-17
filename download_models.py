#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import requests
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import hashlib
from huggingface_hub import hf_hub_download

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("model_download.log", rotation="10 MB")

# Model configurations
MODELS = {
    "sam_vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sha256": "7c2b7c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0",
        "size": 2.4 * 1024 * 1024 * 1024,  # 2.4GB
        "description": "Segment Anything Model (SAM) - 用于精确的指甲区域分割"
    },
    "grounding_dino": {
        "filename": "groundingdino_swint_ogc.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "sha256": "7c2b7c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0",
        "size": 1.2 * 1024 * 1024 * 1024,  # 1.2GB
        "description": "Grounding DINO - 用于手部检测和定位"
    }
}

def calculate_sha256(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, filename, expected_size):
    """Download a file with progress bar and size verification."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            logger.warning(f"无法获取文件大小: {filename}")
            total_size = expected_size
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        # Verify file size
        actual_size = os.path.getsize(filename)
        if actual_size != expected_size:
            logger.warning(f"文件大小不匹配 {filename}. 预期: {expected_size}, 实际: {actual_size}")
            return False
        return True

    except Exception as e:
        logger.error(f"下载失败 {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def download_models():
    """Download all required models."""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA可用: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Check disk space
    free_space = models_dir.stat().st_fsize
    required_space = sum(model["size"] for model in MODELS.values())
    if free_space < required_space:
        logger.warning(f"可用磁盘空间可能不足. 需要: {required_space/1024**3:.1f}GB")

    # Download each model
    for model_name, model_info in MODELS.items():
        model_path = models_dir / model_info["filename"]
        
        logger.info(f"\n准备下载: {model_name}")
        logger.info(f"描述: {model_info['description']}")
        logger.info(f"大小: {model_info['size']/1024**3:.1f}GB")
        
        # Skip if file exists and has correct size
        if model_path.exists():
            actual_size = os.path.getsize(model_path)
            if actual_size == model_info["size"]:
                logger.info(f"{model_name} 已存在且大小正确，跳过下载")
                continue
    else:
                logger.warning(f"{model_name} 存在但大小不正确，重新下载...")
                os.remove(model_path)

        logger.info(f"开始下载 {model_name}...")
        success = download_file(
            model_info["url"],
            model_path,
            model_info["size"]
        )

        if success:
            # Verify SHA256 (if provided)
            if model_info["sha256"] != "7c2b7c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0":
                actual_sha256 = calculate_sha256(model_path)
                if actual_sha256 != model_info["sha256"]:
                    logger.error(f"{model_name} SHA256验证失败")
                    os.remove(model_path)
                    return False
            logger.success(f"成功下载 {model_name}")
        else:
            logger.error(f"下载 {model_name} 失败")
            return False

    logger.success("\n所有模型下载完成！")
    logger.info(f"模型文件保存在: {models_dir.absolute()}")
    return True

if __name__ == "__main__":
    try:
        print("\n=== 指甲颜色预览系统 - 模型下载工具 ===\n")
        print("本工具将下载以下模型：")
        for model_name, model_info in MODELS.items():
            print(f"\n{model_name}:")
            print(f"- 描述: {model_info['description']}")
            print(f"- 大小: {model_info['size']/1024**3:.1f}GB")
        
        print("\n开始下载...\n")
        success = download_models()
        if not success:
            logger.error("模型下载失败，请检查日志了解详情")
            sys.exit(1)
    except Exception as e:
        logger.exception("下载过程中发生意外错误")
            sys.exit(1)