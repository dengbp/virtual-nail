# -*- coding: utf-8 -*-
"""
train_u2net_memory_optimized.py - 内存优化的U²-Net训练脚本

针对GPU内存爆满问题的优化方案：
1. 减少批次大小到2
2. 降低图像分辨率到512
3. 简化数据增强
4. 启用梯度累积
5. 优化内存管理
6. 添加内存监控

作者：AI助手优化
"""
import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import time
from pathlib import Path
import cv2
import gc
import psutil
import GPUtil

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_u2net_memory_optimized.log', encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# 内存监控函数
def get_memory_info():
    """获取内存使用情况"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # 使用第一个GPU
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        else:
            gpu_memory_percent = 0
            
        ram_percent = psutil.virtual_memory().percent
        return gpu_memory_percent, ram_percent
    except:
        return 0, 0

def log_memory_usage(stage=""):
    """记录内存使用情况"""
    gpu_percent, ram_percent = get_memory_info()
    logging.info(f"内存使用 {stage}: GPU {gpu_percent:.1f}%, RAM {ram_percent:.1f}%")

# ------------------ 内存优化的数据集 ------------------
class MemoryOptimizedNailDataset(Dataset):
    """内存优化的数据集，减少数据增强复杂度"""
    def __init__(self, image_paths, mask_paths, max_size=512, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.max_size = max_size
        self.is_train = is_train
        
        # 简化的数据增强策略
        if is_train:
            self.transform = A.Compose([
                # 基础几何变换
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5
                ),
                
                # 基础缩放
                A.LongestMaxSize(max_size=self.max_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=cv2.BORDER_CONSTANT),
                
                # 简化光照变换
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # 验证时只做基本处理
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=self.max_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像和掩码
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        
        # 应用变换
        transformed = self.transform(image=image, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        
        # 确保掩码是浮点数且范围在[0,1]
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # 添加通道维度
        
        return img, mask

# ------------------ U2NET模型结构 ------------------
from generate_initial_masks import U2NET

# ------------------ 简化的损失函数 ------------------
class SimpleLoss(nn.Module):
    """简单的损失函数，只使用主输出"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target):
        # 只使用主输出 (d0)
        pred = outputs[0]
        
        # 确保形状匹配
        if pred.shape != target.shape:
            target = nn.functional.interpolate(target, size=pred.shape[2:], mode='nearest')
        
        return self.bce(pred, target)

# ------------------ 简化的评估指标 ------------------
def calculate_metrics_simple(pred, target, threshold=0.5):
    """简化的评估指标计算"""
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = (target > threshold).float()
    
    # IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()
    
    # Dice
    dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    dice = dice.item()
    
    return {
        'iou': iou,
        'dice': dice
    }

# ------------------ 内存优化的训练函数 ------------------
def train_memory_optimized():
    """内存优化的训练函数"""
    
    # 内存优化配置
    image_dir = 'data/training_precise/images'
    mask_dir = 'data/training_precise/masks'
    max_size = 512  # 降低分辨率
    batch_size = 2  # 减少批次大小
    num_epochs = 120
    lr = 1e-4
    accumulation_steps = 2  # 梯度累积步数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"使用设备: {device}")
    log_memory_usage("训练开始前")
    
    # 获取数据路径
    image_files, mask_files = get_image_mask_pairs(image_dir, mask_dir)
    if len(image_files) == 0:
        logging.error('未找到任何有效的图片和掩码对')
        return
    
    logging.info(f"找到 {len(image_files)} 个图像-掩码对")
    
    # 训练验证集划分
    X_train, X_val, y_train, y_val = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    # 创建内存优化的数据集
    train_dataset = MemoryOptimizedNailDataset(X_train, y_train, max_size, is_train=True)
    val_dataset = MemoryOptimizedNailDataset(X_val, y_val, max_size, is_train=False)
    
    # 创建数据加载器（内存优化设置）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 减少worker数量
        pin_memory=False,  # 禁用pin_memory
        persistent_workers=False  # 禁用persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    logging.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
    logging.info(f"批次大小: {batch_size}, 有效批次大小: {batch_size * accumulation_steps}")
    
    # 创建模型
    model = U2NET(3, 1).to(device)
    
    # 加载预训练权重（如果存在）
    pretrained_path = 'models/u2net.pth'
    if os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            logging.info("成功加载预训练模型")
        except Exception as e:
            logging.warning(f"预训练模型加载失败: {e}")
    
    # 创建损失函数和优化器
    criterion = SimpleLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练监控
    best_val_iou = 0.0
    os.makedirs('models', exist_ok=True)
    save_path = 'models/u2net_nail_memory_optimized.pth'
    
    logging.info("开始内存优化训练")
    
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')
        for batch_idx, (imgs, masks) in enumerate(train_pbar):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            # 缩放损失以适应梯度累积
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 计算指标（简化版本）
            with torch.no_grad():
                pred = outputs[0]
                metrics = calculate_metrics_simple(pred, masks)
            
            train_loss += loss.item() * accumulation_steps
            train_iou += metrics['iou']
            train_dice += metrics['dice']
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'IoU': f'{metrics["iou"]:.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
            })
            
            # 定期清理内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 计算平均训练指标
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]')
            for imgs, masks in val_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                
                pred = outputs[0]
                metrics = calculate_metrics_simple(pred, masks)
                
                val_loss += loss.item()
                val_iou += metrics['iou']
                val_dice += metrics['dice']
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{metrics["iou"]:.4f}',
                    'Dice': f'{metrics["dice"]:.4f}'
                })
                
                # 定期清理内存
                torch.cuda.empty_cache()
        
        # 计算平均验证指标
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录日志和内存使用
        epoch_time = time.time() - start_time
        log_memory_usage(f"Epoch {epoch} 结束后")
        logging.info(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
        logging.info(f"  Train: Loss={train_loss:.4f}, IoU={train_iou:.4f}, Dice={train_dice:.4f}")
        logging.info(f"  Val:   Loss={val_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f}")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            logging.info(f"保存最佳模型 (IoU: {val_iou:.4f})")
        
        # 早停检查
        if epoch > 10 and val_iou == 0:
            logging.info("IoU持续为0，停止训练")
            break
        
        # 强制清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    logging.info("内存优化训练完成")

def get_image_mask_pairs(image_dir, mask_dir):
    """获取图像-掩码对"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = []
    mask_paths = []
    
    for img_name in image_files:
        base, _ = os.path.splitext(img_name)
        mask_name = f"{base}_mask.png"
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            logging.warning(f"掩码不存在: {mask_path}")
    
    return image_paths, mask_paths

if __name__ == '__main__':
    train_memory_optimized() 