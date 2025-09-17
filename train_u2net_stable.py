# -*- coding: utf-8 -*-
"""
train_u2net_stable.py - 超稳定指甲分割训练脚本

专门解决NaN和内存问题的训练脚本
"""
import os
import sys
import time
import logging
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path

# 添加U2Net路径
sys.path.append('u2net')

from model import U2NET
from data_loader import RescaleT, ToTensorLab, SalObjDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_stable.log'),
        logging.StreamHandler()
    ]
)

class SimpleNailDataset(Dataset):
    """简化的指甲分割数据集"""
    
    def __init__(self, image_paths, mask_paths, max_size=1024):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.max_size = max_size
        
        # 简单的数据增强
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整大小
        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 填充到正方形
        h, w = image.shape[:2]
        size = max(h, w)
        padded_image = np.zeros((size, size, 3), dtype=np.uint8)
        padded_mask = np.zeros((size, size), dtype=np.uint8)
        
        # 居中放置
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
        padded_mask[y_offset:y_offset+h, x_offset:x_offset+w] = mask
        
        # 转换为PIL图像
        image_pil = Image.fromarray(padded_image)
        mask_pil = Image.fromarray(padded_mask)
        
        # 应用变换
        image_tensor = self.transform(image_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # 添加通道维度
        
        return image_tensor, mask_tensor

class SimpleLoss(nn.Module):
    """简化的损失函数，只使用主输出"""
    
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, target):
        # 只使用主输出 (d0)
        main_output = outputs[0]
        
        # 确保输出和目标的形状匹配
        if main_output.shape != target.shape:
            target = nn.functional.interpolate(target, size=main_output.shape[2:], mode='nearest')
        
        # 计算BCE损失
        loss = self.bce_loss(main_output, target)
        
        return loss

def calculate_metrics(pred, target, threshold=0.5):
    """计算IoU和Dice系数"""
    # 应用sigmoid和阈值
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    
    # 确保形状匹配
    if pred_binary.shape != target.shape:
        target = nn.functional.interpolate(target, size=pred_binary.shape[2:], mode='nearest')
    
    # 计算IoU
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # 计算Dice
    dice = (2 * intersection) / (pred_binary.sum() + target.sum() + 1e-6)
    
    return iou.item(), dice.item()

def get_data_paths():
    """获取训练和验证数据路径"""
    train_image_dir = "data/training_precise/images"
    train_mask_dir = "data/training_precise/masks"
    
    # 获取所有图像和掩码文件
    image_files = sorted([f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(train_mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # 构建完整路径
    train_image_paths = [os.path.join(train_image_dir, f) for f in image_files]
    train_mask_paths = [os.path.join(train_mask_dir, f) for f in mask_files]
    
    # 分割训练和验证集 (80/20)
    split_idx = int(len(train_image_paths) * 0.8)
    
    train_images = train_image_paths[:split_idx]
    train_masks = train_mask_paths[:split_idx]
    val_images = train_image_paths[split_idx:]
    val_masks = train_mask_paths[split_idx:]
    
    return train_images, train_masks, val_images, val_masks

def train():
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 获取数据路径
    train_images, train_masks, val_images, val_masks = get_data_paths()
    logging.info(f"训练样本数: {len(train_images)}")
    logging.info(f"验证样本数: {len(val_images)}")
    
    # 创建数据集
    train_dataset = SimpleNailDataset(train_images, train_masks)
    val_dataset = SimpleNailDataset(val_images, val_masks)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # 小批量
        shuffle=True, 
        num_workers=0,  # 不使用多进程
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # 创建模型
    model = U2NET(3, 1)
    model = model.to(device)
    
    # 加载预训练权重（如果存在）
    pretrained_path = "models/u2net.pth"
    if os.path.exists(pretrained_path):
        logging.info(f"加载预训练权重: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        logging.info("从头开始训练")
    
    # 创建损失函数和优化器
    criterion = SimpleLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练参数
    num_epochs = 50
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                iou, dice = calculate_metrics(outputs[0], masks)
            
            train_loss += loss.item()
            train_iou += iou
            train_dice += dice
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
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
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                iou, dice = calculate_metrics(outputs[0], masks)
                
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        # 计算平均验证指标
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录日志
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"  Train: Loss={train_loss:.4f}, IoU={train_iou:.4f}, Dice={train_dice:.4f}")
        logging.info(f"  Val:   Loss={val_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f}")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model_stable.pth')
            logging.info("保存最佳模型")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            logging.info(f"早停触发，{patience}个epoch没有改善")
            break
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    logging.info("训练完成")

if __name__ == "__main__":
    train() 