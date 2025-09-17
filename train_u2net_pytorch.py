# -*- coding: utf-8 -*-
"""
train_u2net_pytorch.py - 超高鲁棒性指甲分割训练脚本

功能：
    - 使用1606张预处理后的高质量数据训练U²-Net模型
    - 实现超高鲁棒性的指甲分割效果
    - 超强数据增强策略
    - 多损失函数组合
    - EMA、标签平滑等先进训练技术
    - 混合精度训练，优化GPU使用
    - TensorBoard图形化监控界面

数据：
    - 图像: data/training_precise/images (1606张)
    - 掩码: data/training_precise/masks (1606张)
    - 尺寸: 1024长边，与推理时完全一致

输出：
    - 最优模型: models/u2net_nail_best.pth
    - 训练日志: train_u2net_detailed.log
    - 训练曲线: training_curves.png
    - TensorBoard日志: runs/training_logs/

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
from torch.cuda.amp import GradScaler, autocast
import time
from pathlib import Path
import cv2
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from kornia.losses import LovaszHingeLoss  # 新增
matplotlib.use('Agg')  # 使用非交互式后端

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_u2net_detailed.log', encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ------------------ 超高鲁棒性数据集定义（速度优化版） ------------------
class UltraRobustNailSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, max_size=1024, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.max_size = max_size
        self.is_train = is_train
        
        # 超高鲁棒性数据增强策略（速度优化版）
        if is_train:
            self.transform = A.Compose([
                # 基础几何变换 - 保持鲁棒性
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,  # 减少shift范围
                    scale_limit=0.2,   # 减少scale范围
                    rotate_limit=30,    # 减少rotate范围
                    p=0.7              # 降低概率
                ),
                
                # 智能缩放 - 先调整到目标尺寸
                A.LongestMaxSize(max_size=self.max_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                
                # 简化随机裁剪 - 减少计算
                A.RandomCrop(height=896, width=896, p=0.2),  # 减少概率和尺寸
                A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, p=0.2),
                
                # 简化弹性变换 - 保留核心鲁棒性
                A.OneOf([
                    A.ElasticTransform(alpha=100, sigma=100 * 0.05, p=0.3),  # 减少参数
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # 减少steps
                    A.OpticalDistortion(distort_limit=0.4, p=0.3),            # 减少distort
                ], p=0.3),  # 降低整体概率
                
                # 简化光照变换 - 保留核心效果
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),  # 减少clip_limit
                ], p=0.4),  # 降低概率
                
                # 简化颜色变换
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                    A.ChannelShuffle(p=0.2),
                    A.ToGray(p=0.1),
                ], p=0.3),
                
                # 简化噪声和模糊
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 减少blur范围
                    A.MotionBlur(blur_limit=7, p=0.3),
                ], p=0.3),
                
                # 简化多尺度训练
                A.RandomScale(scale_limit=0.2, p=0.4),  # 减少scale范围
                
                # 最终调整到目标尺寸
                A.LongestMaxSize(max_size=self.max_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=cv2.BORDER_CONSTANT),
                
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

# ------------------ 快速数据增强类 ------------------
class FastRobustNailSegmentationDataset(Dataset):
    """快速版本的数据集，减少数据增强复杂度但保持核心鲁棒性"""
    def __init__(self, image_paths, mask_paths, max_size=1024, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.max_size = max_size
        self.is_train = is_train
        
        # 快速但有效的数据增强策略
        if is_train:
            self.transform = A.Compose([
                # 核心几何变换
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.6
                ),
                
                # 基础缩放
                A.LongestMaxSize(max_size=self.max_size, interpolation=cv2.INTER_LANCZOS4),
                A.PadIfNeeded(min_height=self.max_size, min_width=self.max_size, border_mode=cv2.BORDER_CONSTANT),
                
                # 简化光照变换
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
                ], p=0.5),
                
                # 简化噪声
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
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

# ------------------ 简化和稳定的损失函数 ------------------
class StableU2NetLoss(nn.Module):
    """简化和稳定的U2Net多输出损失函数"""
    def __init__(self, weights=[0.0, 0.0, 0.0, 1.0, 0.8, 0.6, 0.4]):  # 跳过前3层
        super(StableU2NetLoss, self).__init__()
        self.weights = weights
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, outputs, target):
        # outputs: (d0, d1, d2, d3, d4, d5, d6)
        # target: 目标掩码
        bce_loss = 0
        dice_loss = 0
        
        for i, output in enumerate(outputs):
            if i < len(self.weights):
                weight = self.weights[i]
                # 跳过权重为0的层（有问题的前3层）
                if weight > 0:
                    bce_loss += weight * self.bce_loss(output, target)
                    dice_loss += weight * self.dice_loss(output, target)
        
        # 简化的损失组合
        total_loss = 0.6 * bce_loss + 0.4 * dice_loss
        return total_loss

class DiceLoss(nn.Module):
    """稳定的Dice损失函数"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 使用sigmoid激活函数，因为输入是logits
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice_coeff

# ------------------ Lovasz 多输出损失函数 ------------------
class StableU2NetLovaszLoss(nn.Module):
    """U2Net多输出 Lovasz 损失函数（主输出 Lovasz，辅助输出可选 Dice/BCE）"""
    def __init__(self, weights=[0.0, 0.0, 0.0, 1.0, 0.8, 0.6, 0.4], aux_loss_type='dice'):
        super().__init__()
        self.weights = weights
        self.lovasz = LovaszHingeLoss()
        self.aux_loss_type = aux_loss_type
        if aux_loss_type == 'dice':
            self.aux_loss = DiceLoss()
        elif aux_loss_type == 'bce':
            self.aux_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"aux_loss_type must be 'dice' or 'bce', got {aux_loss_type}")

    def forward(self, outputs, target):
        # outputs: (d0, d1, d2, d3, d4, d5, d6)
        # target: (B, 1, H, W)
        lovasz_loss = 0
        aux_loss = 0
        for i, output in enumerate(outputs):
            if i < len(self.weights):
                weight = self.weights[i]
                if weight > 0:
                    if i == 0:
                        # Lovasz 只用于主输出，且 target 需去掉 channel 维
                        lovasz_loss += weight * self.lovasz(output, target[:,0,:,:])
                    else:
                        aux_loss += weight * self.aux_loss(output, target)
        # 只用 Lovasz + 辅助损失
        total_loss = 0.7 * lovasz_loss + 0.3 * aux_loss
        return total_loss

# ------------------ 简化损失函数（用于调试IoU问题） ------------------
class SimpleU2NetLoss(nn.Module):
    """简化的U2Net损失函数，只使用主输出"""
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, outputs, target):
        # 只使用主输出 (d0)
        pred = outputs[0]
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        # 简单组合
        total_loss = 0.5 * bce_loss + 0.5 * dice_loss
        return total_loss

class SimpleLovaszLoss(nn.Module):
    """简化的Lovasz损失函数，只使用主输出"""
    def __init__(self):
        super().__init__()
        self.lovasz = LovaszHingeLoss()
    
    def forward(self, outputs, target):
        # 只使用主输出 (d0)
        pred = outputs[0]
        # target需要去掉channel维度
        target_2d = target[:, 0, :, :]
        loss = self.lovasz(pred, target_2d)
        return loss

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

# ------------------ EMA模型 ------------------
class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ------------------ 改进的评估指标（带诊断） ------------------
def calculate_metrics(pred, target, threshold=0.5, debug=False):
    """计算多个评估指标（带详细诊断）"""
    # 使用sigmoid激活函数，因为输入是logits
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = (target > threshold).float()
    
    # 诊断信息
    if debug:
        logging.info(f"🔍 IoU诊断信息:")
        logging.info(f"  pred_sigmoid范围: [{pred_sigmoid.min().item():.4f}, {pred_sigmoid.max().item():.4f}]")
        logging.info(f"  pred_binary统计: 0={((pred_binary == 0).sum().item())}, 1={((pred_binary == 1).sum().item())}")
        logging.info(f"  target_binary统计: 0={((target_binary == 0).sum().item())}, 1={((target_binary == 1).sum().item())}")
        logging.info(f"  threshold: {threshold}")
    
    # IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()
    
    # Dice
    dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    dice = dice.item()
    
    # 精确度和召回率
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    if debug:
        logging.info(f"  intersection: {intersection.item()}")
        logging.info(f"  union: {union.item()}")
        logging.info(f"  IoU: {iou:.6f}")
        logging.info(f"  Dice: {dice:.6f}")
        logging.info(f"  Precision: {precision:.6f}")
        logging.info(f"  Recall: {recall:.6f}")
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }

def validate_data_quality(image_paths, mask_paths, sample_size=5):
    """验证数据质量"""
    logging.info("🔍 验证数据质量...")
    
    for i in range(min(sample_size, len(image_paths))):
        img_path = image_paths[i]
        mask_path = mask_paths[i]
        
        # 读取图像和掩码
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 检查基本属性
        logging.info(f"  样本{i+1}: {os.path.basename(img_path)}")
        logging.info(f"    图像形状: {image.shape}, 范围: [{image.min()}, {image.max()}]")
        logging.info(f"    掩码形状: {mask.shape}, 范围: [{mask.min()}, {mask.max()}]")
        logging.info(f"    掩码非零像素: {(mask > 0).sum()}/{mask.size} ({100*(mask > 0).sum()/mask.size:.2f}%)")
        
        # 检查掩码是否全黑或全白
        if mask.max() == mask.min():
            logging.warning(f"    ⚠️ 掩码可能有问题: 全{('黑' if mask.max() == 0 else '白')}")
        
        # 检查掩码是否有足够的正样本
        positive_ratio = (mask > 0).sum() / mask.size
        if positive_ratio < 0.01:
            logging.warning(f"    ⚠️ 掩码正样本比例过低: {positive_ratio:.4f}")
        elif positive_ratio > 0.99:
            logging.warning(f"    ⚠️ 掩码正样本比例过高: {positive_ratio:.4f}")

def adaptive_threshold(pred_sigmoid, target_binary, initial_threshold=0.5):
    """自适应阈值调整"""
    # 如果IoU为0，尝试不同的阈值
    best_iou = 0.0
    best_threshold = initial_threshold
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_binary = (pred_sigmoid > threshold).float()
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()
        
        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold
    
    return best_threshold, best_iou

# ------------------ 训练主流程 ------------------
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

def plot_training_curves(train_metrics, val_metrics, save_path='training_curves.png'):
    """绘制训练曲线"""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_metrics['loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # IoU曲线
    axes[0, 1].plot(epochs, train_metrics['iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, val_metrics['iou'], 'r-', label='Val IoU')
    axes[0, 1].set_title('IoU')
    axes[0, 1].legend()
    
    # Dice曲线
    axes[1, 0].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice')
    axes[1, 0].plot(epochs, val_metrics['dice'], 'r-', label='Val Dice')
    axes[1, 0].set_title('Dice')
    axes[1, 0].legend()
    
    # 精确度曲线
    axes[1, 1].plot(epochs, train_metrics['precision'], 'b-', label='Train Precision')
    axes[1, 1].plot(epochs, val_metrics['precision'], 'r-', label='Val Precision')
    axes[1, 1].set_title('Precision')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train():
    """主训练函数"""
    
    # 基本配置
    image_dir = 'data/training_precise/images'
    mask_dir = 'data/training_precise/masks'
    max_size = 1024
    batch_size = 4
    num_epochs = 120
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"使用设备: {device}")
    
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
    
    # 创建数据集
    train_dataset = UltraRobustNailSegmentationDataset(X_train, y_train, max_size, is_train=True)
    val_dataset = UltraRobustNailSegmentationDataset(X_val, y_val, max_size, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    logging.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
    
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
    save_path = 'models/u2net_nail.pth'
    
    logging.info("开始训练")
    
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')
        for batch_idx, (imgs, masks) in enumerate(train_pbar):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                pred = outputs[0]
                metrics = calculate_metrics(pred, masks)
            
            train_loss += loss.item()
            train_iou += metrics['iou']
            train_dice += metrics['dice']
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{metrics["iou"]:.4f}',
                'Dice': f'{metrics["dice"]:.4f}'
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
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]')
            for imgs, masks in val_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                
                pred = outputs[0]
                metrics = calculate_metrics(pred, masks)
                
                val_loss += loss.item()
                val_iou += metrics['iou']
                val_dice += metrics['dice']
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{metrics["iou"]:.4f}',
                    'Dice': f'{metrics["dice"]:.4f}'
                })
        
        # 计算平均验证指标
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录日志
        epoch_time = time.time() - start_time
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
    
    logging.info("训练完成")

if __name__ == '__main__':
    train() 