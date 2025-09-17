import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import logging
import sys
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_generation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class REBNCONV(torch.nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

def _upsample_like(src,tar):
    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear',align_corners=False)
    return src

class RSU7(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)
        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU6(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU5(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU4(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU4F(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))
        return hx1d + hxin

class U2NET(torch.nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage2 = RSU6(64,32,128)
        self.pool23 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage3 = RSU5(128,64,256)
        self.pool34 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage4 = RSU4(256,128,512)
        self.pool45 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage5 = RSU4F(512,256,512)
        self.pool56 = torch.nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage6 = RSU4F(512,256,512)
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        self.side1 = torch.nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = torch.nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = torch.nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = torch.nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = torch.nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = torch.nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = torch.nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

class MaskGenerator:
    def __init__(self, model_path):
        """初始化掩码生成器"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            self.model = U2NET(3, 1)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            logging.info(f"模型加载成功: {model_path}")
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """预处理图像"""
        try:
            logging.info(f"开始预处理图像: {image_path}")
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            logging.info(f"原始图像大小: {original_size}")
            
            # 调整大小到320x320
            image = image.resize((320, 320), Image.Resampling.LANCZOS)
            logging.info("图像已调整大小到320x320")
            
            # 转换为numpy数组并归一化
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float()
            image = image.unsqueeze(0)
            image = (image / 255.0)  # 归一化到[0,1]范围
            logging.info("图像预处理完成")
            
            return image.to(self.device), original_size
            
        except Exception as e:
            logging.error(f"图像预处理失败: {str(e)}")
            raise

    def generate_mask(self, image):
        """生成掩码"""
        try:
            logging.info("开始生成掩码")
            with torch.no_grad():
                # 获取模型输出
                logging.info("运行模型推理")
                d0, d1, d2, d3, d4, d5, d6 = self.model(image)
                logging.info("模型推理完成")
                
                # 使用主输出d0
                pred = d0[:, 0, :, :]
                
                # 转换为numpy数组
                pred = pred.cpu().numpy()
                
                # 确保是单通道图像
                pred = pred.squeeze()  # 移除多余的维度
                
                # 转换为uint8格式
                pred = (pred * 255).astype(np.uint8)
                
                # 使用Otsu阈值处理
                _, mask = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                logging.info("掩码生成完成")
                
                return mask
                
        except Exception as e:
            logging.error(f"掩码生成失败: {str(e)}")
            raise

    def postprocess_mask(self, mask, original_size):
        """后处理掩码"""
        try:
            # 调整回原始大小
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # 形态学操作改善掩码质量
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logging.error(f"掩码后处理失败: {str(e)}")
            raise

    def process_directory(self, input_dir, output_dir):
        """处理目录中的所有图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"创建输出目录: {output_dir}")

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(image_files)
        logging.info(f"找到 {total_files} 个图像文件")

        success_count = 0
        for i, image_file in enumerate(tqdm(image_files, desc="处理图像")):
            try:
                logging.info(f"开始处理第 {i+1}/{total_files} 个图像: {image_file}")
                input_path = os.path.join(input_dir, image_file)
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_mask.png")
                
                # 预处理
                image, original_size = self.preprocess_image(input_path)
                
                # 生成掩码
                mask = self.generate_mask(image)
                
                # 后处理
                mask = self.postprocess_mask(mask, original_size)
                
                # 保存掩码
                cv2.imwrite(output_path, mask)
                success_count += 1
                logging.info(f"成功生成掩码: {output_path}")
                
            except Exception as e:
                logging.error(f"处理图像 {image_file} 时出错: {str(e)}")
                continue

        logging.info(f"处理完成。成功生成 {success_count}/{total_files} 个掩码")

def main():
    try:
        # 设置路径
        model_path = "models/u2net.pth"
        input_dir = "data/images"  # 修改为正确的输入目录
        output_dir = "data/masks"  # 修改为对应的输出目录
        
        # 创建生成器
        generator = MaskGenerator(model_path)
        
        # 处理目录
        generator.process_directory(input_dir, output_dir)
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 