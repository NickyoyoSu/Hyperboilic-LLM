from PIL import Image
import os
import random
from torchvision.datasets import ImageFolder

class RobustImageFolder(ImageFolder):
    def __getitem__(self, index):
        """增强版__getitem__，可以处理损坏的图像"""
        for attempt in range(5):  # 最多尝试5次
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"警告: 读取索引 {index} 的图像失败: {str(e)}")
                if attempt < 4:  # 不是最后一次尝试
                    index = random.randint(0, len(self) - 1)  # 随机选择另一张图片
                else:  # 最后尝试失败，返回黑色图像
                    if self.transform is not None:
                        # 创建一个全黑的PIL图像，并应用变换
                        from PIL import Image
                        import numpy as np
                        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                        return self.transform(dummy), 0
                    else:
                        # 如果没有变换，返回张量
                        import torch
                        return torch.zeros(3, 224, 224), 0

def check_imagenet_images(data_dir):
    """
    遍历 ImageNet 数据集，检查图片是否损坏
    参数:
        data_dir: 数据集根目录
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    def check_images_in_dir(directory):
        print(f"检查目录: {directory}")
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return
        
        total_files = 0
        corrupted_files = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    total_files += 1
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            # 不仅验证，还要尝试转换 - 这与实际加载相同
                            img.load()
                            img.convert("RGB")
                    except Exception as e:
                        corrupted_files += 1
                        print(f"损坏的文件: {file_path}, 错误: {str(e)}")
        
        print(f"总文件数: {total_files}, 损坏文件数: {corrupted_files}")
    
    # 检查训练集
    check_images_in_dir(train_dir)
    
    # 检查验证集
    check_images_in_dir(val_dir)

# 调用检查函数
data_dir = "/ext/imagenet"  # 替换为你的 ImageNet 数据集路径
check_imagenet_images(data_dir)