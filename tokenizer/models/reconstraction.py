import torch
import torch.nn as nn
import argparse
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# 导入您项目的模型定义
# 确保这个脚本能找到 'models' 目录
from models.AE import StandardAutoencoder
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from hyplib.manifolds.lorentzian import Lorentz
from models.adaptive_curvature import AdaptiveCurvature
import utils # 假设 utils.py 也在路径中

def load_model_from_checkpoint(checkpoint_path, device):
    """从checkpoint加载模型"""
    print(f"正在从 '{checkpoint_path}' 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从checkpoint中获取模型参数
    args = checkpoint['args']
    
    print("根据checkpoint中的参数重新创建模型...")
    model = StandardAutoencoder(
        h_dim=args['h_dim'],
        n_res_layers=args['n_res_layers'],
        c=args['initial_c'],  # 使用checkpoint中的初始c值
        adaptive_c=args['adaptive_c']
    ).to(device)
    
    # 加载编码器和解码器的状态字典
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    
    # 如果有自适应曲率，也加载其状态（手动忽略 c_history）
    if args['adaptive_c'] and 'curvature' in checkpoint:
        # 先复制 state_dict，然后删除问题键
        curvature_state = checkpoint['curvature'].copy()  # 复制以避免修改原 checkpoint
        if 'c_history' in curvature_state:
            del curvature_state['c_history']  # 删除尺寸不匹配的键
            print("已手动删除 checkpoint 中的 'c_history' 键（推理时不需要）")
        
        # 现在加载（使用 strict=False 以防其他小问题）
        model.curvature.load_state_dict(curvature_state, strict=False)
        print("自适应曲率加载成功（忽略了 c_history 尺寸不匹配）")
    else:
        print("未找到自适应曲率状态或已禁用，使用默认初始化")
    
    model.eval() # 设置为评估模式
    print("模型加载成功并已设置为评估模式。")
    return model

def reconstruct_images(args):
    """主函数，执行重建任务"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 1. 加载模型
    try:
        model = load_model_from_checkpoint(args.checkpoint, device)
    except FileNotFoundError:
        print(f"错误: Checkpoint文件未找到于 '{args.checkpoint}'")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # 2. 准备数据转换
    # 这个转换流程需要与您训练时使用的数据加载器完全一致
    # 假设您的训练数据加载器将图像转换为[-1, 1]范围
    img_size = 256 # 假设您的图像大小，根据需要修改
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 转换到 [-1, 1]
    ])

    # 3. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"重建结果将保存在: '{args.output_dir}'")

    # 4. 遍历输入文件夹中的所有图像
    valid_images = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_images:
        print(f"错误: 在 '{args.input_dir}' 中未找到任何有效的图像文件。")
        return
        
    for i, image_name in enumerate(valid_images):
        image_path = os.path.join(args.input_dir, image_name)
        print(f"正在处理 ({i+1}/{len(valid_images)}): {image_name}")
        
        try:
            # 打开图像并应用转换
            image_pil = Image.open(image_path).convert("RGB")
            image_tensor = data_transform(image_pil).unsqueeze(0).to(device)
            
            # 使用模型进行重建
            with torch.no_grad():
                reconstructed_tensor, _ = model(image_tensor)
            
            # 将原始图像和重建图像合并
            # 我们需要将原始图像张量也移动到CPU
            comparison = torch.cat([image_tensor.cpu(), reconstructed_tensor.cpu()], dim=0)
            
            # 保存对比图像
            # normalize=True 会自动将[-1, 1]的范围映射到[0, 1]以便保存
            output_path = os.path.join(args.output_dir, f"recon_{image_name}")
            save_image(comparison, output_path, nrow=2, normalize=True)

        except Exception as e:
            print(f"  处理文件 '{image_name}' 时失败: {e}")

    print("\n所有图像处理完毕！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的自编码器模型进行图像重建")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="训练好的模型checkpoint文件路径 (.pth)")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="包含输入图像的文件夹路径")
    parser.add_argument("-o", "--output_dir", type=str, default="./reconstructions_output", help="保存重建结果的输出文件夹路径")
    
    args = parser.parse_args()
    reconstruct_images(args)
