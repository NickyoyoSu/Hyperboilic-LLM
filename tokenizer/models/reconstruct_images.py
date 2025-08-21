import torch
import torch.nn as nn
import argparse
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# 导入您的模型定义（确保路径正确）
from models.AE import StandardAutoencoder
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from hyplib.manifolds.lorentzian import Lorentz
from models.adaptive_curvature import AdaptiveCurvature

def main():
    parser = argparse.ArgumentParser(description="使用预训练自编码器重建图像")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint 文件路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./reconstructed', help='输出文件夹路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--image_size', type=int, default=256, help='图像大小（假设方形）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print(f"加载 checkpoint: {args.checkpoint} (从 epoch {checkpoint.get('epoch', '未知')} 保存，损失 {checkpoint.get('loss', '未知'):.4f})")

    # 从 checkpoint 中获取模型参数
    model_args = checkpoint['args']
    h_dim = model_args['h_dim']
    n_res_layers = model_args['n_res_layers']
    adaptive_c = model_args['adaptive_c']
    initial_c = model_args['initial_c']

    # 创建模型
    model = StandardAutoencoder(h_dim=h_dim, n_res_layers=n_res_layers, c=initial_c, adaptive_c=adaptive_c).to(args.device)

    # 加载权重
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    if adaptive_c:
        model.curvature.load_state_dict(checkpoint['curvature'])
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 假设输入范围 [-1, 1]
    ])

    # 加载输入图像
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(image_files)} 张图像")

    images = []
    for f in image_files:
        img_path = os.path.join(args.input_dir, f)
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(args.device)  # 添加批次维度
        images.append((f, img))

    # 分批重建
    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i+args.batch_size]
        batch_tensors = torch.cat([img for _, img in batch_images])

        with torch.no_grad():
            reconstructions, _ = model(batch_tensors)

        # 保存每个重建图像
        for j, (filename, orig_img) in enumerate(batch_images):
            recon = reconstructions[j].unsqueeze(0)  # 添加批次维度以便 make_grid

            # 创建对比图（原始 + 重建）
            comparison = torch.cat([orig_img, recon])
            grid = make_grid(comparison, nrow=2, normalize=True)  # 自动归一化

            output_path = os.path.join(args.output_dir, f"recon_{filename}")
            save_image(grid, output_path)
            print(f"保存重建图像至: {output_path}")

    print("重建完成！")

if __name__ == "__main__":
    main() 