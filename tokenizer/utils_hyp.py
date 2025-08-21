import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from torchvision.utils import make_grid
import kornia
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        self.slice4 = vgg[16:23]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        def get_feats(x):
            h = self.slice1(x)
            h_relu1_2 = h
            h = self.slice2(h)
            h_relu2_2 = h
            h = self.slice3(h)
            h_relu3_3 = h
            h = self.slice4(h)
            h_relu4_3 = h
            return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        feats_input = get_feats(input)
        feats_target = get_feats(target)
        loss = 0.0
        for f_in, f_t in zip(feats_input, feats_target):
            loss = loss + F.l1_loss(f_in, f_t)
        return loss

def calc_loss_with_regularization(x, x_hat, z_e_hyp, embedding_loss, diversity_loss, perceptual_loss_fn, args, step):
    """
    Calculates the complete loss for the VQVAE model, combining all components.
    This function encapsulates the loss logic migrated from train_autoencoder.py,
    adding the VQVAE-specific embedding_loss.
    """
    # Reconstruction Loss
    recon_loss = F.mse_loss(x_hat, x)
    
    # Perceptual Loss
    p_loss = perceptual_loss_fn(x_hat, x)
    
    # Color Loss (on Saturation and Value)
    x_norm = (x + 1.0) / 2.0
    x_hat_norm = (x_hat + 1.0) / 2.0
    x_hsv = kornia.color.rgb_to_hsv(x_norm)
    x_hat_hsv = kornia.color.rgb_to_hsv(x_hat_norm)
    color_loss = F.l1_loss(x_hsv[:, 1:, :, :], x_hat_hsv[:, 1:, :, :])

    # Geometric Regularization Loss (with warm-up)
    current_lambda_reg = args.final_lambda_reg if step >= args.reg_warmup_steps else 0.0
    
    reg_loss = torch.tensor(0.0, device=x.device)
    if current_lambda_reg > 0:
        radius = torch.acosh(torch.clamp(z_e_hyp[..., 0], min=1.0+1e-7))
        radius_std_reg = -torch.sqrt(torch.var(radius) + 1e-8)
        mean_radius_reg = (radius.mean() - 5.0)**2
        reg_loss = current_lambda_reg * (0.5 * radius_std_reg + 0.3 * mean_radius_reg)

    # Total Loss - 关键修复：应用所有新的权重
    total_loss = (args.lambda_recon * recon_loss + 
                  args.lambda_p * p_loss + 
                  args.lambda_color * color_loss + 
                  embedding_loss + 
                  reg_loss +
                  args.lambda_diversity * diversity_loss)

    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'perceptual': p_loss.item(),
        'color': color_loss.item(),
        'embed': embedding_loss.item(),
        'reg': reg_loss.item(),
        'diversity': diversity_loss.item()
    }
    
    return total_loss, loss_dict

def visualize_reconstructions(model, loader, device, step, output_dir):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        
        # 修复：模型现在返回6个值，更新解包以匹配
        _, reconstructions, _, _, _, _ = model(images)

    # Denormalize for visualization
    images = (images + 1.0) / 2.0
    reconstructions = (reconstructions + 1.0) / 2.0

    comparison = torch.cat([images.cpu(), reconstructions.cpu()])
    grid = make_grid(comparison, nrow=8, normalize=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f"Step {step}: Original (top) vs Reconstructed (bottom)")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)
    plt.savefig(os.path.join(recon_dir, f"recon_step_{step}.png"), bbox_inches='tight')
    plt.close()
    model.train()

def plot_cumulative_losses(history, output_dir, smoothing_window=100, legend_window=1000):
    """
    绘制从开始到现在的累计损失曲线，并更新图例中的最新平均值。
    使用移动平均对曲线进行平滑处理。
    """
    plt.figure(figsize=(18, 10))
    
    # 定义要绘制的损失项和对应的颜色/标签
    loss_items = {
        'total': ('r', 'Total Loss'),
        'recon': ('b', 'Recon Loss'),
        'perceptual': ('g', 'Perceptual Loss'),
        'color': ('m', 'Color Loss'),
        'reg': ('c', 'Reg Loss'),
        'embed': ('y', 'Embed Loss'),
        'cb_reg': ('orange', 'CB Reg Loss')
    }

    custom_lines = []
    legend_texts = []

    for key, (color, label) in loss_items.items():
        if key in history and history[key]:
            data = np.array(history[key])
            
            # 使用移动平均平滑数据
            if len(data) >= smoothing_window:
                # 计算移动平均值
                smoothed_data = np.convolve(data, np.ones(smoothing_window)/smoothing_window, mode='valid')
                # x轴也需要相应调整，使其与平滑后的数据对齐
                x_steps = np.arange(smoothing_window - 1, len(data))
                plt.plot(x_steps, smoothed_data, color=color, linestyle='-', alpha=0.9, linewidth=1.5, label=label)
            else:
                # 如果数据点不够，直接绘制原始数据
                plt.plot(data, color=color, linestyle='-', alpha=0.7, linewidth=1.5, label=label)

            # 为图例计算最近N个批次的平均值
            if len(data) >= legend_window:
                avg_val = np.mean(data[-legend_window:])
                legend_text = f'{label} (last {legend_window} avg: {avg_val:.4f})'
            else:
                legend_text = label
            
            custom_lines.append(Line2D([0], [0], color=color, lw=2))
            legend_texts.append(legend_text)

    plt.title(f'Cumulative VQVAE Loss Curve (Smoothed over {smoothing_window} steps)')
    plt.xlabel('Step')
    plt.ylabel('Loss (Linear Scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(custom_lines, legend_texts, loc='upper right')
    
    save_path = os.path.join(output_dir, "latest_vqvae_loss.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def analyze_embeddings(model, x, step, output_dir):
    with torch.no_grad():
        # Correctly call the encode method which is now part of the model
        z_hyp = model._encode(x)
        radius = torch.acosh(torch.clamp(z_hyp[..., 0], min=1.0+1e-5))
        radius_flat = radius.reshape(-1).cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(radius_flat, bins=30)
        plt.title(f"Step {step} - hyp radius distribution")
        plt.xlabel("hyp radius")
        plt.ylabel("frequency")
        plt.axvline(np.mean(radius_flat), color='r', linestyle='--')
        stats_text = (f"mean: {np.mean(radius_flat):.4f}\n"
                     f"std: {np.std(radius_flat):.4f}\n"
                     f"min: {np.min(radius_flat):.4f}\n"
                     f"max: {np.max(radius_flat):.4f}")
        plt.annotate(stats_text, xy=(0.7, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
        plt.savefig(f"{output_dir}/embeddings/radius_dist_step_{step}.png")
        plt.close()

def plot_latent_space_distribution(model, z_e_hyp_batch, step, output_dir, n_samples=2048):
    """
    通过PCA降维来可视化编码器输出z_e_hyp的分布。
    生成两个子图：
    1. 3D散点图，按双曲半径着色。
    2. 极坐标图，显示半径和角度的分布。
    """
    if z_e_hyp_batch is None:
        return

    # --- 数据准备 ---
    model.eval()
    with torch.no_grad():
        # 将 [B, C+1, H, W] -> [B*H*W, C+1]
        z_flat = z_e_hyp_batch.permute(0, 2, 3, 1).reshape(-1, z_e_hyp_batch.shape[1])
        
        # 如果向量太多，进行随机下采样
        if z_flat.shape[0] > n_samples:
            indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_samples]
            z_sample = z_flat[indices]
        else:
            z_sample = z_flat
        
        # 提取空间部分和计算双曲半径
        spatial_vectors = z_sample[:, 1:]
        radii = model.manifold.dist0(z_sample).cpu().numpy()

    # --- PCA降维 ---
    # 3D PCA
    pca_3d = PCA(n_components=3)
    z_3d = pca_3d.fit_transform(spatial_vectors.cpu().numpy())
    
    # 2D PCA for polar plot
    pca_2d = PCA(n_components=2)
    z_2d = pca_2d.fit_transform(spatial_vectors.cpu().numpy())
    
    # 从2D PCA结果计算角度
    angles = np.arctan2(z_2d[:, 1], z_2d[:, 0])
    
    # --- 绘图 ---
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f'Latent Space Distribution at Step {step}', fontsize=16)

    # 1. 3D散点图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c=radii, cmap='viridis', s=10, alpha=0.7)
    ax1.set_title('3D PCA of Spatial Latent Vectors (Colored by Hyperbolic Radius)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')
    fig.colorbar(scatter1, ax=ax1, label='Hyperbolic Radius')
    ax1.grid(True)

    # 2. 极坐标图
    ax2 = fig.add_subplot(122, polar=True)
    scatter2 = ax2.scatter(angles, radii, c=radii, cmap='viridis', s=10, alpha=0.7)
    ax2.set_title('Polar Distribution (Radius vs. Angle from 2D PCA)')
    ax2.set_rlabel_position(-30)  # 移动径向标签以避免重叠
    ax2.grid(True)
    
    # --- 保存图像 ---
    save_dir = os.path.join(output_dir, "latent_space_dist")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'latent_dist_step_{step}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    model.train()


def plot_angular_distribution_at_radius(model, z_e_hyp_batch, step, output_dir, n_samples=1024, radius_window=0.1):
    """
    可视化在特定半径壳层内，潜在向量在空间角度上的分布。
    1. 找到最常见的半径。
    2. 筛选出该半径附近的一个壳层内的所有向量。
    3. 使用PCA将这些向量降到3D并绘制散点图。
    """
    if z_e_hyp_batch is None:
        return

    # --- 数据准备 ---
    model.eval()
    with torch.no_grad():
        z_flat = z_e_hyp_batch.permute(0, 2, 3, 1).reshape(-1, z_e_hyp_batch.shape[1])
        
        # 计算所有向量的半径
        all_radii = model.manifold.dist0(z_flat)
        
        # 如果没有有效的半径，则退出
        if all_radii.numel() == 0:
            print(f"[警告] 步骤 {step}: 无法计算半径用于角度分布图。")
            model.train()
            return
            
        # 确定目标半径 (使用均值)
        target_radius = all_radii.mean().item()
        
        # 筛选出在半径壳层内的向量
        mask = (all_radii >= target_radius - radius_window) & (all_radii <= target_radius + radius_window)
        
        filtered_vectors = z_flat[mask]
        
        if filtered_vectors.shape[0] < 10:
            print(f"[信息] 步骤 {step}: 在半径 {target_radius:.2f}±{radius_window} 范围内找到的向量不足 (<10)，跳过角度分布图。")
            model.train()
            return

        # 如果向量太多，进行随机下采样
        if filtered_vectors.shape[0] > n_samples:
            indices = torch.randperm(filtered_vectors.shape[0], device=filtered_vectors.device)[:n_samples]
            z_sample = filtered_vectors[indices]
        else:
            z_sample = filtered_vectors
        
        spatial_vectors = z_sample[:, 1:]

    # --- PCA降维 ---
    pca_3d = PCA(n_components=3)
    z_3d = pca_3d.fit_transform(spatial_vectors.cpu().numpy())
    
    # --- 绘图 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=15, alpha=0.8)
    
    ax.set_title(f'Angular Distribution at Radius ≈ {target_radius:.2f} (±{radius_window}), Step {step}\n{z_sample.shape[0]} samples shown')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.grid(True)
    
    # --- 保存图像 ---
    save_dir = os.path.join(output_dir, "angular_dist")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'angular_dist_step_{step}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    model.train()


def monitor_gradients(model, step):
    max_grad = 0
    max_param_name = ""
    nan_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_params.append(name)
                continue
            grad_abs_max = param.grad.abs().max().item()
            if grad_abs_max > max_grad:
                max_grad = grad_abs_max
                max_param_name = name
    if nan_params:
        print(f"[步骤 {step}] 发现NaN梯度: {nan_params}")
    print(f"[步骤 {step}] 最大梯度: {max_grad:.6f} ({max_param_name})")
    if max_grad > 100:
        print(f"[步骤 {step}] 警告: 梯度爆炸!")

def monitor_model_weights(model, step):
    print(f"\n===== 批次{step}权重状态检查 =====")
    problem_found = False
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            problem_found = True
            print(f"警告: 参数 {name} 包含NaN值!")
        elif torch.isinf(param.data).any():
            problem_found = True
            print(f"警告: 参数 {name} 包含Inf值!")
        elif param.data.abs().max() > 1e3:
            print(f"注意: 参数 {name} 包含极大值: {param.data.abs().max().item():.4e}")
    if not problem_found:
        print(f"批次{step}: 所有权重正常") 

def plot_codebook_usage(usage, step, output_dir):
    """
    可视化码本的使用频率。
    """
    if usage is None:
        return
    usage_tensor = usage.detach().cpu()
    # 防御性编程：检查传入的是否是标量
    if usage_tensor.dim() == 0:
        print(f"[警告] 步骤 {step}: plot_codebook_usage 接收到了一个标量，无法绘制码本使用图。请检查模型返回值。")
        return
    plt.figure(figsize=(15, 6))
    usage_np = usage_tensor.numpy()
    
    # 过滤掉使用频率非常低（接近0）的，以获得更清晰的视图
    used_indices = np.where(usage_np > 1e-7)[0]
    used_usage = usage_np[used_indices]
    
    plt.bar(range(len(used_usage)), used_usage, color='skyblue')
    
    total_codes = len(usage_np)
    used_codes_count = len(used_indices)
    usage_percentage = (used_codes_count / total_codes) * 100
    
    plt.title(f'Codebook Usage at Step {step}\n{used_codes_count}/{total_codes} codes used ({usage_percentage:.2f}%)')
    plt.xlabel('Used Codebook Index (Sorted by Usage)')
    plt.ylabel('Usage Frequency')
    plt.yscale('log') # 使用对数刻度可以更好地观察小的使用频率
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(output_dir, f'codebook_usage_step_{step}.png')
    plt.savefig(save_path)
    plt.close() 