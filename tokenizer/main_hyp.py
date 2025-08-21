import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
import os
import time
import random
import json
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from PIL import ImageFile
import torch.nn.functional as F
import kornia
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from models.vqvae_hyp import VQVAE_HYP as VQVAE 
from geoopt.optim import RiemannianAdam
from geoopt.manifolds import Lorentz as GeooptLorentz

# 从新的工具文件中导入所有辅助函数
from utils_hyp import (
    VGGPerceptualLoss,
    calc_loss_with_regularization,
    visualize_reconstructions,
    plot_cumulative_losses,
    analyze_embeddings,
    monitor_gradients,
    monitor_model_weights,
    plot_codebook_usage,
    plot_latent_space_distribution,
    plot_angular_distribution_at_radius
)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="增强版VQVAE训练")
    
    # 基本训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=128, help="有效批量大小")
    parser.add_argument("--n_updates", type=int, default=4000080)
    parser.add_argument("--learning_rate", type=float, default=3e-4) # 稍微调整学习率
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_interval", type=int, default=100) # 调整日志间隔
    parser.add_argument("--val_interval", type=int, default=2000, help="验证间隔")
    parser.add_argument("--dataset", type=str, default='IMAGENET')
    
    # 模型参数
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--n_embeddings", type=int, default=16384)
    parser.add_argument("--beta", type=float, default=1, help="承诺损失权重")
    
    # 正则化与稳定性参数
    parser.add_argument("--weight_decay", type=float, default=1e-5) # 减小权重衰减
    parser.add_argument("--grad_clip_value", type=float, default=3.0)
    
    # 学习率调度
    parser.add_argument("--use_lr_scheduler", action="store_true", default=True) # 默认开启
    parser.add_argument("--lr_scheduler", type=str, default="plateau", choices=["cosine", "plateau"])
    parser.add_argument("--min_lr", type=float, default=1e-6) # 调整最小学习率
    
    # 混合精度训练
    parser.add_argument("--fp16", action="store_true")
    
    # 恢复和保存
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./new_results")
    
    # 双曲特定参数
    parser.add_argument('--adaptive_c', action='store_true')
    parser.add_argument('--initial_c', type=float, default=1.0)
    
    # 从AE借鉴的损失和训练参数
    parser.add_argument('--pretrained_path', type=str, default=None)
    # 关键修复：为重建损失添加权重，并大幅提高感知和颜色损失的默认权重
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="重建损失的权重")
    parser.add_argument("--lambda_p", type=float, default=1.0, help="感知损失的权重 (从0.1上调)")
    parser.add_argument("--lambda_color", type=float, default=0.5, help="颜色损失的权重 (从0.1上调)")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="码本多样性损失的权重")
    parser.add_argument("--reg_warmup_steps", type=int, default=10000)
    parser.add_argument("--final_lambda_reg", type=float, default=0.005)
    parser.add_argument("--visualization_interval", type=int, default=2000)
    parser.add_argument("--monitor_interval", type=int, default=100)
    # 新增：码本正则化参数
    parser.add_argument("--lambda_cb_reg", type=float, default=1.0, help="码本几何正则化损失的权重")
    parser.add_argument("--target_cb_radius", type=float, default=5.0, help="码本目标平均半径")
    # 修复：添加缺失的命令行参数定义
    parser.add_argument("--quantizer_warmup_steps", type=int, default=5000, help="在切换到z_q之前，用z_e训练解码器的步数。")
    parser.add_argument("--reset_codes_interval", type=int, default=2500, help="每隔多少步重置一次僵尸码, 0表示不重置")
    
    return parser.parse_args()

def calc_codebook_regularization_loss(codebook_vectors, manifold, target_mean_radius, lambda_std=1.0, lambda_mean=1.0):
    """
    计算码本自身的几何正则化损失。
    鼓励码本向量在双曲空间中良好分布。
    """
    if codebook_vectors.numel() == 0:
        return torch.tensor(0.0, device=codebook_vectors.device), {}

    radii = manifold.dist0(codebook_vectors) # [n_embeddings]
    
    # 1. 半径标准差最大化 (鼓励散开), 我们最小化它的负值
    radius_std_reg = -lambda_std * torch.std(radii)

    # 2. 平均半径惩罚 (鼓励远离原点，朝向目标半径)
    mean_radius_reg = lambda_mean * F.mse_loss(radii.mean(), torch.tensor(target_mean_radius, device=radii.device))
    
    total_reg_loss = radius_std_reg + mean_radius_reg
    
    loss_dict = {
        'cb_radius_std': radius_std_reg.item(),
        'cb_mean_radius': mean_radius_reg.item()
    }
    return total_reg_loss, loss_dict

def train(args):
    set_seed(args.seed)

    gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16' if args.fp16 else 'no'
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"config_{utils.readable_timestamp()}.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
    
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"使用设备: {device}")
    
    # 加载数据
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)

    # 初始化模型 (不在此时 .to(device))
    model = VQVAE(
        h_dim=args.n_hiddens,
        n_res_layers=args.n_residual_layers,
        n_embeddings=args.n_embeddings,
        beta=args.beta,
        c=args.initial_c,
        adaptive_c=args.adaptive_c
    )
    # 将量化器预热步数传给模型（支持解码器绕过）
    if hasattr(model, 'quantizer_warmup_steps'):
        model.quantizer_warmup_steps = max(0, int(args.quantizer_warmup_steps))
    
    # 加载预训练权重
    if args.pretrained_path:
        if accelerator.is_main_process:
            print(f"加载预训练AE权重从: {args.pretrained_path}")
        # 所有进程都需要加载权重，但只有主进程打印信息
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.encoder.load_state_dict(checkpoint['encoder'], strict=False)
        model.decoder.load_state_dict(checkpoint['decoder'], strict=False)
        if args.adaptive_c and 'curvature' in checkpoint:
            if isinstance(checkpoint['curvature'], dict):
                model.curvature.load_state_dict(checkpoint['curvature'])
            else:
                c_val = checkpoint['curvature']
                model.manifold.c.data.fill_(c_val)
                model.encoder.manifold.c.data.fill_(c_val)
                model.decoder.manifold.c.data.fill_(c_val)
                model.vq.manifold.c.data.fill_(c_val)
        if accelerator.is_main_process:
            print("预训练权重加载完成！")

    optimizer = RiemannianAdam(
        model.parameters(), 
        lr=args.learning_rate, 
        stabilize=10,
        weight_decay=args.weight_decay,
    )
    
    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )
    
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=args.min_lr)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=args.n_updates, eta_min=args.min_lr)

    start_step = 0
    best_val_loss = float('inf')
    
    # 实例化损失函数和历史记录 (借鉴AE)
    perceptual_loss = VGGPerceptualLoss().to(device)
    history = {
        'total': [], 'recon': [], 'perceptual': [], 'color': [], 'reg': [], 'embed': [], 'perplexity': [], 'diversity': [], 'cb_reg': []
    }
    
    if args.resume and args.checkpoint:
        # 恢复逻辑（简化版）
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint.get('step', 0) + 1
        if accelerator.is_main_process:
            print(f"成功恢复到步骤 {start_step}")
    
    model.train()
    if accelerator.is_main_process:
        print(f"开始训练，共 {args.n_updates} 步...")
    
    train_iterator = iter(training_loader)
    
    for step in range(start_step, args.n_updates):
        try:
            x, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(training_loader)
            x, _ = next(train_iterator)
        
        # 定期清理缓存
        if step % 500 == 0:
            torch.cuda.empty_cache()
        
        x = x.to(device)
        
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # 决定是否在当前步骤执行重置
            reset_dead_codes_this_step = (
                args.reset_codes_interval > 0 and 
                step > 0 and 
                step % args.reset_codes_interval == 0
            )

            embedding_loss, x_hat, perplexity, diversity_loss, z_e_hyp, codebook_usage = model(
                x, 
                global_step=step, 
                reset_dead_codes=reset_dead_codes_this_step
            )
            
            # --- 使用工具文件中的函数计算所有损失 ---
            total_loss, loss_dict = calc_loss_with_regularization(
                x, x_hat, z_e_hyp, embedding_loss, diversity_loss, perceptual_loss, args, step
            )
            
            # 新增：计算并加入码本正则化损失
            unwrapped_model = accelerator.unwrap_model(model)
            # 一些量化器实现未必有 embedding 属性，先做健壮性判断
            cb_reg_loss = torch.tensor(0.0, device=device)
            cb_reg_loss_dict = {}
            vq = getattr(unwrapped_model, 'vq', None)
            embedding_weight = getattr(vq, 'embedding', None)
            embedding_weight = getattr(embedding_weight, 'weight', None) if embedding_weight is not None else None
            if embedding_weight is not None:
                cb_reg_loss, cb_reg_loss_dict = calc_codebook_regularization_loss(
                    embedding_weight,
                    unwrapped_model.manifold,
                    args.target_cb_radius
                )
            total_loss = total_loss + args.lambda_cb_reg * cb_reg_loss
            loss_dict.update(cb_reg_loss_dict) # 将码本损失项加入字典以便记录
            
            accelerator.backward(total_loss)
            
            if args.grad_clip_value > 0:
                # 修复：仅在梯度同步时（即累积周期的最后一步）执行梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            
            optimizer.step()

        # --- 日志、监控、可视化 (借鉴AE) ---
        if accelerator.is_main_process:
            # 使用loss_dict更新历史记录
            for key, value in loss_dict.items():
                if key in history:
                    history[key].append(value)
            history['perplexity'].append(perplexity.item())
            # history['cb_reg'] is now handled by the loop above

            if step > 0 and step % args.log_interval == 0:
                # 兼容预热阶段（此时 codebook_usage 可能是标量占位）
                if codebook_usage.dim() == 0:
                    usage_percent = 0.0
                else:
                    used_codes = torch.sum(codebook_usage > 1e-7).item()
                    total_codes = codebook_usage.numel()
                    usage_percent = (used_codes / total_codes) * 100

                log_str = f"\n[Step {step}/{args.n_updates}] | "
                for key, value in loss_dict.items():
                    log_str += f"{key.capitalize()}: {value:.4f} | "
                log_str += f"Perplexity: {perplexity.item():.2f} | "
                log_str += f"CB_Reg: {loss_dict.get('cb_radius_std', 0) + loss_dict.get('cb_mean_radius', 0):.4f} | "
                log_str += f"Usage: {usage_percent:.2f}% | "
                log_str += f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                print(log_str)
            
            if step > 0 and step % args.monitor_interval == 0:
                monitor_gradients(accelerator.unwrap_model(model), step)

            if step > 0 and step % args.visualization_interval == 0 or step == 100 or step == 200 or step == 300 or step == 400 or step == 500 or step == 600 or step == 700 or step == 800 or step == 900 or step == 1000:
                print("--- 生成可视化图 ---")
                unwrapped_model = accelerator.unwrap_model(model)
                plot_cumulative_losses(history, args.output_dir)
                visualize_reconstructions(unwrapped_model, validation_loader, device, step, args.output_dir)
                analyze_embeddings(unwrapped_model, x, step, args.output_dir)
                plot_codebook_usage(codebook_usage, step, args.output_dir)
                plot_latent_space_distribution(unwrapped_model, z_e_hyp, step, args.output_dir)
                plot_angular_distribution_at_radius(unwrapped_model, z_e_hyp, step, args.output_dir)

        # --- 增强的验证循环 ---
        if step > 0 and step % args.val_interval == 0:
            model.eval()
            val_losses = {'total': [], 'recon': [], 'embed': []}
            val_perplexities = []
            print("\n--- 开始验证 ---")
            with torch.no_grad():
                for i, (val_x, _) in enumerate(validation_loader):
                    if i >= 50: break # 限制验证样本数，加快速度
                    val_x = val_x.to(device)
                    val_embed_loss, val_x_hat, val_perplexity, _, _, _ = model(val_x)
                    val_recon_loss = F.mse_loss(val_x_hat, val_x)

                    # 验证时可以简化总损失，只关心核心部分
                    val_total_loss = val_recon_loss + val_embed_loss 
                    
                    val_losses['total'].append(val_total_loss.item())
                    val_losses['recon'].append(val_recon_loss.item())
                    val_losses['embed'].append(val_embed_loss.item())
                    val_perplexities.append(val_perplexity.item())
            
            avg_val_loss = np.mean(val_losses['total'])
            avg_perplexity = np.mean(val_perplexities)
            print(f"--- 验证完成 (Step {step}) ---")
            print(f"Avg Total Loss: {avg_val_loss:.6f} | "
                  f"Avg Recon: {np.mean(val_losses['recon']):.6f} | "
                  f"Avg Embed: {np.mean(val_losses['embed']):.6f} | "
                  f"Avg Perplexity: {avg_perplexity:.2f}")
            
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if args.save:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_path = os.path.join(args.output_dir, 'best_model.pth')
                    torch.save({'model': unwrapped_model.state_dict(), 'step': step, 'loss': best_val_loss}, save_path)
                    if accelerator.is_main_process:
                        print(f"*** 保存新的最佳模型至: {save_path} ***")
            model.train()
        
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        if args.save and step > 0 and step % args.save_interval == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f'checkpoint_{step}.pth')
            torch.save({'model': unwrapped_model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, save_path)
            if accelerator.is_main_process:
                print(f"--- 保存检查点至: {save_path} ---")
            
    print("训练完成！")

if __name__ == "__main__":
    args = parse_args()
    train(args)
