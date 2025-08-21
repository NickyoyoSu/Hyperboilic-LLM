import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import utils
import torchvision.models as models
from models.AE import StandardAutoencoder
from geoopt.optim import RiemannianSGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 引入Accelerate库
from accelerate import Accelerator
from accelerate.utils import set_seed
from matplotlib.lines import Line2D
import kornia

# 在所有 import 之后，但在第一个函数定义之前，添加 VGGPerceptualLoss 类
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # 冻结VGG网络的所有参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 我们从VGG的这些层提取特征
        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:16]  # relu3_3
        self.slice4 = vgg[16:23] # relu4_3
        
        # 为ImageNet预训练模型定义的标准化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        # 标准化输入和目标图像以匹配VGG的期望输入
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        # 提取特征
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
            loss = loss + F.l1_loss(f_in, f_t) # 使用L1损失计算特征差异
            
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="超曲面自编码器预训练(Accelerate版)")
    
    # 基本训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--effective_batch_size", type=int, default=256, 
                        help="通过梯度累积实现的有效批量大小")
    parser.add_argument("--epochs", type=int, default=30)  
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='IMAGENET')
    
    # 模型参数
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--n_res_layers", type=int, default=2)
    parser.add_argument("--lambda_p", type=float, default=0.1, help="感知损失的权重")
    parser.add_argument("--reg_warmup_batches", type=int, default=10000, help="几何正则化预热的批次数")
    
    # 其他参数
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--adaptive_c", action="store_true", help="使用自适应曲率")
    parser.add_argument("--initial_c", type=float, default=1.0, help="初始曲率值")
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./pretrained_autoencoder")
    parser.add_argument("--log_with", type=str, default=None, 
                        choices=["tensorboard", "wandb", None], help="使用什么工具记录日志")
    
    return parser.parse_args()

def save_tensor_as_images(tensor, prefix):
    """将批次图像保存为PNG文件，包含自动归一化"""
    os.makedirs("debug_images", exist_ok=True)
    
    # 打印范围信息以帮助调试
    print(f"图像张量 {prefix} 范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    for i in range(min(len(tensor), 8)):  # 最多保存8张
        img = tensor[i].detach().cpu().permute(1, 2, 0).numpy()
        
        # 添加自动归一化
        if img.min() < -0.01 or img.max() > 1.01:
            print(f"  归一化图像 {i}, 原始范围: [{img.min():.4f}, {img.max():.4f}]")
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img = np.clip(img, 0.0, 1.0)  # 确保范围在[0,1]
        img = (img * 255).astype('uint8')
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"{prefix} 样本 {i}")
        plt.axis('off')
        plt.savefig(f"debug_images/{prefix}_{i}.png")
        plt.close()

def monitor_model_weights(model, batch_count):
    """监控模型权重，检测NaN和极值"""
    if 2890 <= batch_count <= 2910:
        print(f"\n===== 批次{batch_count}权重状态检查 =====")
        problem_found = False
        problem_layers = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                problem_found = True
                problem_layers.append(name)
                print(f"警告: 参数 {name} 包含NaN值!")
            elif torch.isinf(param.data).any():
                problem_found = True
                problem_layers.append(name)
                print(f"警告: 参数 {name} 包含Inf值!")
            elif param.data.abs().max() > 1e3:
                print(f"注意: 参数 {name} 包含极大值: {param.data.abs().max().item():.4e}")
        
        if problem_found:
            print(f"发现问题层: {problem_layers}")
            return False
        else:
            print(f"批次{batch_count}: 所有权重正常")
            return True


def monitor_gradients(model, batch_count):
    """监控梯度值，检测梯度爆炸"""
    if 2890 <= batch_count <= 2910:
        print(f"\n===== 批次{batch_count}梯度检查 =====")
        max_grad_norm = 0
        problem_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                if torch.isnan(param.grad).any():
                    problem_grads.append((name, "NaN"))
                    print(f"警告: {name} 梯度包含NaN!")
                elif torch.isinf(param.grad).any():
                    problem_grads.append((name, "Inf"))
                    print(f"警告: {name} 梯度包含Inf!")
                elif grad_norm > 100:
                    problem_grads.append((name, f"{grad_norm:.2e}"))
                    print(f"警告: {name} 梯度范数过大: {grad_norm:.2e}")
        
        print(f"最大梯度范数: {max_grad_norm:.4e}")
        return problem_grads
def visualize_reconstructions(model, validation_loader, epoch, output_dir):
    model.eval()
    with torch.no_grad():
        # 获取验证集批次
        batch = next(iter(validation_loader))
        
        # 修复：通过模型参数获取设备，而不是直接访问 model.device
        device = next(model.parameters()).device
        images = batch[0][:8].to(device)  # 取前8张图片，并确保在正确的设备上
        
        # 获取重建结果
        reconstructions, _ = model(images)
        
        # 打印重建张量的范围以供调试
        print(f"调试 [Epoch {epoch}]: 重建张量范围 [{reconstructions.min().item():.4f}, {reconstructions.max().item():.4f}]")

        # 将原始图像和重建图像合并，然后统一进行归一化显示
        # normalize=True 会自动将 [-1, 1] 的范围安全地映射到 [0, 1] 以便显示
        comparison = torch.cat([images.cpu(), reconstructions.cpu()])
        grid = make_grid(comparison, nrow=8, normalize=True)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Epoch {epoch}: Original (top) vs Reconstructed (bottom)")
        
        # 保存图像
        recon_dir = os.path.join(output_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        plt.savefig(os.path.join(recon_dir, f"recon_epoch_{epoch}.png"), bbox_inches='tight')
        plt.close()
    
    model.train()

# 在visualize_reconstructions函数下方添加以下函数

def plot_cumulative_losses(total_losses, recon_losses, perceptual_losses, color_losses, output_dir, window_size=2000):
    """
    绘制从开始到现在的累计损失曲线，并更新图例中的最新平均值。
    该函数会覆盖旧的图像，实现单张图持续更新。
    """
    plt.figure(figsize=(15, 8))
    
    # 绘制四种损失曲线
    plt.plot(total_losses, 'r-', alpha=0.7, linewidth=1.5, label='Total Loss')
    plt.plot(recon_losses, 'b-', alpha=0.6, linewidth=1.0, label='Recon Loss')
    plt.plot(perceptual_losses, 'g-', alpha=0.6, linewidth=1.0, label='Perceptual Loss')
    plt.plot(color_losses, 'm-', alpha=0.6, linewidth=1.0, label='Color Loss')
    
    # 为图例计算最近N个批次的平均值
    if len(recon_losses) >= window_size:
        avg_total = np.mean(total_losses[-window_size:])
        avg_recon = np.mean(recon_losses[-window_size:])
        avg_perceptual = np.mean(perceptual_losses[-window_size:])
        avg_color = np.mean(color_losses[-window_size:])
        legend_text_total = f'Total Loss (last {window_size} avg: {avg_total:.4f})'
        legend_text_recon = f'Recon Loss (last {window_size} avg: {avg_recon:.4f})'
        legend_text_perceptual = f'Perceptual Loss (last {window_size} avg: {avg_perceptual:.4f})'
        legend_text_color = f'Color Loss (last {window_size} avg: {avg_color:.4f})'
    else:
        # 如果数据不足，则不显示平均值
        legend_text_total = 'Total Loss'
        legend_text_recon = 'Recon Loss'
        legend_text_perceptual = 'Perceptual Loss'
        legend_text_color = 'Color Loss'

    # 创建自定义图例
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='m', lw=2)]
    
    plt.title('Cumulative Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    # 使用自定义图例和文本
    plt.legend(custom_lines, [legend_text_total, legend_text_recon, legend_text_perceptual, legend_text_color], loc='upper right')
    # plt.yscale('log')  # <-- 移除对数刻度
    
    # 保存图像，覆盖上一次的文件
    save_path = os.path.join(output_dir, "latest_loss.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150) # 使用较低的DPI以加快保存速度
    plt.close()


def calc_loss_with_regularization(x, x_hat, z_hyp, manifold, x_train_var, perceptual_loss_fn, lambda_reg=5.0, lambda_p=0.1):
    """多目标几何正则化损失"""
    # 重建损失
    recon_loss = F.mse_loss(x_hat, x)
    
    # 感知损失
    p_loss = perceptual_loss_fn(x_hat, x)

    x_norm = (x + 1.0)/2.0
    x_hat_norm = (x_hat + 1.0)/2.0
    
    x_hsv = kornia.color.rgb_to_hsv(x_norm)
    x_hat_hsv = kornia.color.rgb_to_hsv(x_hat_norm)
    
    hue_loss = F.l1_loss(x_hsv[:,0,:,:], x_hat_hsv[:,0,:,:])
    sat_loss = F.l1_loss(x_hsv[:,1,:,:], x_hat_hsv[:,1,:,:])
    color_loss = hue_loss + sat_loss
    
    # 提取半径信息
    radius = torch.acosh(torch.clamp(z_hyp[..., 0], min=1.0+1e-7))
    
    # 1. 鼓励半径分散: 惩罚小的标准差。目标是让std变大。
    # 我们最小化 -std，相当于最大化std。
    radius_std_reg = -torch.sqrt(torch.var(radius) + 1e-8)
    
    # 2. 鼓励平均半径在目标附近: 惩罚与目标的偏差
    mean_target = 5.0
    mean_radius_reg = (radius.mean() - mean_target)**2

    # 3. 鼓励分布均匀 (可选，但比之前更合理的实现)
    # 惩罚KS统计量，值越大代表与均匀分布差异越大
    sorted_radius, _ = torch.sort(radius.reshape(-1))
    target_cdf = (sorted_radius - sorted_radius.min()) / (sorted_radius.max() - sorted_radius.min() + 1e-8)
    uniform_cdf = torch.linspace(0, 1, len(sorted_radius), device=z_hyp.device)
    uniformity_reg = torch.max(torch.abs(target_cdf - uniform_cdf))

    # 综合正则化项 (注意是正号，作为惩罚)
    # 权重可以调整，这里给一个示例
    reg_loss = lambda_reg * (0.5 * radius_std_reg + 0.3 * mean_radius_reg + 0.2 * uniformity_reg)
    
    lambda_color = 0.1
    total_loss = recon_loss + lambda_p * p_loss + reg_loss + lambda_color * color_loss
    
    # 返回正确的值 (添加 color_loss_val)
    return total_loss, recon_loss, p_loss, recon_loss.item(), reg_loss.item(), p_loss.item(), color_loss.item()
    
def analyze_embeddings(z_hyp, epoch, batch_idx, output_dir):
    """分析双曲空间嵌入分布"""
    with torch.no_grad():
        # 提取半径并确保是一维的
        radius = torch.acosh(torch.clamp(z_hyp[:, 0], min=1.0+1e-5))
        
        # 关键修复：无论输入维度如何，都将tensor展平为1维
        radius_flat = radius.reshape(-1).cpu().numpy()
        
        # 创建直方图
        plt.figure(figsize=(10, 6))
        plt.hist(radius_flat, bins=30)  # 现在使用展平的数组
        plt.title(f"Epoch {epoch}, Batch {batch_idx} - hyp radius distribution")
        plt.xlabel("hyp radius")
        plt.ylabel("frequency")
        
        # 添加统计信息 (使用numpy函数而非tensor方法)
        plt.axvline(np.mean(radius_flat), color='r', linestyle='--')
        stats_text = (f"mean: {np.mean(radius_flat):.4f}\n"
                     f"std: {np.std(radius_flat):.4f}\n"
                     f"min: {np.min(radius_flat):.4f}\n"
                     f"max: {np.max(radius_flat):.4f}")
        plt.annotate(stats_text, xy=(0.7, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
        plt.savefig(f"{output_dir}/embeddings/radius_dist_e{epoch}_b{batch_idx}.png")
        plt.close()

def train_autoencoder():
    args = parse_args()
    train_losses = []  # 每个epoch的平均损失
    val_losses = []    # 每个epoch的验证损失
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 计算梯度累积步数
    gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    print(f"梯度累积步数: {gradient_accumulation_steps} (有效批量大小: {args.effective_batch_size})")
    
    # 初始化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=args.log_with
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 使用accelerator.print替代print，以便在分布式环境中只打印一次
    accelerator.print(f"使用设备: {accelerator.device}")
    
    # 加载数据
    accelerator.print("加载数据集...")
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    accelerator.print(f"数据集大小: {len(training_data)} 训练样本, {len(validation_data)} 验证样本")
    
    # 创建模型
    accelerator.print("创建自编码器模型...")
    model = StandardAutoencoder(
        h_dim=args.h_dim,
        n_res_layers=args.n_res_layers,
        c=args.initial_c,
        adaptive_c=args.adaptive_c

    )
    nan_grad_detected = False
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            if 2880 <= batch_count <= 2910:
                print(f"清零参数 {name} 的NaN/Inf梯度")
            param.grad.zero_()
            nan_grad_detected = True

    # 如果检测到大量NaN梯度，重置优化器动量状态
    if nan_grad_detected and 2880 <= batch_count <= 2910:
        print("检测到NaN梯度，重置优化器动量状态...")
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].zero_()
    # 优化器
    optimizer = RiemannianSGD(
        model.parameters(),
        lr=args.learning_rate,
        stabilize=1000,
        weight_decay=args.weight_decay,
        momentum=0.95
    )
    
    # 学习率调度
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,  # 每次减半
        patience=1,   # 1个epoch没改善就降低
        verbose=True,
        min_lr=1e-6
    )
    
    # 将模型、优化器、数据加载器等交给accelerator管理
    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )
    
    # 跟踪最佳模型
    best_loss = float('inf')
    
    # 训练循环
    batches_per_epoch = len(training_data) // args.batch_size
    visualization_interval = max(1, batches_per_epoch // 100)  # 每1/100个epoch可视化一次
    
    accelerator.print(f"每个epoch约有 {batches_per_epoch} 个批次")
    accelerator.print(f"每 {visualization_interval} 个批次进行一次可视化 (约每1/10个epoch)")
    
    # 训练循环
    accelerator.print(f"开始训练，共 {args.epochs} 个 epochs...")
    
    # 设置日志追踪
    if accelerator.is_main_process and args.log_with is not None:
        accelerator.init_trackers("autoencoder_training")
    
    # 在epoch循环开始前，为特殊检查点定义追踪变量
    if accelerator.is_main_process:
        best_loss_in_first_fifth = float('inf')
        # --- 恢复历史记录 ---
        history_total_losses = []
        history_recon_losses = []
        history_perceptual_losses = []
        history_color_losses = []
        
    # --- 恢复实例化VGG损失 ---
    perceptual_loss = VGGPerceptualLoss().to(accelerator.device)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 为动态 lambda_reg 定义最终目标值
        final_lambda_reg = 0.005 # 这是一个比5.0更温和、更合理的值

        # 为特殊检查点计算目标批次数
        if epoch == 0 and accelerator.is_main_process:
            target_batch_for_checkpoint = batches_per_epoch // 5
            accelerator.print(f"--- [Epoch 0] 特别监控启动：将在前 {target_batch_for_checkpoint} 个批次中寻找最佳模型。 ---")
        
        # 用于记录当前epoch的批次损失
        current_epoch_batch_losses = []
        # 用于每2000个batch绘制一次的临时列表
        interval_batch_losses = []
        last_plot_batch = 0
        
        # 添加try-except块来处理损坏图像
        for batch_idx, batch_data in enumerate(training_loader):
            # 删除try-except块，让错误直接暴露出来
            current_batch = batch_count
            

            with accelerator.accumulate(model):
                if batch_count % 500 == 0:
                    torch.cuda.empty_cache()
                x, _ = batch_data
                
                # 前向传播 - 仅重建
                x_hat, z_hyp = model(x, batch_count=batch_count)
                
                # --- 最终的预热逻辑 ---
                if batch_count < args.reg_warmup_batches:
                    current_lambda_reg = 0.0
                else:
                    current_lambda_reg = final_lambda_reg
                
                # 计算损失 (注意：目标x，不是x_target)
                loss, recon_loss, p_loss, recon_loss_val, reg_loss_val, p_loss_val, color_loss_val = calc_loss_with_regularization(
                    x, x_hat, z_hyp, model.manifold, x_train_var, perceptual_loss,
                    lambda_reg=current_lambda_reg,
                    lambda_p=args.lambda_p
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 损失为{loss.item()}，跳过此批次")
                    continue
                # 反向传播
                accelerator.backward(loss)


                # 在这里添加梯度监控 (批次2895附近)
                if 5770  <= batch_count <= 5800:
                    problem_grads = monitor_gradients(model, batch_count)
                    if problem_grads:
                        print(f"\n======= 批次{batch_count}检测到梯度异常! =======")
                        print(f"问题梯度: {len(problem_grads)}个，包括: {problem_grads[:3]}")
                
                # 批次2895后紧密监控权重状态
                if 5770 <= current_batch <= 5800:
                    try:
                        weights_ok = monitor_model_weights(model, current_batch)
                        if not weights_ok:
                            print(f"\n======= 批次{current_batch}检测到权重异常! =======")
                            # 诊断每个模块
                            for name, module in model.named_modules():
                                if hasattr(module, 'weight') and module.weight is not None:
                                    try:
                                        if isinstance(module.weight, torch.Tensor):
                                            w = module.weight.data
                                            print(f"层 {name}: 权重范围[{w.min().item():.4e}, {w.max().item():.4e}], "
                                                  f"NaN数量: {torch.isnan(w).sum().item()}, "
                                                  f"Inf数量: {torch.isinf(w).sum().item()}")
                                    except Exception as e:
                                        print(f"检查层 {name} 权重时出错: {str(e)}")
                                
                                if hasattr(module, 'bias') and module.bias is not None:
                                    try:
                                        if isinstance(module.bias, torch.Tensor):
                                            b = module.bias.data
                                            print(f"层 {name}: 偏置范围[{b.min().item():.4e}, {b.max().item():.4e}], "
                                                  f"NaN数量: {torch.isnan(b).sum().item()}, "
                                                  f"Inf数量: {torch.isinf(b).sum().item()}")
                                        else:
                                            print(f"层 {name}: 偏置类型为{type(module.bias)}，跳过检查")
                                    except Exception as e:
                                        print(f"检查层 {name} 偏置时出错: {str(e)}")
                    except Exception as e:
                        print(f"权重监控过程中出错: {str(e)}")
                # 梯度裁剪
                if args.grad_clip_value > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_value)

                nan_grad_detected = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        if 5770 <= batch_count <= 5800:
                            print(f"清零参数 {name} 的NaN/Inf梯度")
                        param.grad.zero_()  # 将NaN梯度归零
                        nan_grad_detected = True

                # 如果检测到大量NaN梯度，重置优化器动量状态
                if nan_grad_detected and 5770 <= batch_count <= 5800:
                    print("检测到NaN梯度，重置优化器动量状态...")
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p in optimizer.state:
                                param_state = optimizer.state[p]
                                if 'momentum_buffer' in param_state:
                                    param_state['momentum_buffer'].zero_()
                    
                optimizer.step()
                optimizer.zero_grad()
                
                    
                # 定期打印信息 (只在主进程中)
                if accelerator.is_main_process and batch_count % 100 == 0:
                    accelerator.print(
                        
                        f"Epoch {epoch}, Batch {batch_count}/{batches_per_epoch}, "
                        f"Total Loss: {loss.item():.6f}, Recon Loss: {recon_loss_val:.6f}, "
                        f"Perceptual Loss: {p_loss_val:.6f}, Reg Loss: {reg_loss_val:.6f}, "
                        f"Lambda_Reg: {current_lambda_reg:.6f}, Color Loss: {color_loss_val:.6f}"
                )
            # 只有在主进程中更新loss计数
            if accelerator.is_main_process:
                current_loss = loss.item()
                epoch_loss += current_loss

                # --- 恢复记录 ---
                history_total_losses.append(current_loss)
                history_recon_losses.append(recon_loss_val)
                history_perceptual_losses.append(p_loss_val)
                history_color_losses.append(color_loss_val)
                
                # 在第0个epoch的前1/5处保存最佳模型
                if epoch == 0 and batch_count < target_batch_for_checkpoint:
                    if current_loss < best_loss_in_first_fifth:
                        best_loss_in_first_fifth = current_loss
                        
                        accelerator.print(
                            f"\n[Epoch 0, 全部阶段] 发现新的最佳损失: {best_loss_in_first_fifth:.6f} at batch {batch_count}. 保存模型..."
                        )
                        
                        # 使用accelerator保存，确保多GPU兼容
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_path = os.path.join(args.output_dir, "best_model_first_fifth_epoch.pth")
                        
                        accelerator.save({
                            'encoder': unwrapped_model.encoder.state_dict(),
                            'decoder': unwrapped_model.decoder.state_dict(),
                            'epoch': epoch,
                            'batch': batch_count,
                            'loss': best_loss_in_first_fifth,
                            'adaptive_c': args.adaptive_c,
                            'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                            'args': vars(args)
                        }, save_path)
                        
                        accelerator.print(f"  -> 模型已保存至: {save_path}\n")

                batch_count += 1
                    # 记录当前批次的损失
                current_epoch_batch_losses.append(current_loss)
                interval_batch_losses.append(current_loss)   
            # 每2000个批次绘制一次累计损失曲线 (只在主进程中)
            if accelerator.is_main_process and batch_count > 0 and batch_count % 2000 == 0:
                plot_cumulative_losses(
                    history_total_losses,
                    history_recon_losses,
                    history_perceptual_losses,
                    history_color_losses,
                    args.output_dir
                )
                accelerator.print(f"Epoch {epoch}, 已更新累计损失图: latest_loss.png")
            
            # 可视化部分 (只在主进程中)
            if accelerator.is_main_process and (batch_count % visualization_interval == 0 or batch_count == 100):
                visualize_reconstructions(model, validation_loader, f"{epoch}_batch{batch_count}", args.output_dir + "/intermediate")
                analyze_embeddings(z_hyp, epoch, batch_count, args.output_dir)
                accelerator.print(f"Epoch {epoch}, Batch {batch_count}: 完成中间可视化和嵌入分析")
                    




        if accelerator.is_main_process and interval_batch_losses and batch_count - last_plot_batch >= 100:
            # 这个部分不再需要，因为损失图现在是累积的
            pass
        # 保存当前epoch的所有批次损失，用于最终的详细损失图
        batch_losses=[]
        if accelerator.is_main_process:
            batch_losses.append(current_epoch_batch_losses)
        
        
        
        # 计算平均损失
        if accelerator.is_main_process:
            avg_loss = epoch_loss / batch_count
            accelerator.print(f"Epoch {epoch} 完成, 平均损失: {avg_loss:.6f}")
            train_losses.append(avg_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for val_x, _ in validation_loader:
                val_x_hat, _ = model(val_x)
                val_batch_loss = F.mse_loss(val_x_hat, val_x) / x_train_var
                val_loss += val_batch_loss.item()
                val_count += 1
        
        # 同步验证结果
        val_loss = accelerator.gather(torch.tensor(val_loss, device=accelerator.device)).sum().item()
        val_count = accelerator.gather(torch.tensor(val_count, device=accelerator.device)).sum().item()
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        accelerator.print(f"验证损失: {avg_val_loss:.6f}")
        if accelerator.is_main_process:
            val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)
        # 记录日志
        if accelerator.is_main_process and args.log_with is not None:
            logs = {"train_loss": avg_loss, "val_loss": avg_val_loss, "learning_rate": optimizer.param_groups[0]['lr']}
            accelerator.log(logs, step=epoch)

        # 定期在日志中添加
        if accelerator.is_main_process and batch_count % 1000 == 0:
            # 获取当前曲率
            c_value = model.manifold.c.item()
            print(f"批次{batch_count} 当前曲率: {c_value}")
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打印综合状态
            accelerator.print("\n" + "="*50)
            accelerator.print(f"状态摘要 [Epoch {epoch}, Batch {batch_count}]")
            accelerator.print(f"曲率: {c_value}")
            accelerator.print(f"学习率: {current_lr}")
            accelerator.print(f"最近10个批次平均损失: {np.mean(current_epoch_batch_losses[-10:]):.6f}")
            accelerator.print(f"最近1000个批次损失标准差: {np.std(current_epoch_batch_losses[-1000:]):.6f}")
            accelerator.print("="*50 + "\n")
        if accelerator.is_main_process:
            # 保存每个epoch的检查点
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_path = f"{args.output_dir}/epoch_{epoch}_autoencoder.pth"
            torch.save({
                'encoder': unwrapped_model.encoder.state_dict(),
                'decoder': unwrapped_model.decoder.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
                'adaptive_c': args.adaptive_c,
                'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                'args': vars(args)
            }, checkpoint_path)
            accelerator.print(f"保存epoch {epoch}的检查点至: {checkpoint_path}")
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                checkpoint_path = f"{args.output_dir}/best_autoencoder.pth"
                torch.save({
                    'encoder': unwrapped_model.encoder.state_dict(),
                    'decoder': unwrapped_model.decoder.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'adaptive_c': args.adaptive_c,
                    'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                    'args': vars(args)
                }, checkpoint_path)
                accelerator.print(f"保存最佳模型至: {checkpoint_path}")
            
            # epoch结束后的可视化
            visualize_reconstructions(unwrapped_model, validation_loader, epoch, args.output_dir)
    
    # 保存最终模型
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = f"{args.output_dir}/final_autoencoder.pth"
        torch.save({
            'encoder': unwrapped_model.encoder.state_dict(),
            'decoder': unwrapped_model.decoder.state_dict(),
            'epoch': args.epochs - 1,
            'loss': avg_val_loss,
            'adaptive_c': args.adaptive_c,
            'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
            'args': vars(args)
        }, final_path)
        accelerator.print(f"保存最终模型至: {final_path}")
        
        accelerator.print(f"预训练完成! 最佳验证损失: {best_loss:.6f}")
    
    # 结束日志追踪
    if accelerator.is_main_process and args.log_with is not None:
        accelerator.end_training()

if __name__ == "__main__":
    train_autoencoder()