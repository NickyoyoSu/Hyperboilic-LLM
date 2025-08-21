import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import geoopt
sys.path.append('/ext/work')
from hyplib.manifolds.lmath import poincare_to_lorentz, lorentz_to_poincare, dist
from hyplib.manifolds.lorentzian import Lorentz
import numpy as np

def from_polar(r, w):
    """
    r: (...), hyperbolic radius
    w: (..., e_dim), unit vector in tangent/Euc space
    returns x in Poincaré ball: x = tanh(r/2) * w
    """
    # 确保r的维度与w匹配以避免意外广播
    if r.dim() == 1 and w.dim() > 1:
        r = r.view(-1, 1)
    return torch.tanh(r / 2.0) * w
def check_tensor(tensor, name, step, detach=True):
    """检查张量是否包含NaN或Inf值"""
    if detach:
        tensor = tensor.detach()
            
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
            
    if has_nan or has_inf:
        print(f"[步骤 {step}] {name} 包含 NaN: {has_nan}, Inf: {has_inf}")
                
        # 获取统计信息
        if not (has_nan and tensor.isnan().all()):
            tensor_valid = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
            if len(tensor_valid) > 0:
                print(f"  有效值 - 最小: {tensor_valid.min().item():.4f}, 最大: {tensor_valid.max().item():.4f}")
        return False
    return True

def check_on_manifold(x_hyp, name, manifold, step, tolerance=1e-5):
    """检查点是否在双曲流形上"""
    t = x_hyp[..., 0:1]  # 时间分量
    x = x_hyp[..., 1:]   # 空间分量
            
    # 计算黎曼度量约束: t^2 - ||x||^2 = 1
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
    t_sq = t**2
    constraint = t_sq - x_norm_sq
            
    deviation = torch.abs(constraint - 1.0)
    max_deviation = deviation.max().item()
            
    if max_deviation > tolerance:
        print(f"[步骤 {step}] {name} 违反几何约束，最大偏差: {max_deviation:.6f}")
        max_idx = torch.argmax(deviation).item()
        flat_t = t.flatten()
        flat_norm = torch.sqrt(x_norm_sq).flatten()
        print(f"  问题点的t值: {flat_t[max_idx].item():.4f}, 空间范数: {flat_norm[max_idx].item():.4f}")
        return False
    return True


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, radial_bins=16, max_radius=1.1, 
                use_ema=False, ema_decay=0.99, c=1.0, initial_temp=5.0):
        super().__init__()
        assert n_e % radial_bins == 0, "n_e 必须被 radial_bins 整除"
        self.n_e          = n_e
        self.e_dim        = e_dim           # Poincaré 球的空间维度
        self.beta         = beta
        self.radial_bins  = radial_bins
        self.angular_bins = n_e // radial_bins
        self.max_radius   = max_radius
        self.manifold     = Lorentz(c=c)
        self.initial_temp = initial_temp
        self.current_temp = initial_temp  
        self.register_buffer('temp', torch.tensor(initial_temp)) 
        
        # 添加EMA支持
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 径向中心（超曲半径 r）
        if radial_bins > 0:
            # 设置新的径向中心分布以匹配编码器输出范围
    # 设置与编码器扩展范围匹配的径向分布
            r_min = 0.15
            r_max = 1.2 # 略大于编码器最大半径(0.6074)
            # 方法1：使用对数空间划分（推荐）
            r_values = torch.linspace(r_min, r_max, radial_bins).tolist()
            
            # 方法2：也可以使用分段线性分布
            # r_values = []
            # r_values.extend(torch.linspace(r_min, 1.0, radial_bins//4).tolist())  # 25%的bins用于[0.33-1.0]
            # r_values.extend(torch.linspace(1.0, 3.0, radial_bins//4).tolist())    # 25%的bins用于[1.0-3.0]
            # r_values.extend(torch.linspace(3.0, 7.0, radial_bins//4).tolist())    # 25%的bins用于[3.0-7.0]
            # r_values.extend(torch.linspace(7.0, r_max, radial_bins-len(r_values)).tolist()) # 剩余bins用于[7.0-10.0]
            
            self.r_centres = nn.Parameter(torch.tensor(r_values))

        # 角度中心：R^e_dim 上的单位向量
        self.angular_codebook = nn.Embedding(self.angular_bins, self.e_dim- 1)
        # 确保初始化时角度向量分布在不同的半径上
        with torch.no_grad():
            # 重新初始化径向中心，确保更均匀的分布
            r_span = r_max - r_min
            for i, r in enumerate(self.r_centres):
                # 轻微扰动半径，避免完全均匀分布
                noise = torch.randn(1).item() * 0.05 * r_span
                self.r_centres.data[i] = r_min + (i / (self.radial_bins-1)) * r_span + noise
                self.r_centres.data[i] = torch.clamp(self.r_centres.data[i], min=r_min, max=r_max)
            
            # --- 修复：正确初始化角度码本 ---
            # 角度码本与径向分量无关，只需初始化一次。
            # 之前复杂的循环逻辑是错误的，因为它试图访问码本不存在的索引。
            v = torch.randn(self.angular_bins, self.e_dim - 1)
            self.angular_codebook.weight.data.copy_(F.normalize(v, dim=-1))
            
        # 为EMA更新添加缓冲区
        if self.use_ema:
            # 径向EMA缓冲区
            self.register_buffer('r_centres_ema', self.r_centres.clone().detach())
            self.register_buffer('r_cluster_size', torch.zeros(radial_bins))
            
            # 角度EMA缓冲区
            self.register_buffer('angular_ema', self.angular_codebook.weight.clone().detach())
            self.register_buffer('angular_cluster_size', torch.zeros(self.angular_bins))
            
            # EMA状态跟踪
            self.register_buffer('ema_initialized', torch.tensor(0))

        # 添加到VectorQuantizer类
    def temp_adjusted_dist(self, x, y, temp=None):
        """温度调节的双曲距离计算"""
        temp = temp if temp is not None else self.temp
        
        # 基本双曲距离
        hyp_dist = self.manifold.dist(x, y)
        
        # 计算半径差异惩罚
        r_x = torch.acosh(torch.clamp(x[:, 0], min=1.0+1e-5))
        r_y = torch.acosh(torch.clamp(y[:, 0], min=1.0+1e-5))
        radius_diff = torch.abs(r_x - r_y)
        
        # 添加半径差异惩罚项 - 从2.0增加到5.0
        radius_weight = 1  # 显著增加半径差异的重要性
        return hyp_dist + radius_weight * radius_diff

    def forward(self, u_hyp, debug_step=None, features_for_clustering=None):
        """
        直接接受双曲输入的向量量化
        输入: u_hyp [B, C+1, H, W] - Lorentz模型上的点（包含时间分量）
        """
        if debug_step is not None and debug_step < 100:
            check_tensor(u_hyp, "量化器输入", debug_step)
        
        u_hyp_shape = u_hyp.shape
        u_hyp_flat = u_hyp.reshape(-1, u_hyp_shape[-1])

        # --- 分解 ---
        u_time = u_hyp_flat[:, 0:1]
        u_space = u_hyp_flat[:, 1:]
        r = torch.acosh(u_time.clamp(min=1.0 + 1e-2))
        w = F.normalize(u_space, dim=1)
        
        # --- 改进的量化过程 --- TODO 直接投到lorentz空间？
        r_centres = torch.clamp(self.r_centres, min=1e-2, max=self.max_radius)
        
        # 1. 预筛选最可能的候选项
        top_k_r = min(3, self.radial_bins)   # 选择最近的几个半径
        top_k_w = min(5, self.angular_bins)  # 选择最相似的几个方向
        
        # 找到top-k最近的径向值
        dist_r = torch.abs(r - r_centres)  # 使用绝对差异而非平方差异
        _, top_r_indices = torch.topk(-dist_r, k=top_k_r, dim=-1)   # 负号使小距离排前面
        
        # 找到top-k最相似的角度
        sim = torch.matmul(w, self.angular_codebook.weight.t())
        _, top_w_indices = torch.topk(sim, k=top_k_w, dim=-1)
        
        # 2. 在筛选后的候选项中找到真正最近的 
        batch_size = u_hyp_flat.size(0)
        best_dists = torch.full((batch_size,), float('inf'), device=u_hyp_flat.device)
        best_r_idx = torch.zeros((batch_size,), dtype=torch.long, device=u_hyp_flat.device)
        best_w_idx = torch.zeros((batch_size,), dtype=torch.long, device=u_hyp_flat.device)
        
        # 高效计算：批处理避免循环
        for i in range(top_k_r):
            r_idx_batch = top_r_indices[:, i]
            r_vals = r_centres[r_idx_batch].unsqueeze(-1)  # [batch, 1]
            
            for j in range(top_k_w):
                w_idx_batch = top_w_indices[:, j]
                w_vals = self.angular_codebook(w_idx_batch)  # [batch, e_dim-1]
                
                # 计算这个r,w组合的双曲距离
                candidate_poinc = from_polar(r_vals, w_vals)
                candidate_hyp = poincare_to_lorentz(candidate_poinc, k=self.manifold.k)
                candidate_hyp = self.manifold.projx(candidate_hyp)
                
                # 计算与输入的双曲距离
                curr_dists = self.manifold.dist(u_hyp_flat, candidate_hyp)
                
                # 更新最佳匹配
                update_mask = curr_dists < best_dists
                best_dists[update_mask] = curr_dists[update_mask]
                best_r_idx[update_mask] = r_idx_batch[update_mask]
                best_w_idx[update_mask] = w_idx_batch[update_mask]
        
        # 获取最佳匹配的编码
        r_hard = r_centres[best_r_idx]
        w_hard = self.angular_codebook(best_w_idx)
        
        # --- 重建 ---
        x_q_poinc = from_polar(r_hard.unsqueeze(-1), w_hard)
        x_q_hyp_flat = poincare_to_lorentz(x_q_poinc, k=self.manifold.k)
        x_q_hyp_flat = self.manifold.projx(x_q_hyp_flat)

        # --- 计算损失 ---
        # VQ 损失: 鼓励编码器输出接近码本向量
        # 使用 detach() 来阻止梯度流向码本
        # 在forward方法中修改损失计算部分
        # --- 计算损失 ---
        if not self.use_ema:
            # 使用温度调节的距离
            codebook_loss = self.temp_adjusted_dist(u_hyp_flat.detach(), x_q_hyp_flat, self.temp).mean()
            commitment_loss = self.temp_adjusted_dist(u_hyp_flat, x_q_hyp_flat.detach(), self.temp).mean()
            loss = codebook_loss + self.beta * commitment_loss
        else:
            # 使用EMA时，只需要commitment loss
            loss = self.beta * self.temp_adjusted_dist(u_hyp_flat, x_q_hyp_flat.detach(), self.temp).mean()

        # --- 直通梯度 ---
        with torch.no_grad():
            if self.temp > 1.5:  # 高温状态
                # 使用简单的线性插值
                direction = x_q_hyp_flat.detach() - u_hyp_flat
                z_q_hyp = u_hyp_flat + direction
            else:
                # 标准双曲测地线
                direction = self.manifold.logmap(x_q_hyp_flat.detach(), u_hyp_flat)
                z_q_hyp = self.manifold.expmap(u_hyp_flat, direction)

        # 确保几何约束
        z_q_hyp = self.manifold.projx(z_q_hyp.view(u_hyp_shape))


        # --- 计算困惑度和使用率 ---
        r_idx_flat = best_r_idx.flatten()
        w_idx_flat = best_w_idx.flatten()
        combined_idx = r_idx_flat * self.angular_bins + w_idx_flat
        e_mean = F.one_hot(combined_idx, self.n_e).float().mean(0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # 修复：codebook_usage 应该返回 e_mean 向量本身，用于可视化
        # 而不是返回一个标量
        codebook_usage = e_mean

        # --- 新增：码本多样性损失 ---
        # 目标是最大化e_mean的熵，等同于最小化负熵
        diversity_loss = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        
        # 清理返回接口，移除多余的None
        return loss, z_q_hyp, perplexity, diversity_loss, codebook_usage
