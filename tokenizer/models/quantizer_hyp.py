import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import geoopt

sys.path.append('/ext/work')

from hyplib.manifolds.lmath import poincare_to_lorentz, lorentz_to_poincare, dist
from hyplib.manifolds.lorentzian import Lorentz

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
    def __init__(self, n_e, e_dim, beta, radial_bins=16, max_radius=18.0, 
                use_ema=False, ema_decay=0.99, c=1.0):
        super().__init__()
        assert n_e % radial_bins == 0, "n_e 必须被 radial_bins 整除"
        self.n_e          = n_e
        self.e_dim        = e_dim           # Poincaré 球的空间维度
        self.beta         = beta
        self.radial_bins  = radial_bins
        self.angular_bins = n_e // radial_bins
        self.max_radius   = max_radius
        self.manifold     = Lorentz(c=c)
        
        # 添加EMA支持
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 径向中心（超曲半径 r）
        log_r = torch.linspace(0., torch.log(torch.tensor(max_radius)), radial_bins)
        self.r_centres = nn.Parameter(torch.exp(log_r))  # (radial_bins,)

        # 角度中心：R^e_dim 上的单位向量
        self.angular_codebook = nn.Embedding(self.angular_bins, self.e_dim- 1)
        with torch.no_grad():
            v = torch.randn(self.angular_bins, self.e_dim-1)
            self.angular_codebook.weight.copy_(F.normalize(v, dim=-1))
            
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

    def forward(self, u_hyp,debug_step=None):
        """
        直接接受双曲输入的向量量化
        输入: u_hyp [B, C+1, H, W] - Lorentz模型上的点（包含时间分量）
        """
        if debug_step is not None and debug_step < 100:
            check_tensor(u_hyp, "量化器输入", debug_step)
        
        u_hyp_shape = u_hyp.shape
        u_hyp_flat = u_hyp.reshape(-1, u_hyp_shape[-1])

        # --- 分解 ---
        # 计算双曲半径 (arccosh(t))
        u_time = u_hyp_flat[:, 0:1]
        u_space = u_hyp_flat[:, 1:]
        r = torch.acosh(u_time.clamp(min=1.0 + 1e-7))
        w = F.normalize(u_space, dim=1)

        # --- 量化 ---
        # 径向量化
        r_centres = torch.clamp(self.r_centres, min=1e-2, max=self.max_radius)
        dist_r2 = (r - r_centres)**2
        r_idx = dist_r2.argmin(dim=-1)
        r_hard = r_centres[r_idx]

        # 角度量化
        sim = torch.matmul(w, self.angular_codebook.weight.t())
        w_idx = sim.argmax(dim=-1)
        w_hard = self.angular_codebook(w_idx)

        # --- 重建 ---
        # 从极坐标重建Poincaré点
        x_q_poinc = from_polar(r_hard.unsqueeze(-1), w_hard)
        # 转换回Lorentz模型
        x_q_hyp_flat = poincare_to_lorentz(x_q_poinc, k=self.manifold.k)
        # 关键：稳定化并投影回流形
        x_q_hyp_flat = self.manifold.projx(x_q_hyp_flat)

        # --- 计算损失 ---
        # VQ 损失: 鼓励编码器输出接近码本向量
        # 使用 detach() 来阻止梯度流向码本
        loss = self.beta * self.manifold.dist(u_hyp_flat, x_q_hyp_flat.detach()).mean()

        # --- 直通梯度 ---ß
        # 将 x_q_hyp_flat 的梯度直接传递给 u_hyp_flat
        z_q_hyp_flat = u_hyp_flat + (x_q_hyp_flat - u_hyp_flat).detach()
        # 确保最终输出仍在流形上
        z_q_hyp = self.manifold.projx(z_q_hyp_flat.view(u_hyp_shape))

        # --- 计算困惑度和使用率 ---
        r_idx_flat = r_idx.flatten()
        w_idx_flat = w_idx.flatten()
        combined_idx = r_idx_flat * self.angular_bins + w_idx_flat
        e_mean = F.one_hot(combined_idx, self.n_e).float().mean(0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        codebook_usage = torch.mean((e_mean > 0).float())

        # --- EMA 更新 (如果启用) ---
        if self.use_ema and self.training:
            # 更新径向码本
            r_one_hot = F.one_hot(r_idx_flat, self.radial_bins).float()
            r_sum = r_one_hot.sum(0)
            dw_r = r_one_hot.t() @ r.flatten()
            
            self.r_cluster_size = self.r_cluster_size * self.ema_decay + (1 - self.ema_decay) * r_sum
            self.r_centres_ema = self.r_centres_ema * self.ema_decay + (1 - self.ema_decay) * dw_r
            
            n_r = self.r_cluster_size.sum()
            r_centres_normalized = self.r_centres_ema / (self.r_cluster_size + 1e-5)
            self.r_centres.data.copy_(r_centres_normalized)

            # 更新角度码本
            w_one_hot = F.one_hot(w_idx_flat, self.angular_bins).float()
            w_sum = w_one_hot.sum(0)
            dw_w = w_one_hot.t() @ w
            
            self.angular_cluster_size = self.angular_cluster_size * self.ema_decay + (1 - self.ema_decay) * w_sum
            self.angular_ema = self.angular_ema * self.ema_decay + (1 - self.ema_decay) * dw_w
            
            n_w = self.angular_cluster_size.sum()
            angular_normalized = self.angular_ema / (self.angular_cluster_size.unsqueeze(1) + 1e-5)
            self.angular_codebook.weight.data.copy_(F.normalize(angular_normalized, dim=-1))

        return loss, z_q_hyp, perplexity, None, None, codebook_usage, e_mean