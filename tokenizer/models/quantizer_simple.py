import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ext/work')
from hyplib.manifolds.lorentzian import Lorentz

class StandardHyperbolicQuantizer(nn.Module):
    """
    一个标准的、基于距离的洛伦兹流形向量量化器。

    此版本使用单个统一的码本，并通过计算真实双曲距离来找到最近邻。
    它比极坐标分解法更健壮，是更可靠的基线模型。
    """
    def __init__(self, n_e, e_dim, beta, c=1.0, manifold=None):
        """
        参数:
            n_e (int): 码本中嵌入向量的数量。
            e_dim (int): 嵌入向量的空间维度。总维度将是 e_dim + 1 (时间分量)。
            beta (float): 承诺损失的权重。
            c (float): 流形的曲率。
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.manifold = manifold if manifold is not None else Lorentz(c=c)

        # --- 码本 ---
        # 使用单个 nn.Embedding 层来存储所有码本向量。
        # 每个向量的维度是 e_dim + 1，以包含时间分量。
        self.embedding = nn.Embedding(self.n_e, self.e_dim + 1)
        self._initialize_codebook_on_manifold()

    def _initialize_codebook_on_manifold(self):
        """
        通过在不同半径上均匀采样，更稳健地初始化码本向量。
        这能确保码本在训练开始时就有一个良好的分布。
        """
        with torch.no_grad():
            # 1. 在 [1.0, 6.0] 范围内均匀采样半径
            radii = torch.rand(self.n_e, 1) * 5 + 1.0  # 半径范围 [1.0, 6.0]
            time_component = torch.cosh(radii)

            # 2. 随机生成空间方向 (在单位球面上)
            spatial_directions = F.normalize(torch.randn(self.n_e, self.e_dim), p=2, dim=-1)

            # 3. 计算空间分量的大小
            # t^2 - ||x||^2 = 1/c (假设 c=1) => ||x|| = sqrt(t^2 - 1)
            spatial_magnitude = torch.sqrt(torch.clamp(time_component**2 - 1.0, min=1e-7))

            # 4. 组合成最终的空间向量
            spatial_vectors = spatial_directions * spatial_magnitude

            # 5. 组合时间分量和空间分量，并投影以确保数值精度
            lorentz_vectors = torch.cat([time_component, spatial_vectors], dim=-1)
            self.embedding.weight.data.copy_(self.manifold.projx(lorentz_vectors))

            # 打印初始化信息以供验证
            init_radii = self.manifold.dist0(self.embedding.weight.data)
            print(f"码本初始化完成。半径范围: [{init_radii.min().item():.2f}, {init_radii.max().item():.2f}], "
                  f"平均半径: {init_radii.mean().item():.2f}, 半径标准差: {init_radii.std().item():.2f}")

    def _reset_dead_codes(self, u_hyp_flat, e_mean):
        """
        识别并重置长时间未使用的码本向量（僵尸码）。
        修复：不再从当前批次复制向量，而是生成新的、安全的向量来替换僵尸码。
        """
        with torch.no_grad():
            # 计算每个码本向量的使用计数（近似值）
            usage_counts = e_mean * u_hyp_flat.shape[0]
            
            # 找到使用次数低于阈值的“僵尸码”
            dead_mask = usage_counts < 1.0
            dead_indices = torch.where(dead_mask)[0]
            num_dead = len(dead_indices)
            
            if num_dead > 0:
                print(f"--- 发现并重置 {num_dead} 个僵尸码 ---")
                
                # --- 更安全的重置逻辑 ---
                # 1. 在安全的半径范围内随机采样
                new_radii = torch.rand(num_dead, 1, device=self.embedding.weight.device) * 5.0 + 1.0  # 半径范围 [1.0, 6.0]
                new_time = torch.cosh(new_radii)

                # 2. 随机生成空间方向
                new_directions = F.normalize(torch.randn(num_dead, self.e_dim, device=self.embedding.weight.device), p=2, dim=-1)

                # 3. 计算对应的空间分量大小
                new_magnitude = torch.sqrt(torch.clamp(new_time**2 - 1.0, min=1e-7))

                # 4. 组合成新的双曲向量
                new_spatial = new_directions * new_magnitude
                new_vectors = torch.cat([new_time, new_spatial], dim=-1)

                # 5. 投影以确保在流形上，然后执行替换
                self.embedding.weight.data[dead_indices] = self.manifold.projx(new_vectors)


    def forward(self, u_hyp, debug_step=None, global_step=None, reset_dead_codes=False):
        """
        参数:
            u_hyp (torch.Tensor): 编码器的输出，形状为 [B, H, W, C+1]。
            global_step (int): 当前训练步数，用于触发僵尸码重置。
            reset_dead_codes (bool): 是否执行重置。
        """
        u_hyp_shape = u_hyp.shape
        u_hyp_flat = u_hyp.reshape(-1, self.e_dim + 1)
        num_vectors = u_hyp_flat.shape[0]

        # --- 分块量化以节省显存 ---
        min_indices = []
        chunk_size = 1024  # 调整此值以适应您的显存，可以从 512, 1024, 2048 尝试

        for i in range(0, num_vectors, chunk_size):
            # 获取当前块
            u_chunk = u_hyp_flat[i:i+chunk_size]

            # 计算当前块与整个码本的距离
            # u_chunk: [chunk_size, C+1] -> [chunk_size, 1, C+1]
            # embedding.weight: [n_e, C+1] -> [1, n_e, C+1]
            # 距离张量形状: [chunk_size, n_e] -> 这是可控的！
            distances_chunk = self.manifold.dist(u_chunk.unsqueeze(1), self.embedding.weight.unsqueeze(0))

            # 找到最近的索引并保存
            min_indices_chunk = torch.argmin(distances_chunk, dim=-1)
            min_indices.append(min_indices_chunk)

            # 释放中间张量显存（可选，但有帮助）
            del distances_chunk, min_indices_chunk, u_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 将所有块的索引拼接起来
        min_indices = torch.cat(min_indices, dim=0)

        # --- 后续步骤保持不变 ---
        x_q_hyp_flat = self.embedding(min_indices)

        # --- 损失计算 ---
        codebook_loss = self.manifold.dist(u_hyp_flat.detach(), x_q_hyp_flat).mean()
        commitment_loss = self.manifold.dist(u_hyp_flat, x_q_hyp_flat.detach()).mean()
        loss = codebook_loss + self.beta * commitment_loss

        # --- 直通梯度估计 ---
        # 修复：应用正确的STE逻辑。
        # 旧的实现方式在前向传播中抵消了量化效果。
        # 新的方式确保前向传播使用量化值，而反向传播的梯度直接流向编码器输出。
        z_q_hyp_flat = u_hyp_flat + (x_q_hyp_flat - u_hyp_flat).detach()
        z_q_hyp = z_q_hyp_flat.view(u_hyp_shape)

        # --- 困惑度计算 ---
        e_mean = F.one_hot(min_indices, self.n_e).float().mean(0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # 修复接口：计算多样性损失并返回与另一个量化器相同的签名
        diversity_loss = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        codebook_usage = e_mean

        # --- 定期重置僵尸码 ---
        if self.training and reset_dead_codes:
            self._reset_dead_codes(u_hyp_flat, e_mean)

        return loss, z_q_hyp, perplexity, diversity_loss, codebook_usage