import torch
import torch.nn as nn
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from hyplib.manifolds.lorentzian import Lorentz
from models.adaptive_curvature import AdaptiveCurvature





class StandardAutoencoder(nn.Module):
    def __init__(self, h_dim, n_res_layers, c=1.0, adaptive_c=True):
        super(StandardAutoencoder, self).__init__()
        
        # 创建统一的流形实例
        self.adaptive_c = adaptive_c
        
        if adaptive_c:
            self.curvature = AdaptiveCurvature(initial_c=c, min_c=0.1, max_c=10.0)
            self.manifold = Lorentz(c=c)
        else:
            self.manifold = Lorentz(c=c)
        
        # 编码器和解码器
        self.encoder = EncoderLorentz(
            in_channels_euc=3, 
            h_dim_tan=h_dim, 
            n_res_layers=n_res_layers,
            c=c,
            manifold=self.manifold
        )
            
        # 确保维度匹配
        encoder_output_dim = self.encoder.out_channels
        embedding_dim = encoder_output_dim + 1
        
        # 解码器
        self.decoder = DecoderLorentz(
            in_channels_hyp=embedding_dim,
            h_dim_tan=h_dim,
            n_res_layers=n_res_layers,
            out_channels_euc=3,
            c=c,
            manifold=self.manifold
        )

    def forward(self, x, batch_count=None):  # 添加batch_count参数
        # 数据预处理 - 标准化到[-1,1]范围
        # x_normalized = x * 2.0 - 1.0
        def stabilize_and_project(tensor, name=""):
            # 检查NaN和Inf
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"警告: 检测到{name}中存在NaN或Inf")
                # 替换NaN和Inf为0
                tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                                        torch.zeros_like(tensor), tensor)
                
            # 强制投影回流形
            return self.manifold.projx(tensor)
        
        # 自适应曲率更新
        if self.adaptive_c:
            c = self.curvature()
            self.manifold.c.data = c.data.to(self.manifold.c.data.dtype)
            self.encoder.manifold.c.data = c.data.to(self.encoder.manifold.c.data.dtype)
            self.decoder.manifold.c.data = c.data.to(self.decoder.manifold.c.data.dtype)
        
        # 编码 - 直接使用原始的、范围在[-1, 1]的输入数据x
        z_e_hyp = self.encoder(x, batch_count=batch_count)
        z_e_hyp = stabilize_and_project(z_e_hyp, "编码器输出")
        
        # 记录半径，但添加更严格的钳位和警告
        radius = torch.acosh(torch.clamp(z_e_hyp[:, 0], min=1.0+1e-5))
        if (radius > 5.0).any():
            print(f"警告: 检测到半径过大 max={radius.max().item()}")
            # 可选：强制限制最大半径
            if radius.max() > 10.0:  # 设置一个硬性上限
                print(f"执行强制半径限制，从{radius.max().item()}降到10.0")
                scale = 10.0 / radius.max().item()
                # 只缩放问题点
                mask = (radius > 10.0).unsqueeze(-1)
                scaled_points = self.manifold.expmap0(self.manifold.logmap0(z_e_hyp) * scale)
                z_e_hyp = torch.where(mask, scaled_points, z_e_hyp)
                z_e_hyp = self.manifold.projx(z_e_hyp)
        
        # 解码
        x_hat = self.decoder(z_e_hyp)
        
        return x_hat, z_e_hyp