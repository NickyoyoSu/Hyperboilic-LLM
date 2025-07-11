import torch
import torch.nn as nn
import geoopt
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from models.quantizer_hyp import VectorQuantizer
from hyplib.manifolds.lorentzian import Lorentz
from torch.utils.checkpoint import checkpoint
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

class VQVAE_HYP(nn.Module):
    def __init__(self, h_dim, n_res_layers,
                 n_embeddings, beta, save_img_embedding_map=False,
                 use_ema=False, ema_decay=0.99, 
                 radial_bins=16, max_radius=20.0, c=1.0):
        super(VQVAE_HYP, self).__init__()
        
        # 创建统一的流形实例
        self.manifold = Lorentz(c=c)
        

        
        # 传递给组件
        self.encoder = EncoderLorentz(
            in_channels_euc=3, 
            h_dim_tan=h_dim, 
            n_res_layers=n_res_layers,
            c=c,
            manifold=self.manifold
        )
            
        # 确保编码器输出的通道数正确
        encoder_output_dim = self.encoder.out_channels
        embedding_dim = encoder_output_dim + 1
        self.vq = VectorQuantizer(
            n_e=n_embeddings,
            e_dim=embedding_dim,
            beta=beta,
            radial_bins=radial_bins,
            max_radius=max_radius,
            use_ema=use_ema,
            ema_decay=ema_decay,
            c=self.manifold.c
        )

        # 双曲解码器
        self.decoder = DecoderLorentz(
            in_channels_hyp=embedding_dim,
            h_dim_tan=h_dim,
            n_res_layers=n_res_layers,
            out_channels_euc=3,
            c=c,
            manifold=self.manifold
        )

        # 可选：保存图像到嵌入映射的字典
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None


    def forward(self, x, verbose=False, debug_step=None):
        # 确保输入有梯度标记
        if self.training and not x.requires_grad:
            x.requires_grad_(True)
            
        debug_enabled = debug_step is not None and (debug_step < 100 or debug_step % 500 == 0)
        
        # 统一的前向传播流程（训练和评估模式相同）
        z_e_hyp = self._encode(x)
        embedding_loss, z_q_hyp, perplexity, min_encodings, encodings, codebook_usage, e_mean = self._quantize(z_e_hyp)
        x_hat = self._decode(z_q_hyp)
        
        # 调试代码
        
        return embedding_loss, x_hat, perplexity, codebook_usage, e_mean

    
    # 将原来forward中的逻辑拆分为三个辅助方法
    def _encode(self, x):
        return self.encoder(x)
    
# 修改辅助方法，使其不接受debug_step参数
    def _quantize(self, z_e_hyp):
        # 确保编码后的张量在流形上
        if not self._check_on_manifold(z_e_hyp):
            z_e_hyp = self.manifold.projx(z_e_hyp)
        
        # 添加返回语句，返回VQ模块的结果
        return self.vq(z_e_hyp)
    
    def _decode(self, z_q_hyp):
        return self.decoder(z_q_hyp)
    
    def _check_on_manifold(self, x, tolerance=1e-5):
        """健壮的流形约束检查"""
        try:
            check_result = self.manifold.check_point_on_manifold(x, rtol=tolerance, atol=tolerance)
            if isinstance(check_result, bool):
                return check_result
            return check_result.all()
        except:
            return False
