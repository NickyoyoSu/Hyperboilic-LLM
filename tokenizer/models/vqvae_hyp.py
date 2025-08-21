import torch
import torch.nn as nn
import geoopt
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from models.quantizer_hyp import VectorQuantizer
from models.quantizer_simple import StandardHyperbolicQuantizer
from hyplib.manifolds.lorentzian import Lorentz
from models.adaptive_curvature import AdaptiveCurvature
from models.cluster_aware_quantizer import ClusterAwareVectorQuantizer
import torch.nn.functional as F
import sys
sys.path.append('/ext/work')

class VQVAE_HYP(nn.Module):
    def __init__(self, h_dim, n_res_layers, n_embeddings, beta, 
                 save_img_embedding_map=False, use_ema=False, ema_decay=0.99, 
                 radial_bins=16, max_radius=20.0, c=1.0, initial_temp=5.0,
                 adaptive_c=True, use_cluster_quantizer=False, num_clusters=8):
        super(VQVAE_HYP, self).__init__()
        
        # 创建统一的流形实例
        self.manifold = Lorentz(c=c)
        self.adaptive_c = adaptive_c
        self.contrast_temp = 0.07

        if adaptive_c:
            self.curvature = AdaptiveCurvature(initial_c=c, min_c=0.1, max_c=10.0)
            # 创建初始流形实例，后续会动态更新曲率
            self.manifold = Lorentz(c=c)
        else:
            # 使用固定曲率
            self.manifold = Lorentz(c=c)

        # 量化器预热步数（默认0表示不预热），可由外部脚本覆盖
        self.quantizer_warmup_steps = 0
        
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
        '''
        if use_cluster_quantizer:
            self.vq = ClusterAwareVectorQuantizer(
                n_e=n_embeddings,
                e_dim=embedding_dim,
                beta=beta,
                radial_bins=radial_bins,
                max_radius=max_radius,
                use_ema=use_ema,
                ema_decay=ema_decay,
                initial_temp=initial_temp,
                c=self.manifold.c,
                num_clusters=num_clusters
            )
        else:
            self.vq = VectorQuantizer(
                n_e=n_embeddings,
                e_dim=embedding_dim,
                beta=beta,
                radial_bins=radial_bins,
                max_radius=max_radius,
                use_ema=use_ema,
                ema_decay=ema_decay,
                initial_temp=initial_temp,
                c=self.manifold.c
            )
        '''
        self.vq = StandardHyperbolicQuantizer(
            n_e=n_embeddings,
            e_dim=encoder_output_dim,  # e_dim 是空间维度，不包含时间
            beta=beta,
            manifold=self.manifold
        )
        
        '''
        self.vq = StandardHyperbolicQuantizer(n_e=n_embeddings,
                                                  e_dim=encoder_output_dim, # 传递空间维度
                                                  beta=beta,
                                          manifold=self.manifold)'''    
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


    def forward(self, x, x_aug=None, verbose=False, debug_step=None, global_step=None, reset_dead_codes=False):
        """支持多视角学习和量化器预热的前向传播"""
        # 自适应曲率更新
        if self.adaptive_c:
            c = self.curvature()
            # 更新参数数据时确保类型匹配
            self.manifold.c.data = c.data.to(self.manifold.c.data.dtype)
            self.vq.manifold.c.data = c.data.to(self.vq.manifold.c.data.dtype)
            self.encoder.manifold.c.data = c.data.to(self.encoder.manifold.c.data.dtype)
            self.decoder.manifold.c.data = c.data.to(self.decoder.manifold.c.data.dtype)
        
        # 原始视图处理
        z_e_hyp = self._encode(x)
        z_e_hyp = self.manifold.projx(z_e_hyp)
        
        # 提取切空间特征用于聚类
        with torch.no_grad():
            z_e_tan = self.manifold.logmap0(z_e_hyp)[:, 1:]
        
        # 量化原始视图（支持预热：前 self.quantizer_warmup_steps 使用 z_e 直接训练解码器）
        bypass_quantizer = (
            self.training and global_step is not None and self.quantizer_warmup_steps is not None
            and global_step < int(self.quantizer_warmup_steps)
        )

        if bypass_quantizer:
            embedding_loss = torch.tensor(0.0, device=x.device)
            perplexity = torch.tensor(0.0, device=x.device)
            diversity_loss = torch.tensor(0.0, device=x.device)
            codebook_usage = torch.zeros(1, device=x.device)
            z_q_hyp = z_e_hyp
        else:
            # 如果是聚类量化器，传递特征给它
            if hasattr(self.vq, 'update_clusters'):
                embedding_loss, z_q_hyp, perplexity, diversity_loss, codebook_usage = self._quantize(
                    z_e_hyp, features_for_clustering=z_e_tan if self.training else None, reset_dead_codes=reset_dead_codes
                )
            else:
                embedding_loss, z_q_hyp, perplexity, diversity_loss, codebook_usage = self._quantize(z_e_hyp, reset_dead_codes=reset_dead_codes)
            
        z_q_hyp = self.manifold.projx(z_q_hyp)
        x_hat = self._decode(z_q_hyp)
        
        # 多视角学习 - 如果提供了增强视图
        contrastive_loss = 0.0
        if x_aug is not None and self.training:
            # 编码增强视图
            z_e_hyp_aug = self._encode(x_aug)
            z_e_hyp_aug = self.manifold.projx(z_e_hyp_aug)
            
            # 将两个视图的特征拍平
            batch_size = z_e_hyp.shape[0]
            z_flat = z_e_hyp.reshape(batch_size, -1, z_e_hyp.shape[-1])
            z_aug_flat = z_e_hyp_aug.reshape(batch_size, -1, z_e_hyp_aug.shape[-1])
            
            # 计算特征维度均值
            z_flat = z_flat.mean(dim=1)  # [B, C+1]
            z_aug_flat = z_aug_flat.mean(dim=1)  # [B, C+1]
            
            # 计算双曲距离
            neg_dist = -self.manifold.dist(
                z_flat.unsqueeze(1).expand(-1, batch_size, -1),
                z_aug_flat.unsqueeze(0).expand(batch_size, -1, -1)
            )  # [B, B]
            
            # InfoNCE损失
            labels = torch.arange(batch_size).to(x.device)
            contrastive_loss = F.cross_entropy(neg_dist / self.contrast_temp, labels)
            
            # 更新总损失
            embedding_loss = embedding_loss + 0.2 * contrastive_loss
        
        return embedding_loss, x_hat, perplexity, diversity_loss, z_e_hyp, codebook_usage

    
    # 将原来forward中的逻辑拆分为三个辅助方法
    def _encode(self, x):
        return self.encoder(x)
    
# 修改辅助方法，使其不接受debug_step参数
    def _quantize(self, z_e_hyp, features_for_clustering=None, reset_dead_codes=False):
        # 确保编码后的张量在流形上
        if not self._check_on_manifold(z_e_hyp):
            z_e_hyp = self.manifold.projx(z_e_hyp)
        
        # 根据是否提供聚类特征决定调用方式
        if features_for_clustering is not None and hasattr(self.vq, 'update_clusters'):
            return self.vq(z_e_hyp, features_for_clustering=features_for_clustering, reset_dead_codes=reset_dead_codes)
        else:
            return self.vq(z_e_hyp, reset_dead_codes=reset_dead_codes)
    
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
