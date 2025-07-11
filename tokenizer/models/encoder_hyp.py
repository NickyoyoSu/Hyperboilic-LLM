import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import geoopt  # 添加geoopt导入
sys.path.append('/ext/work')  # 添加HyperCore的父目录
from hypercore.manifolds.lorentzian import Lorentz
from hypercore.nn.conv.lorentz_batch_norm import LorentzBatchNorm2d
from hypercore.nn.conv.lorentz_residual_block import LorentzResidualBlock
from hypercore.nn.conv.conv_util_layers import LorentzActivation
from hypercore.nn.conv.lorentz_convolution import LorentzConv2d

class EncoderLorentz(nn.Module):
    def __init__(self, in_channels_euc=3, h_dim_tan=64, n_res_layers=2, c=1.0, manifold=None):
        super().__init__()
        self.manifold = manifold if manifold is not None else Lorentz(c=c)
        self.h_dim = h_dim_tan
        self.ch_mult = (1, 2, 2)
        self.num_resolutions = len(self.ch_mult)
        
        # 欧几里得到双曲的初始映射
        # 1. 欧几里得处理阶段
        self.conv_euc = nn.Conv2d(in_channels_euc, h_dim_tan, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn_euc = nn.BatchNorm2d(h_dim_tan)
        self.act_euc = nn.SiLU()
        
        
        # 下采样路径 - 使用HypLib组件
        curr_channels = h_dim_tan
        self.down_path = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            level_modules = nn.ModuleList()
            out_channels = h_dim_tan * self.ch_mult[i_level]  # 空间分量计划输出维度
            
            # 残差块修改 - 明确处理时间分量
            for i_block in range(n_res_layers):
                if i_block == 0 and i_level == 0:
                    # 第一个块的输入没有时间分量
                    in_chan = curr_channels
                else:
                    # 后续所有块的输入已包含时间分量
            
                    in_chan = curr_channels 
                
                #print(f"创建残差块: in_channels={in_chan}, out_channels={out_channels}")
                level_modules.append(
                    LorentzResidualBlock(
                        manifold_in=self.manifold,
                        in_channels=in_chan,  # 修改为包含时间分量的维度
                        out_channels=out_channels # LorentzResidualBlock会输出out_channels+1的维度
                    )
                )
                curr_channels = out_channels  # 更新跟踪的通道数
            
            # 下采样修改
            if i_level != self.num_resolutions - 1:
                level_modules.append(self._make_downsample_block(curr_channels + 1))  # +1因为已有时间分量
            
            self.down_path.append(level_modules)
            
        # 中间块 - 最终处理
        self.mid_block = LorentzResidualBlock(
            manifold_in=self.manifold,
            in_channels=curr_channels + 1,
            out_channels=curr_channels
        )
        
        # 输出维度
        self.out_channels = curr_channels
        
    def _make_downsample_block(self, channels):
        def get_Activation(manifold_in, act=F.relu, manifold_out=None):
            return LorentzActivation(manifold_in, act, manifold_out)
        """几何一致的下采样块"""
        return nn.Sequential(
            LorentzConv2d(
                manifold_in=self.manifold,
                in_channels=channels,
                out_channels=channels - 1,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            LorentzBatchNorm2d(
                manifold_in=self.manifold,
                num_channels=channels
            ),
            get_Activation(self.manifold, F.silu)
        )
            
    def forward(self, x):
        """前向传播，保持几何一致性"""
        # STAGE 1: 欧几里得处理
        x = self.conv_euc(x)
        x = self.bn_euc(x)
        x = self.act_euc(x)
        
        # STAGE 2: 映射到双曲空间
        x = x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
        time_component = torch.sqrt(1.0 + torch.sum(x**2, dim=-1, keepdim=True))
        x_hyp = torch.cat([time_component, x], dim=-1)
        
        # 使用geoopt的stabilize确保数值稳定性
        if hasattr(geoopt.utils, 'stabilize'):
            x_hyp = geoopt.utils.stabilize(x_hyp)
        
        # 确保点在流形上并检查维度
        x_hyp = self.manifold.projx(x_hyp)
        #print(f"初始双曲映射后维度: {x_hyp.shape}, 时间分量范围: {x_hyp[...,0].min():.4f}-{x_hyp[...,0].max():.4f}")
        
        # STAGE 3: 下采样路径，带维度跟踪
        expected_dims = []  # 用于跟踪预期维度
        curr_dim = self.h_dim + 1  # 初始维度包含时间分量
        
        for i_level, level_modules in enumerate(self.down_path):
            for i_block, layer in enumerate(level_modules):
                # 记录进入层前的维度
                #print(f"第{i_level}级第{i_block}层前，形状: {x_hyp.shape}")
                
                # 应用层并确保流形约束
                x_hyp_before = x_hyp.clone()
                x_hyp = layer(x_hyp)
                
                # 在每个模块后强制投影，防止误差累积
                x_hyp = self.manifold.projx(x_hyp)
        
        return x_hyp
    
    def _check_on_manifold(self, x, tolerance=1e-5):
        """健壮的流形约束检查"""
        try:
            check_result = self.manifold.check_point_on_manifold(x, rtol=tolerance, atol=tolerance)
            if isinstance(check_result, bool):
                return check_result
            return check_result.all()
        except:
            return False
        
    def _check_dimensions(self, x, expected_dim, name="张量"):
        """验证张量维度是否符合预期，调试用"""
        if x.shape[-1] != expected_dim:
            #print(f"维度错误: {name} 维度为 {x.shape[-1]}，期望 {expected_dim}")
            return False
        return True