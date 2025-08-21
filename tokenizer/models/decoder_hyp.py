import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ext/work')  # 添加HyperCore的父目录
from hypercore.manifolds.lorentzian import Lorentz
from hypercore.nn.conv.lorentz_convolution import LorentzConv2d, LorentzConvTranspose2d
from hypercore.nn.conv.lorentz_batch_norm import LorentzBatchNorm2d
from hypercore.nn.conv.lorentz_residual_block import LorentzResidualBlock
from hypercore.nn.conv.conv_util_layers import LorentzActivation

class DecoderLorentz(nn.Module):
    def __init__(self, in_channels_hyp, h_dim_tan=64, n_res_layers=2, out_channels_euc=3, c=1.0, manifold=None):
        super().__init__()
        self.manifold = manifold if manifold is not None else Lorentz(c=c)
        self.h_dim = h_dim_tan
        
        # 注意：in_channels_hyp 已经包含时间分量 (+1)
        self.in_channels = in_channels_hyp
        
        # 固定通道比例序列，与编码器匹配
        ch_mult = (4, 2, 1)
        self.num_resolutions = len(ch_mult)
        curr_channels = self.in_channels - 1  # 减去时间分量
        
        # 中间块
        self.mid_block = LorentzResidualBlock(
            manifold_in=self.manifold,
            in_channels=curr_channels ,  # 仅空间分量
            out_channels=curr_channels  # 仅空间分量
        )

        #print(f"===== 解码器初始化 =====")
        #print(f"in_channels_hyp: {in_channels_hyp}")
        #print(f"curr_channels (无时间分量): {curr_channels}")
        #print(f"mid_block.in_channels: {curr_channels + 1}")
        #print(f"mid_block.out_channels: {curr_channels}")
        #print(f"ch_mult: {ch_mult}")
        #===上采样路径构建===
        # 上采样路径
        self.up_path = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            # 每个分辨率级别的组件
            level_modules = nn.ModuleList()
            out_channels = h_dim_tan * ch_mult[min(i_level + 1, self.num_resolutions - 1)]
            
            #print(f"第{i_level}层上采样块: in_channels={curr_channels}, out_channels={out_channels}")


        #===残差块添加===
            # 残差块
            for i_block in range(n_res_layers):
                level_modules.append(
                    LorentzResidualBlock(
                        manifold_in=self.manifold,
                        in_channels=curr_channels,  # 不需要条件判断
                        out_channels=out_channels
                    )
                )
                #print(f"  第{i_level}层第{i_block}个残差块: in={curr_channels}, out={out_channels}")
                curr_channels = out_channels
        #===上采样层添加===
            # 上采样
            if i_level != self.num_resolutions - 1:
                # 使用双曲上采样
                level_modules.append(self._make_upsample_block(curr_channels, out_channels))  # +1包含时间分量
                #print(f"  第{i_level}层上采样: in={curr_channels}, out={out_channels} (空间分量)")
            curr_channels = out_channels
            self.up_path.append(level_modules)
        
        # 双曲到欧几里得的映射
        #=== 定义从双曲到欧几里得空间的转换，并生成最终RGB图像输出。===
        self.final_hyp_conv = LorentzConv2d(
            manifold_in=self.manifold,
            in_channels=curr_channels+1,
            out_channels=h_dim_tan//2,
            kernel_size=3,
            padding=1
        )
        #print(f"最终双曲卷积: in={curr_channels}, out={h_dim_tan//2}")
        # 2. 从双曲映射到几里得空间
        self.final_euc_conv = nn.Conv2d(h_dim_tan//2, out_channels_euc, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()  # 修改：使用Tanh以输出[-1, 1]范围
        #print(f"最终欧几里得卷积: in={h_dim_tan//2}, out={out_channels_euc}")
        
    def _make_upsample_block(self, in_channels, out_channels):
        """几何一致的上采样块 - channels包含时间分量"""
        def get_Activation(manifold_in, activation=F.relu, manifold_out=None):
            return LorentzActivation(manifold_in, activation, manifold_out)
        """
        """
        #print(f"\n===== 构建上采样块 =====")
        #print(f"传入参数: in_channels={in_channels} (空间通道), out_channels={out_channels} (空间通道)")
        #print(f"对应总通道数: in_total={in_channels+1}, out_total={out_channels+1} (包含时间分量)")
            
        return nn.Sequential(
            LorentzConvTranspose2d(
                manifold_in=self.manifold,
                in_channels=in_channels+1,  #含时间
                out_channels= out_channels,  #不含
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            LorentzBatchNorm2d(
                manifold_in=self.manifold,
                num_channels=out_channels +1, #
                space_method=True
            ),
            get_Activation(self.manifold, F.silu)
        )
        '''进入 LorentzConvTranspose2d → 经过转置卷积 → 变成258通道(+1)
        然后通过内部的 LorentzConv2d → 输出通道为 out_channels + 1 → 259通道
        最后传给 LorentzBatchNorm2d,它期望通道数与前一层输出匹配'''

    def forward(self, x):
        """
        解码器前向传播
        参数:
            x: 来自量化器的双曲张量 (B, C, H, W)，其中 C = embedding_dim + 1
        """
        # --- 输入验证和投影 ---
        # 确保输入在流形上，防止数值误差累积
        x = self.manifold.projx(x)

        # 中间块处理
        h = self.mid_block(x)
        
        # 上采样路径
        for i_level, level_modules in enumerate(self.up_path):
            for layer in level_modules:
                h = layer(h)
                h = self.manifold.projx(h)

        # STAGE 3: 最终双曲卷积
        h = self.final_hyp_conv(h)
        
        # STAGE 4: 映射到欧几里得空间
        # 提取空间分量进行最终的欧几里得卷积
        h_euc = h[..., 1:]  # [B, H, W, C_spatial]
        h_euc = h_euc.permute(0, 3, 1, 2)  # [B, C_spatial, H, W]
        
        x_hat = self.final_euc_conv(h_euc)
        x_hat = self.final_act(x_hat)
        
        return x_hat
