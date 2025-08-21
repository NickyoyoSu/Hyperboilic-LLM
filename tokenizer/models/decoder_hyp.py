import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ext/work')  
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
        
        # Note: in_channels_hyp already includes the time component (+1)
        self.in_channels = in_channels_hyp
        
        # Fixed channel ratio sequence, matching the encoder
        ch_mult = (4, 2, 1)
        self.num_resolutions = len(ch_mult)
        curr_channels = self.in_channels - 1  # Subtract the time component
        
        # 中间块
        self.mid_block = LorentzResidualBlock(
            manifold_in=self.manifold,
            in_channels=curr_channels ,  # spatial component only
            out_channels=curr_channels  # spatial component only
        )

        #print(f"===== 解码器初始化 =====")
        #print(f"in_channels_hyp: {in_channels_hyp}")
        #print(f"curr_channels (无时间分量): {curr_channels}")
        #print(f"mid_block.in_channels: {curr_channels + 1}")
        #print(f"mid_block.out_channels: {curr_channels}")
        #print(f"ch_mult: {ch_mult}")
        #===上采样路径构建===
        # upsampling
        self.up_path = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            # Components for each resolution level
            level_modules = nn.ModuleList()
            out_channels = h_dim_tan * ch_mult[min(i_level + 1, self.num_resolutions - 1)]
            
            #print(f"第{i_level}层上采样块: in_channels={curr_channels}, out_channels={out_channels}")


        #===Residual block addition===
            # Residual block
            for i_block in range(n_res_layers):
                level_modules.append(
                    LorentzResidualBlock(
                        manifold_in=self.manifold,
                        in_channels=curr_channels,  
                        out_channels=out_channels
                    )
                )
                #print(f"  第{i_level}层第{i_block}个残差块: in={curr_channels}, out={out_channels}")
                curr_channels = out_channels
        #===Upsampling layer added===
            # upsampling
            if i_level != self.num_resolutions - 1:
                # Use hyperbolic upsampling
                level_modules.append(self._make_upsample_block(curr_channels, out_channels))  # +1 includes the time component
               
            curr_channels = out_channels
            self.up_path.append(level_modules)
        
        # Hyperbolic to Euclidean mapping
        #=== Defines the transformation from hyperbolic to Euclidean space and generates the final RGB image output.===
        self.final_hyp_conv = LorentzConv2d(
            manifold_in=self.manifold,
            in_channels=curr_channels+1,
            out_channels=h_dim_tan//2,
            kernel_size=3,
            padding=1
        )
        #print(f"Final hyperbolic convolution: in={curr_channels}, out={h_dim_tan//2}")
        # 2. From hyperbolic mapping to Gelidean space
        self.final_euc_conv = nn.Conv2d(h_dim_tan//2, out_channels_euc, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()  # Modification: Use Tanh to output [-1, 1] range
        #print(f"最终欧几里得卷积: in={h_dim_tan//2}, out={out_channels_euc}")
        
    def _make_upsample_block(self, in_channels, out_channels):
        """Geometrically consistent upsampling block - channels contain temporal components"""
        def get_Activation(manifold_in, activation=F.relu, manifold_out=None):
            return LorentzActivation(manifold_in, activation, manifold_out)
        """
        """
        #print(f"\n=====Constructing upsampling blocks =====")
        #print(f"传入参数: in_channels={in_channels} (空间通道), out_channels={out_channels} (空间通道)")
        #print(f"对应总通道数: in_total={in_channels+1}, out_total={out_channels+1} (包含时间分量)")
            
        return nn.Sequential(
            LorentzConvTranspose2d(
                manifold_in=self.manifold,
                in_channels=in_channels+1,  #time compoent
                out_channels= out_channels,  #no time
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
        '''Entering LorentzConvTranspose2d → undergoing transposed convolution → becoming 258 channels (+1)
        Then passing through the internal LorentzConv2d → the output channels are out_channels + 1 → 259 channels
        Finally, passing to LorentzBatchNorm2d, which expects the number of channels to match the output of the previous layer'''

    def forward(self, x):
        """
        Decoder forward pass
        Parameters:
        x: Hyperbolic tensor (B, C, H, W) from the quantizer, where C = embedding_dim + 1
        """
        # --- Input Validation and Projection ---
        # Ensure input is on the manifold to prevent numerical error accumulation
        x = self.manifold.projx(x)

        # middle block
        h = self.mid_block(x)
        
        # upsampling path
        for i_level, level_modules in enumerate(self.up_path):
            for layer in level_modules:
                h = layer(h)
                h = self.manifold.projx(h)

        # STAGE 3: Final hyperbolic convolution
        h = self.final_hyp_conv(h)
        
        # STAGE 4: Mapping to Euclidean space
        # Extract spatial components for final Euclidean convolution
        h_euc = h[..., 1:]  # [B, H, W, C_spatial]
        h_euc = h_euc.permute(0, 3, 1, 2)  # [B, C_spatial, H, W]
        
        x_hat = self.final_euc_conv(h_euc)
        x_hat = self.final_act(x_hat)
        
        return x_hat
