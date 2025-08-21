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
        self.ch_mult = (1, 2, 2) #定义三个分辨率层级通道倍增因子
        self.num_resolutions = len(self.ch_mult)
        # 警告节流
        self._warn_counter = 0
        self._warn_interval = 200
        
        # 1. 欧几里得处理阶段，在欧几里得空间处理输入图像
        self.conv_euc = nn.Conv2d(in_channels_euc, h_dim_tan, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn_euc = nn.BatchNorm2d(h_dim_tan, eps=1e-4)
        self.act_euc = nn.SiLU()
          
        # 下采样路径 跟踪通道数和创建模块列表，为每个分辨率级别准备容器。
        curr_channels = h_dim_tan
        self.down_path = nn.ModuleList()
        
        #残差块构建
        for i_level in range(self.num_resolutions):
            level_modules = nn.ModuleList()
            out_channels = h_dim_tan * self.ch_mult[i_level]  # 空间分量计划输出维度
            
            # 残差块修改 - 明确处理时间分量
            for i_block in range(n_res_layers):
                if i_block == 0 and i_level == 0:
                    # 第一个块的输入没有时间分量
                    in_chan = curr_channels 
                else:
                    in_chan = curr_channels 

                #残差块添加，添加双曲残差块，并更新跟踪的通道数。 注意：LorentzResidualBlock会输出out_channels+1的维度，因为包含时间分量。
                #print(f"创建残差块: in_channels={in_chan}, out_channels={out_channels}")
                level_modules.append(
                    LorentzResidualBlock(
                        manifold_in=self.manifold,
                        in_channels=in_chan,  
                        out_channels=out_channels 
                    )
                )
                curr_channels = out_channels  # 更新跟踪的通道数

            
            # 下采样修改
            #在每个分辨率级别(除了最后一个)后添加下采样块。 注意：curr_channels + 1表示包含时间分量
            if i_level != self.num_resolutions - 1:
                level_modules.append(self._make_downsample_block(curr_channels + 1))  # +1因为已有时间分量
            
            self.down_path.append(level_modules)
            
        # 中间块 - 最终处理
        self.mid_block = LorentzResidualBlock(
            manifold_in=self.manifold,
            in_channels=curr_channels ,
            out_channels=curr_channels
        )
        
        # 输出维度
        self.out_channels = curr_channels
        # 新增：安全系数，用于避免数值不稳定
        self.safety_factor = 0.95

    # 添加这个详细诊断函数

    def diagnose_tensor(self, tensor, name, batch_count=None, threshold=5770):
        """超详细的张量诊断，识别NaN/Inf的确切来源"""
        # 仅在接近关键批次时激活诊断
        if batch_count is None or batch_count >= threshold-5:
            if tensor is None:
                print(f"ERROR: {name} is None!")
                return True
                
            has_problem = torch.isnan(tensor).any() or torch.isinf(tensor).any()
            
            if has_problem:
                # 基础统计
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                total = tensor.numel()
                
                print(f"\n==== 详细诊断: {name} {'(批次 '+str(batch_count)+')' if batch_count else ''} ====")
                print(f"形状: {tensor.shape}, NaN: {nan_count}/{total}, Inf: {inf_count}/{total}")
                
                # 针对双曲点的特殊诊断
                if tensor.dim() > 1 and tensor.size(-1) > 1:
                    # 分析时间分量
                    time_comp = tensor[..., 0]
                    space_comp = tensor[..., 1:]
                    
                    tc_has_nan = torch.isnan(time_comp).any().item()
                    tc_has_inf = torch.isinf(time_comp).any().item()
                    tc_min = time_comp.min().item() if not tc_has_nan else "NaN"
                    tc_max = time_comp.max().item() if not tc_has_nan else "NaN"
                    
                    print(f"时间分量: 范围[{tc_min}~{tc_max}], NaN:{tc_has_nan}, Inf:{tc_has_inf}")
                    
                    # 分析空间分量
                    sc_has_nan = torch.isnan(space_comp).any().item()
                    sc_has_inf = torch.isinf(space_comp).any().item()
                    
                    if not sc_has_nan and not sc_has_inf:
                        space_norm = torch.norm(space_comp, dim=-1)
                        sn_min = space_norm.min().item()
                        sn_max = space_norm.max().item()
                        print(f"空间分量: 范数范围[{sn_min:.4f}~{sn_max:.4f}]")
                        
                        # 计算双曲约束
                        if not tc_has_nan and not tc_has_inf:
                            constraint = -time_comp**2 + torch.sum(space_comp**2, dim=-1)
                            c_min = constraint.min().item()
                            c_max = constraint.max().item()
                            expected = -1.0/self.manifold.c.item()
                            print(f"双曲约束(-t²+||x||²): [{c_min:.6f}~{c_max:.6f}], 应为:{expected:.6f}")
                            print(f"偏差: {abs(c_min-expected):.6e}~{abs(c_max-expected):.6e}")
                    else:
                        print("空间分量包含NaN/Inf，无法计算约束")
                        
                # 如果是小张量，打印所有值找出确切位置
                if tensor.numel() < 50 or (has_problem and tensor.numel() < 1000):
                    problem_mask = torch.isnan(tensor) | torch.isinf(tensor)
                    if problem_mask.any():
                        indices = problem_mask.nonzero()
                        for idx in indices[:10]:  # 只打印前10个问题
                            idx_tuple = tuple(idx.tolist())
                            val = tensor[idx_tuple]
                            print(f"  问题位置 {idx_tuple}: {val}")
                        
                        if len(indices) > 10:
                            print(f"  ...以及{len(indices)-10}个其他问题位置")
                
                return True
            return False
    # 新增：更健壮的稳定化函数
    def stabilize_tensor(self, tensor, name=""):
        """增强的数值稳定性处理"""
        # 全流程避免就地修改，保护 autograd 图
        out = tensor

        # 1) 替换 NaN/Inf（非就地）
        has_problem = torch.isnan(out).any() or torch.isinf(out).any()
        if has_problem:
            if torch.isnan(out).any():
                print(f"警告: {name}中存在NaN值，正在修复...")
            if torch.isinf(out).any():
                print(f"警告: {name}中存在Inf值，正在修复...")
            out = torch.where(torch.isnan(out) | torch.isinf(out), torch.zeros_like(out), out)

        # 2) 空间范数过大时缩放（非就地，重建张量）
        if out.shape[-1] > 1:
            space = out[..., 1:]
            max_norm = torch.norm(space, dim=-1, keepdim=True).max()
            if max_norm > 1e4:
                print(f"警告: {name}中空间分量过大 (最大范数: {max_norm.item():.2e})，正在缩放...")
                scale = min(1e4 / max_norm.item(), 1.0) * self.safety_factor
                space_scaled = space * scale
                time_recomputed = torch.sqrt(1.0 + torch.sum(space_scaled**2, dim=-1, keepdim=True))
                out = torch.cat([time_recomputed, space_scaled], dim=-1)

        # 3) 可选稳定化（假定返回非就地结果）
        if hasattr(geoopt.utils, 'stabilize'):
            out = geoopt.utils.stabilize(out)

        # 4) 最终投影（非就地）
        return self.manifold.projx(out)
        
    #下采样块构建函数, 创建几何一致的下采样块
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
                num_channels=channels,
                space_method=True 
            ),
            get_Activation(self.manifold, F.silu)
        )

    def _check_on_manifold(self, x, tolerance=1e-3):  # 适当放宽初始容差
        """与库无关的洛伦兹约束检查，兼容不同实现的符号约定"""
        try:
            t = x[..., 0]
            space = x[..., 1:]
            x_norm_sq = torch.sum(space**2, dim=-1)
            t_sq = t**2
            # 约定1（常见于hypercore）: -t^2 + ||x||^2 = -1/c
            c_val = None
            if hasattr(self.manifold, 'c') and isinstance(self.manifold.c, torch.Tensor):
                c_val = float(self.manifold.c.detach().cpu().item())
            elif hasattr(self.manifold, 'c'):
                try:
                    c_val = float(self.manifold.c)
                except Exception:
                    c_val = None
            expected1 = -1.0 / c_val if c_val and c_val != 0.0 else -1.0
            dev1 = torch.max(torch.abs((-t_sq + x_norm_sq) - expected1))
            # 约定2（常见于hyplib）: t^2 - ||x||^2 = 1
            expected2 = 1.0
            dev2 = torch.max(torch.abs((t_sq - x_norm_sq) - expected2))
            return (dev1 <= tolerance) or (dev2 <= tolerance)
        except Exception:
            # 发生异常时保守返回True，避免训练中断
            return True

            
    def forward(self, x, batch_count=None):
        # 在欧几里得到双曲转换之前增加输入监控
        is_critical = batch_count is not None and 5770 <= batch_count <= 5800
        
        if is_critical:
            # 监控输入统计
            print(f"\n----- 批次{batch_count}编码器输入监控 -----")
            print(f"输入形状: {x.shape}")
            print(f"输入统计: 均值={x.mean().item():.6f}, 标准差={x.std().item():.6f}")
            print(f"输入范围: 最小值={x.min().item():.6f}, 最大值={x.max().item():.6f}")
            
            # 检查异常值
            extreme_vals = (x < -3) | (x > 3)
            if extreme_vals.any():
                print(f"警告: 检测到{extreme_vals.sum().item()}个异常值")
        
        # STAGE 1: 欧几里得处理
        x1 = self.conv_euc(x)
        self.diagnose_tensor(x1, "卷积后", batch_count) 
        
        # 监控卷积输出
        if is_critical:
            print(f"卷积输出: 均值={x1.mean().item():.6f}, 标准差={x1.std().item():.6f}")
            print(f"卷积范围: 最小值={x1.min().item():.6f}, 最大值={x1.max().item():.6f}")
            print(f"卷积输出NaN: {torch.isnan(x1).sum().item()}")
        
        # 在BN层前特别监控
        if is_critical and hasattr(self, 'bn_euc'):
            bn = self.bn_euc
            print(f"BN层状态: running_mean范围=[{bn.running_mean.min().item():.4f}, {bn.running_mean.max().item():.4f}]")
            print(f"BN层状态: running_var范围=[{bn.running_var.min().item():.6f}, {bn.running_var.max().item():.6f}]")
            
            # 检查BN参数是否正常
            if torch.isnan(bn.running_mean).any() or torch.isnan(bn.running_var).any():
                print("严重警告: BN统计包含NaN!")
            if torch.isinf(bn.running_mean).any() or torch.isinf(bn.running_var).any():
                print("严重警告: BN统计包含Inf!")
            
            # 检查BN权重
            if hasattr(bn, 'weight') and bn.weight is not None:
                print(f"BN权重: min={bn.weight.min().item():.4f}, max={bn.weight.max().item():.4f}")
                if torch.isnan(bn.weight).any():
                    print("严重警告: BN权重包含NaN!")
        
        # 应用BN和激活
        x1 = self.bn_euc(x1)
        self.diagnose_tensor(x1, "BN后", batch_count)
        x1 = self.act_euc(x1)
        self.diagnose_tensor(x1, "激活后", batch_count) 
        
        # STAGE 2: 关键的双曲映射监控
        x1 = x1.permute(0, 2, 3, 1)
        
        # 关键监控: 平方范数计算，可能是问题源头
        sq_norm = torch.sum(x1**2, dim=-1, keepdim=True)
        
        if is_critical:
            print("\n----- 双曲映射关键计算监控 -----")
            print(f"平方范数统计: 均值={sq_norm.mean().item():.6f}, 标准差={sq_norm.std().item():.6f}")
            print(f"平方范数范围: 最小值={sq_norm.min().item():.6f}, 最大值={sq_norm.max().item():.6f}")
            
            # 特别警告极大值
            if sq_norm.max().item() > 1e4:
                print(f"严重警告: 平方范数出现极大值: {sq_norm.max().item():.4e}")
                # 打印出现极值的位置
                extreme_idx = (sq_norm > 1e4).nonzero()
                if len(extreme_idx) > 0:
                    print(f"极值索引示例: {extreme_idx[0].tolist()}")
                    # 查看对应位置的原始特征值
                    pos = extreme_idx[0].tolist()[:-1]  # 移除最后一维
                    print(f"对应原始特征值: {x1[pos].norm().item():.4e}")
        
        # 时间分量计算 - 这是可能出现数值问题的地方
        time_component = torch.sqrt(1.0 + sq_norm)
        
        if is_critical:
            print(f"时间分量统计: 均值={time_component.mean().item():.6f}, 标准差={time_component.std().item():.6f}")
            print(f"时间分量范围: 最小值={time_component.min().item():.6f}, 最大值={time_component.max().item():.6f}")
        
        # 拼接形成双曲点
        x_hyp = torch.cat([time_component, x1], dim=-1)
        
        # 使用geoopt的stabilize确保数值稳定性，数值稳定处理

        #x_hyp = geoopt.utils.stabilize(x_hyp)
        x_hyp = self.manifold.projx(x_hyp)
        x_hyp = self.stabilize_tensor(x_hyp, "初始双曲映射")
        #print(f"初始双曲映射后维度: {x_hyp.shape}, 时间分量范围: {x_hyp[...,0].min():.4f}-{x_hyp[...,0].max():.4f}")
        #应用所有下采样层和残差块，并在每步后投影回流形。
        for i_level, level_modules in enumerate(self.down_path):
            for i_block, layer in enumerate(level_modules):                # 层间检查
                # 先投影，再检查，避免误报
                x_hyp = self.manifold.projx(x_hyp)
                check_result = self._check_on_manifold(x_hyp)
                if not check_result:
                    if (self._warn_counter % self._warn_interval) == 0:
                        print(f"警告：第{i_level}级第{i_block}层前点不在流形上")
                    self._warn_counter += 1
                    x_hyp = self.manifold.projx(x_hyp)

                
                # 记录进入层前的维度
                #print(f"第{i_level}级第{i_block}层前，形状: {x_hyp.shape}")
                x_hyp = layer(x_hyp)
                # 在每个模块后立即投影，防止误差累积
                x_hyp = self.manifold.projx(x_hyp)
                # 进一步稳定（可选），仅在残差块或位于最后一层时执行，降低开销
                if isinstance(layer, LorentzResidualBlock) or (i_level == self.num_resolutions - 1 and i_block == len(level_modules) - 1):
                    x_hyp = self.stabilize_tensor(x_hyp, f"层{i_level}块{i_block}")
        
        #下采样玩了就中间快
        x_hyp = self.mid_block(x_hyp)
        x_hyp = self.stabilize_tensor(x_hyp, "最终输出")
        x_hyp = self.manifold.projx(x_hyp)

        # 半径安全检查与限制
        radius = torch.acosh(torch.clamp(x_hyp[..., 0], min=1.0+1e-5))
        max_radius = radius.max().item()
        
        if max_radius > 10.0:  # 更严格的半径限制
            print(f"警告: 输出半径过大 ({max_radius:.4f})，正在缩放...")
            # 计算缩放因子，限制最大半径
            scale_factor = 10.0 / max_radius * self.safety_factor
            
            # 转换到切空间，缩放，再转回来
            x_tan = self.manifold.logmap0(x_hyp)
            x_tan = x_tan * scale_factor
            x_hyp = self.manifold.expmap0(x_tan)
            
            # 再次确保在流形上
            x_hyp = self.manifold.projx(x_hyp)
            
            # 验证修复是否成功
            new_radius = torch.acosh(torch.clamp(x_hyp[..., 0], min=1.0+1e-5)).max().item()
            print(f"缩放后最大半径: {new_radius:.4f}")
        
        return x_hyp
        
