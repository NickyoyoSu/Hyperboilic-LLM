import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from .manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter

class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.num_heads = num_heads  # 确保 num_heads 一直是 4
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat


    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p


# filepath: [hypformer_multimodality.py](http://_vscodecontentref_/2)
    def full_attention(self, qs, ks, vs, attention_mask=None, output_attn=False):
        """双曲空间多头注意力实现 - 保留双曲特性"""
        batch_size, seq_len, num_heads, head_dim = qs.shape
        
        # 重塑张量形状以正确应用双曲内积
        # 从[batch, seq, heads, dim]变为[batch*heads, seq, dim]
        q_reshaped = qs.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        k_reshaped = ks.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        v_reshaped = vs.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        
        # 计算双曲内积注意力分数
        # 注意：我们希望得到[batch*heads, seq, seq]形状的注意力分数
        att_weights = 2 + 2 * self.manifold.cinner(q_reshaped, k_reshaped)  # [batch*heads, seq, seq]
        
        # 缩放并添加偏置
        att_weights = att_weights / self.scale + self.bias
        
        # 重塑为[batch, heads, seq, seq]
        att_weights = att_weights.reshape(batch_size, num_heads, seq_len, seq_len)
        
        # 应用因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=qs.device) * (-1e9), 
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        att_weights = att_weights + causal_mask
        
        # 应用注意力掩码(如果有)
        if attention_mask is not None:
            # 将注意力掩码转换为正确的4D形式
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, num_heads, seq_len, -1)  # [batch, heads, seq_len, seq_len]
            
            # 创建有效的掩码 (0 -> -inf, 1 -> 0)
            attn_mask = (1.0 - attn_mask.float()) * -1e9
            att_weights = att_weights + attn_mask
        
        # 应用softmax获取注意力权重
        att_weights = F.softmax(att_weights, dim=-1)  # [batch, heads, seq, seq]
        
        # 重塑v以适应矩阵乘法
        v_view = v_reshaped.reshape(batch_size, num_heads, seq_len, head_dim)
        
        # 应用注意力权重到值向量
        context = torch.matmul(att_weights, v_view)  # [batch, heads, seq, dim]
        
        # 将结果转回原始维度顺序
        context = context.permute(0, 2, 1, 3)  # [batch, seq, heads, dim]
        
        # 使用双曲空间中点运算聚合 - 保留双曲特性
        output = self.manifold.mid_point(context)
        
        if output_attn:
            return output, att_weights
        else:
            return output
    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
            qs = hyp_qs[..., 1:]
            ks = hyp_ks[..., 1:]
            v = hyp_vs[..., 1:]
            phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]
            phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]

            phi_qs = self.fp(phi_qs, p=self.power_k)  # [N, H, D]
            phi_ks = self.fp(phi_ks, p=self.power_k)  # [N, H, D]

            # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
            k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]

            # Step 2: Compute the kernel-transformed dot product of Q with the above result
            numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]

            # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
            denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
            denominator = denominator.unsqueeze(-1)  #

            # Step 4: Normalize the numerator with the denominator
            attn_output = numerator / (denominator + 1e-6)  # [N, H, D]

            # Map vs through v_map_mlp and ensure it is the correct shape
            vss = self.v_map_mlp(v)  # [N, H, D]
            attn_output = attn_output + vss  # preserve its rank, [N, H, D]

            if self.trans_heads_concat:
                attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
            else:
                attn_output = attn_output.mean(dim=1)

            attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
            attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

            if output_attn:
                return attn_output, attn_output
            else:
                return attn_output

    def forward(self, query_input, source_input, attention_mask=None, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            #print(f"Wq[{i}] output shape:", self.Wq[i](query_input).shape)  # 调试：打印每个头的 q
            #print(f"Wk[{i}] output shape:", self.Wk[i](source_input).shape)  # 调试：打印每个头的 k
            #print(f"Wv[{i}] output shape:", self.Wv[i](source_input).shape)  # 调试：打印每个头的 v
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)
        #print(f"🔍 Before stacking: q_list[0].shape = {q_list[0].shape}")

        query = torch.stack(q_list, dim=-2)  # [N, H, D]
        #print(f"🚀 After stacking query.shape = {query.shape}")  
        key = torch.stack(k_list, dim=-2)  # [N, H, D]
        value = torch.stack(v_list, dim=-2)  # [N, H, D]
        #print("Stacked query shape:", query.shape)
        #print("Stacked key shape:", key.shape)
        #print("Stacked value shape:", value.shape)

    # 修改这里，确保传递attention_mask
        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn)
            elif self.attention_type == 'full':
                # 添加attention_mask参数
                attention_output, attn = self.full_attention(
                    query, key, value, attention_mask=attention_mask, output_attn=output_attn) 
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value)
            elif self.attention_type == 'full':
                # 添加attention_mask参数
                attention_output = self.full_attention(
                    query, key, value, attention_mask=attention_mask)
            else:
                raise NotImplementedError


        final_output = attention_output
        # multi-head attention aggregation
        # final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
        super().__init__()
        #print(f"🔥 TransConv initialized with num_heads = {num_heads}")  # 🔥 打印 num_heads # 🔥 打印 num_heads
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels, num_heads=self.num_heads, use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input, attention_mask=None):
        layer_ = []
        
        # 原始代码保持不变
        x = self.fcs[0](x_input, x_manifold='euc')
        
        # 改进：根据position_ids进行位置编码
        if self.add_pos_enc:
            # 用提供的position_ids生成位置编码
                
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=-2))
   
        
        # 其他预处理保持不变
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)
        
        # 在每个卷积层中传递掩码
        for i, conv in enumerate(self.convs):
            #print(f"DEBUG - x shape: {x.shape}, attention_mask shape: {attention_mask.shape}")
            x = conv(x, x, attention_mask=attention_mask)
            #print(f"After conv {i}:", x.shape )#[32, 4, 33])
            if self.residual:
                #print(f"Before residual at layer {i}:", x.shape, layer_[i].shape) #([32, 4, 33]) ([32, 64, 33]) 下采样？？？？？
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=-2))
                #print(f"After residual at layer {i}:", x.shape) # 
            if self.use_bn:
                x = self.bns[i + 1](x)
            # if self.use_act:
            #     x = self.activation(x)
            # # x = self.dropout(x, training=self.training)
            layer_.append(x)
            #print(f"Layer {i} final output:", x.shape)

        x = self.fcs[-1](x)
        #print("Final FC output:", x.shape)
        return x


    '''
    def forward(self, x_input):
      layer_ = []
      print("x_input in TransConv",x_input.shape)
      x = self.fcs[0](x_input, x_manifold='euc')
      print("After first FC:", x.shape) # ([8, 256, 33])
      
      if self.add_pos_enc:
          x_pos = self.positional_encoding(x_input, x_manifold='euc')
          print("Positional encoding shape:", x_pos.shape)# ([8, 256, 33])
          x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=-2)) #TODO 把dim = 1 改为 dim = -2
          print("After adding pos encoding:", x.shape) # [8, 2, 33] ？？？？？？？？？？？？？？？fix 了 现在是[8, 256, 33]匹配了
      
      if self.use_bn:
          x = self.bns[0](x)
      if self.use_act:
          x = self.activation(x)
      x = self.dropout(x, training=self.training)
      layer_.append(x)
      print("After dropout:", x.shape)
      
      for i, conv in enumerate(self.convs):
          x = conv(x, x)
          print(f"After conv {i}:", x.shape )
          if layer_[i].shape[1] != x.shape[1]:
              repeat_factor = x.shape[1] // layer_[i].shape[1]
              layer_[i] = layer_[i].repeat(1, repeat_factor, 1)
          layer_.append(x)
          if self.residual:
              print(f"Before residual at layer {i}:", x.shape, layer_[i].shape)
              x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=-2))
              print(f"After residual at layer {i}:", x.shape)
          if self.use_bn:
              x = self.bns[i + 1](x)
          layer_.append(x)
          print(f"Layer {i} final output:", x.shape)
      
      x = self.fcs[-1](x)
      print("Final FC output:", x.shape)
      return x
      '''


    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class HypFormer(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,multimodal=False, text_vocab_size=None, image_vocab_size=None,
                 args=None):
        """
        This is a multimodal transformer model that uses hyperbolic space.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of hidden channels.
            out_channels (int): The number of output channels.
            trans_num_layers (int, optional): The number of layers in the TransConv module. Defaults to 1.
            trans_num_heads (int, optional): The number of attention heads in the TransConv module. Defaults to 1.
            trans_dropout (float, optional): The dropout rate in the TransConv module. Defaults to 0.5.
            trans_use_bn (bool, optional): Whether to use batch normalization in the TransConv module. Defaults to True.
            trans_use_residual (bool, optional): Whether to use residual connections in the TransConv module. Defaults to True.
            trans_use_weight (bool, optional): Whether to use learnable weights in the TransConv module. Defaults to True.
            trans_use_act (bool, optional): Whether to use activation functions in the TransConv module. Defaults to True.
            multimodal: 是否为多模态模式
            text_vocab_size: 文本词表大小（多模态模式下需要）
            image_vocab_size: 图像词表大小（多模态模式下需要）
            args (optional): Additional arguments.

        Raises:
            NotImplementedError: If the decoder_type is not 'euc' or 'hyp'.

        """
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        # self.manifold_hidden = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type
        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.multimodal = multimodal

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)

        if multimodal:
            if self.decoder_type == 'euc':
                self.text_decoder = nn.Linear(self.hidden_channels, text_vocab_size)
                self.image_decoder = nn.Linear(self.hidden_channels, image_vocab_size)
            elif self.decoder_type == 'hyp':
                self.text_decoder = HypCLS(self.manifold_out, self.hidden_channels, text_vocab_size)
                self.image_decoder = HypCLS(self.manifold_out, self.hidden_channels, image_vocab_size)
            else:
                raise NotImplementedError
        else:
            # 单模态模式保持原有解码头
            if self.decoder_type == 'euc':
                self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
                self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
            elif self.decoder_type == 'hyp':
                self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
                self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
            else:
                raise NotImplementedError

    def forward(self, x,attention_mask=None, token_types=None):
        """
        前向传播，支持多模态。
        
        Args:
            x: 输入特征
            attention_mask: 注意力掩码
            position_ids: 位置编码
            token_types: 模态类型(0=文本, 1=图像)，多模态模式下使用
            
        Returns:
            单模态模式: x
            多模态模式: (text_logits, img_logits, text_mask, img_mask)
        """

        # 共享编码器产生隐藏表示
        x1 = self.trans_conv(x, attention_mask=attention_mask)

        # 单模态处理逻辑保持不变
        if not self.multimodal or token_types is None:
            if self.decoder_type == 'euc':
                x = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
            elif self.decoder_type == 'hyp':
                x = self.decode_trans(x1)
            return x
        
        # 多模态处理逻辑
        text_mask = (token_types == 0)
        img_mask = (token_types == 1)
        
        # 为所有位置生成预测
        if self.decoder_type == 'hyp':
            # 应用文本解码器
            text_logits = self.text_decoder(x1)
            
            # 应用图像解码器到图像token位置
            img_logits = None
            if img_mask.any():
                img_indices = torch.where(img_mask)
                if len(img_indices[0]) > 0:
                    # 验证索引有效性
                    valid_indices = (img_indices[0] < x1.shape[0]) & (img_indices[1] < x1.shape[1])
                    valid_row_indices = img_indices[0][valid_indices]
                    valid_col_indices = img_indices[1][valid_indices]
                    
                    if len(valid_row_indices) > 0:
                        img_features = x1[valid_row_indices, valid_col_indices]
                        img_logits = self.image_decoder(img_features)
        
        else:  # 'euc'
            # 先转换到欧氏空间
            euc_features = self.manifold_out.logmap0(x1)[..., 1:]
            
            # 应用文本解码器
            text_logits = self.text_decoder(euc_features)
            
            # 应用图像解码器到图像token位置
            img_logits = None
            if img_mask.any():
                img_indices = torch.where(img_mask)
                if len(img_indices[0]) > 0:
                    # 验证索引有效性
                    valid_indices = (img_indices[0] < euc_features.shape[0]) & (img_indices[1] < euc_features.shape[1])
                    valid_row_indices = img_indices[0][valid_indices]
                    valid_col_indices = img_indices[1][valid_indices]
                    
                    if len(valid_row_indices) > 0:
                        img_features = euc_features[valid_row_indices, valid_col_indices]
                        img_logits = self.image_decoder(img_features)
        
        return text_logits, img_logits, text_mask, img_mask

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        if self.use_graph:
            self.graph_conv.reset_parameters()
