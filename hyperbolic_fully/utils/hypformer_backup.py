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


    def full_attention(self, qs, ks, vs, output_attn=False):
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)
        def causal_mask(seq_len, device):
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * (-1e9), diagonal=1)
            return mask

        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))
        #print("Initial att_weight shape:", att_weight.shape)  # 例如 [H, N, N]
        att_weight = att_weight / self.scale + self.bias# [H, N, N]
        seq_len = att_weight.shape[-1]
        mask = causal_mask(seq_len, att_weight.device)
        if mask.shape != att_weight.shape:
            mask = mask.unsqueeze(0).expand(att_weight.shape)
        #print("Mask shape:", mask.shape)
        att_weight = att_weight + mask
        att_weight = nn.Softmax(dim=-1)(att_weight)
        #print("After softmax, att_weight shape:", att_weight.shape)# [H, N, N]
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight) # [N, H, D]
        att_output = att_output.transpose(0, 1)# [N, H, D]

        att_output = self.manifold.mid_point(att_output)
        #print("Final attention output shape in full_attention:", att_output.shape)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

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

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
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

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(
                    query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value)  # [N, H, D]
            elif self.attention_type == 'full':
                attention_output = self.full_attention(
                    query, key, value)
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

    def forward(self, x_input):
      layer_ = []
      #print("x_input in TransConv",x_input.shape)#([32, 64, 18])
      # the original inputs are in Euclidean 
      x = self.fcs[0](x_input, x_manifold='euc')
      #print("After first FC:", x.shape) #([32, 64, 33])
      # add positional embedding 
      if self.add_pos_enc:
          x_pos = self.positional_encoding(x_input, x_manifold='euc') 
          #print("Positional encoding shape:", x_pos.shape) #([32, 64, 33])
          x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=-2))
          #print("After adding pos encoding:", x.shape) #([32, 64, 33])

      if self.use_bn:
          x = self.bns[0](x)
      if self.use_act:
          x = self.activation(x)
      x = self.dropout(x, training=self.training)
      layer_.append(x)
      #print("After dropout:", x.shape)#([32, 64, 33])

      for i, conv in enumerate(self.convs):
          x = conv(x, x)
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
                 trans_use_weight=True, trans_use_act=True,
                 args=None):
        """
        Initializes a HypFormer object.

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

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)

        if self.decoder_type == 'euc':
            self.decode_trans = nn.Linear(self.hidden_channels, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        
        elif self.decoder_type == 'hyp':
            self.decode_graph = HypLinear(self.manifold_out, self.hidden_channels, self.hidden_channels)
            self.decode_trans = HypCLS(self.manifold_out, self.hidden_channels, self.out_channels)
        
        else:
            raise NotImplementedError

    def forward(self, x):
        x1 = self.trans_conv(x)
        #print ("xi shapge afer trans_conv: ", x1.shape)
        ### SEE HERE FORM IMAGE_FULLY!!!!
        if self.decoder_type == 'euc':
            x = self.decode_trans(self.manifold_out.logmap0(x1)[..., 1:])
        
        elif self.decoder_type == 'hyp':
            x = self.decode_trans(x1) 
        else:
            raise NotImplementedError
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        if self.use_graph:
            self.graph_conv.reset_parameters()
