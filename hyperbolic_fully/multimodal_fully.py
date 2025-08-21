#### python
# filepath: /Users/nick/Documents/yale/科研/代码/hpyllama/hyperbolic_fully/multimodal_fully.py

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

##############################################################################
# 引入：CNN Encoder / Decoder、HypFormer、双曲优化器、LFQ量化等
##############################################################################
from utils.improved_model import Encoder, Decoder       # 图像用CNN编码器 & 解码器
from utils.hypformer_backup import HypFormer            # 双曲空间的 Transformer
from utils.manifolds.hyp_layer import Optimizer         # 自定义双曲优化器
from utils.quantize.lookupFree import LFQ               # 基于 codebook 的量化

##############################################################################
# 引入：Hugging Face 用于文本处理
##############################################################################
from datasets import load_dataset
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    """
    简化版多模态数据集示例：既有图像，也有文本
    - image_folder: torchvision 的 ImageFolder 或类似
    - text_dataset: Hugging Face 文本数据集
    - tokenizer: 文本分词器
    - transform: 图像预处理
    """
    def __init__(self, image_folder, text_dataset, tokenizer, transform=None, max_len=64):
        super().__init__()
        self.image_folder = image_folder
        self.text_dataset = text_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return min(len(self.image_folder), len(self.text_dataset))

    def __getitem__(self, idx):
        # 从图像子集加载
        img, _ = self.image_folder[idx]
        if self.transform:
            img = self.transform(img)

        # 从文本子集加载
        text_data = self.text_dataset[idx]["text"]

        # 这里不直接做 tokenizer，让 DataLoader collate_fn 或外部做
        return img, text_data


class MultiModalModel(nn.Module):
    """
    包含文本 & 图像两条分支:
    1) 图像分支: CNN Encoder + LFQ量化 -> tokenized_x -> embedding -> HypFormer
    2) 文本分支: tokenizer output (input_ids) -> embedding -> HypFormer
    3) 最终可将 HypFormer 输出传给 CNN Decoder 生成图像，实现“文本→图像”。
    """
    def __init__(self, args, text_vocab_size):
        super().__init__()
        self.device = torch.device(args.device)

        # ------------------ 图像分支：编码器 + 量化 ------------------
        self.cnn_encoder = Encoder(
            ch=64, out_ch=3, in_channels=3, num_res_blocks=2,
            z_channels=18, ch_mult=(1, 2, 2, 4),
            resolution=args.image_size, double_z=False
        )
        self.lfq = LFQ(codebook_size=args.vocab_size, dim=16)

        # ------------------ 文本分支：Embedding ------------------
        # 必须保证文本embedding维度 == 图像encoder输出的z_channels
        self.text_embedding_dim = 18
        self.text_embedding = nn.Embedding(num_embeddings=text_vocab_size, embedding_dim=self.text_embedding_dim)

        # ------------------ HypFormer：主干模型 ------------------
        # 用于多模态特征在双曲空间中做 Transformer-style 的处理
        self.hypformer = HypFormer(
            in_channels=self.text_embedding_dim,      
            hidden_channels=args.hidden_channels,
            out_channels=args.vocab_size,
            trans_num_layers=args.num_layers,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            args=None
        )

        # ------------------ 图像解码器：文本生成图 ------------------
        # 将 HypFormer 输出(或某种聚合后表示)解码回图像
        self.img_decoder = Decoder(
            ch=64, out_ch=3, in_channels=18,   # 和 hypformer 输出维度对应
            num_res_blocks=2, resolution=args.image_size
        )

    @torch.no_grad()
    def forward_image_branch(self, images: torch.Tensor):
        """
        图像 -> Encoder -> LFQ -> tokenize -> text_embedding -> HypFormer
        返回： (batch_size, seq_len, vocab_size), 以及 tokenized_x
        """
        features = self.cnn_encoder(images)                      # (b, 18, H, W)
        _, _, tokenized_x_raw = self.lfq(features)               # token化
        tokenized_x = tokenized_x_raw.view(images.size(0), -1).long()
        # 嵌入
        img_embed = self.text_embedding(tokenized_x)             # (b, seq_len, 18)
        # 输入 HypFormer
        output = self.hypformer(img_embed)                       # (b, seq_len, vocab_size)
        return output, tokenized_x

    def forward_text_branch(self, input_ids: torch.Tensor):
        """
        文本 -> embedding -> HypFormer
        返回：(batch_size, seq_len, vocab_size)
        """
        txt_embed = self.text_embedding(input_ids)               # (b, seq_len, 18)
        output = self.hypformer(txt_embed)                       # (b, seq_len, vocab_size)
        return output

    def generate_image_from_text(self, input_ids: torch.Tensor):
        """
        文本 -> HypFormer -> reshape -> Decoder -> 生成图像
        """
        # 1) 文本序列嵌入
        txt_embed = self.text_embedding(input_ids)               # (b, seq_len, 18)
        # 2) HypFormer 得到序列输出
        seq_out = self.hypformer(txt_embed)                      # (b, seq_len, vocab_size)
        # --- 此处仅演示，用 seq_out 的最后一个 time step 做全局表示 ---
        # 注：真实应用中可做注意力池化或拼接
        global_repr = seq_out.mean(dim=1)                        # (b, vocab_size)
        
        # 3) 将 global_repr 投影到 18 维，然后 reshape -> (b, 18, H, W)
        #   这里只做简化处理：假设 vocab_size 与 18 不同，需要手工做一层线性变换
        hidden_dim = 18
        linear_map = nn.Linear(seq_out.shape[-1], hidden_dim).to(self.device)
        global_repr_18 = linear_map(global_repr)                 # (b, 18)
        # reshape
        b_size = global_repr_18.shape[0]
        # 假设最后解码器需要 (b, 18, 8, 8) 之类
        global_repr_4d = global_repr_18.view(b_size, hidden_dim, 8, 8)

        # 4) Decoder 解码
        generated_img = self.img_decoder(global_repr_4d)         # (b, 3, H, W)
        return generated_img


def collate_fn(batch_list):
    """
    用于 DataLoader 的 collate_fn 演示:  
    - 将图像打包成一个tensor
    - 将文本放在列表中
    """
    imgs, texts = zip(*batch_list)
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(texts)


def main():
    parser = argparse.ArgumentParser(description="MultiModal HypFormer at extremes!")
    # 训练相关参数
    parser.add_argument('--data_dir', type=str, default="/path/to/tiny-imagenet-200")
    parser.add_argument('--text_dataset_name', type=str, default="wikitext")
    parser.add_argument('--text_dataset_config', type=str, default="wikitext-2-raw-v1")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=2**18)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # -------------------- 图像数据集 --------------------
    from torchvision import transforms, datasets
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "train"), transform=transform)

    # -------------------- 文本数据集 --------------------
    raw_datasets = load_dataset(args.text_dataset_name, args.text_dataset_config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 保证 text_dataset 与 image_dataset 数量相当
    text_train_data = raw_datasets["train"].select(range(len(image_dataset)))

    # -------------------- 多模态数据集 & Dataloader --------------------
    multimodal_data = MultiModalDataset(
        image_folder=image_dataset,
        text_dataset=text_train_data,
        tokenizer=tokenizer,
        transform=None
    )
    train_loader = DataLoader(multimodal_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # -------------------- 模型初始化 --------------------
    model = MultiModalModel(args, tokenizer.vocab_size).to(device)
    optimizer = Optimizer(model.hypformer, args)
    loss_fn = nn.CrossEntropyLoss()

    # -------------------- 训练示例 --------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch_img, batch_text in train_loader:
            batch_img = batch_img.to(device)
            # 对文本做Tokenizer
            tokenized_text = tokenizer(batch_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(device)  # (b, seq_len)

            # 前向：图像分支
            with torch.no_grad():
                img_output, img_tokens = model.forward_image_branch(batch_img)
            # 前向：文本分支
            txt_output = model.forward_text_branch(input_ids)

            # 以图像分支为例来做loss
            b, seq_len, v_size = img_output.shape
            img_output_flat = img_output.view(b * seq_len, v_size)
            # shift 1 仅为示例
            target = img_tokens[:, 1:1+seq_len].reshape(b * seq_len)

            loss = loss_fn(img_output_flat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    print("多模态训练结束！")

    # -------------------- 文本→图像推理（示例） --------------------
    model.eval()
    sample_text = ["A red cat jumping on the moon"]  # 输入任意英文描述
    with torch.no_grad():
        tokenized_text = tokenizer(sample_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        gen_img = model.generate_image_from_text(input_ids)
        print("生成图片张量形状：", gen_img.shape)
        # 您可将结果转换为 PIL 或保存到文件中进行可视化

if __name__ == "__main__":
    main()