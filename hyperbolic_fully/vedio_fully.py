import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import UCF101
from PIL import Image as PILImage
import shutil

from utils.quantize.lookupFree import LFQ
from utils.hypformer_backup import HypFormer
from utils.improved_video_model import Encoder, Decoder
from utils.manifolds.hyp_layer import Optimizer
from huggingface_hub import hf_hub_download

# ===================== Argument Parser ======================
parser = argparse.ArgumentParser(description="Train HypFormer on a video dataset (using UCF101)")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--image_size', type=int, default=64, help='Resize frames to this size')
parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per video clip')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
# --data_dir 用于存放 UCF101 数据集的帧图片（假设组织结构符合 UCF101 要求）
parser.add_argument('--data_dir', type=str, default='./ucf101_frames', help='Path to UCF101 frames directory')
parser.add_argument('--annotation_path', type=str, default='./ucf101_annotations', help='Path to UCF101 annotation files')
parser.add_argument('--vocab_size', type=int, default=2**18, help='Vocabulary size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for HypFormer')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')

args = parser.parse_args()

# ============ Set Random Seed for Reproducibility ===========
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# ========= 定义视频预处理 ==========
# 注意：UCF101 数据集中的视频原始格式为 "TCHW"（默认），这里我们将其转换为 (C, T, H, W)
video_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # 如果 UCF101 返回的是 (T, C, H, W)，这里再转换为 (C, T, H, W)
    transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
])

# ========= 使用 torchvision.datasets.UCF101 构建数据集 ==========
# UCF101 的 API 参数包括：
#   - root：视频数据目录（假设目录下包含 train/、val/、test/ 子文件夹，每个类别为一个文件夹）
#   - annotation_path：标注文件所在目录（下载后解压标注文件）
#   - frames_per_clip：每个视频剪辑的帧数
#   - step_between_clips：剪辑间隔
#   - fold：选择哪个折（1-3）
#   - train：True 表示训练集，否则为测试集
#   - transform：这里传入 video_transform
ucf101_train_full = UCF101(
    root=os.path.join(args.data_dir, "train"),
    annotation_path=args.annotation_path,
    frames_per_clip=args.num_frames,
    step_between_clips=1,
    fold=1,
    train=True,
    transform=video_transform,
    output_format="TCHW"  # 输出格式为 TCHW
)
# 将训练集划分为训练集和校准集（例如 80% / 20%）
train_size = int(0.8 * len(ucf101_train_full))
cal_size = len(ucf101_train_full) - train_size
train_dataset, cal_dataset = random_split(ucf101_train_full, [train_size, cal_size])

test_dataset = UCF101(
    root=os.path.join(args.data_dir, "test"),
    annotation_path=args.annotation_path,
    frames_per_clip=args.num_frames,
    step_between_clips=1,
    fold=1,
    train=False,
    transform=video_transform,
    output_format="TCHW"
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Train set: {len(train_dataset)} clips, Cal set: {len(cal_dataset)} clips, Test set: {len(test_dataset)} clips")

# ========= 初始化视频编码器 ==========
# improved video model 的 Encoder 接收 5D 输入 (B, C, T, H, W)
ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 18,  # 输出特征的通道数
    "ch_mult": (1, 2, 2, 4),
    "resolution": args.image_size,
    "double_z": False,
}

video_encoder = Encoder(**ddconfig).to(device).eval()

# ========= 初始化 LFQ 模块 ==========
# LFQ 的 dim 参数设置为 18 以匹配视频编码器输出的 z_channels
lfq = LFQ(codebook_size=args.vocab_size, dim=18).to(device)
ckpt_path = hf_hub_download(repo_id="TencentARC/Open-MAGVIT2-Tokenizer-262144-Video", filename="video_128_262144.ckpt")
ckpt = torch.load(ckpt_path, map_location=device)
if "state_dict" in ckpt:
    lfq.load_state_dict(ckpt["state_dict"], strict=False)
elif "codebook" in ckpt:
    lfq.register_buffer("codebook", ckpt["codebook"])
lfq.eval()

# ========= 定义 HypFormer 模型 ==========
class ArgsH:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = "hyp"
        self.device = device
        self.add_positional_encoding = True
        self.attention_type = "full"
        self.power_k = 2
        self.trans_heads_concat = False

hyp_args = ArgsH()

model = HypFormer(
    in_channels=18,
    hidden_channels=args.hidden_channels,
    out_channels=args.vocab_size,
    trans_num_layers=args.num_layers,
    trans_num_heads=args.num_heads,
    trans_dropout=args.dropout,
    trans_use_bn=True,
    trans_use_residual=True,
    trans_use_weight=True,
    trans_use_act=True,
    args=hyp_args
).to(device)

# 为视频离散化准备一个嵌入层，embedding_dim 与视频编码器输出相匹配（18）
embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=18).to(device)

# ========= 训练循环 ==========
optimizer = Optimizer(model, args)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in train_loader:
        # UCF101 返回 (video, audio, label)，我们只取视频部分，视频形状为 (T, C, H, W)
        # 转换为 (C, T, H, W)（我们的视频编码器要求 5D 输入，需增加 batch 维度）
        videos, _, _ = batch
        # videos: [B, T, C, H, W] -> rearrange to [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4).to(device)

        with torch.no_grad():
            # Encoder 输出形状例如 (B, 18, T', H', W')
            features = video_encoder(videos)
            quantized, entropy_aux_loss, tokenized_x_raw = lfq(features)
            # 假设 tokenized_x_raw 的形状为 (B, T', H', W')，展平为 token 序列 (B, L)
            tokenized_x = tokenized_x_raw.view(videos.size(0), -1).long()
            tokenized_x_embed = embedding_layer(tokenized_x)

        output = model(tokenized_x_embed)  # 输出形状 (B, L, vocab_size)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size)
        # 简单采用后续 token 作为目标（序列右移1）
        target = tokenized_x[:, 1:1+seq_len].reshape(b * seq_len)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    train_perplexity = math.exp(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.4f}")

# ========= 校准（calibration）循环 ==========
model.eval()
cal_total_loss = 0
cal_total_tokens = 0
with torch.no_grad():
    for batch in cal_loader:
        videos, _, _ = batch
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        features = video_encoder(videos)
        _, _, tokenized_x = lfq(features)
        tokenized_x = tokenized_x.view(videos.size(0), -1).long()
        tokenized_x_embed = embedding_layer(tokenized_x)
        output = model(tokenized_x_embed)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size)
        target = tokenized_x[:, 1:1+seq_len].reshape(b * seq_len)
        loss = loss_fn(output, target)
        cal_total_loss += loss.item() * target.numel()
        cal_total_tokens += target.numel()

cal_avg_loss = cal_total_loss / cal_total_tokens
cal_perplexity = math.exp(cal_avg_loss)
print(f"Cal Perplexity: {cal_perplexity:.4f}")

# ========= 测试循环 ==========
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch in test_loader:
        videos, _, _ = batch
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        features = video_encoder(videos)
        _, _, tokenized_x = lfq(features)
        tokenized_x = tokenized_x.view(videos.size(0), -1).long()
        tokenized_x_embed = embedding_layer(tokenized_x)
        output = model(tokenized_x_embed)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size)
        target = tokenized_x[:, 1:1+seq_len].reshape(b * seq_len)
        loss = loss_fn(output, target)
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

test_avg_loss = total_loss / total_tokens
test_perplexity = math.exp(test_avg_loss)
print(f"Test Perplexity: {test_perplexity:.4f}")

