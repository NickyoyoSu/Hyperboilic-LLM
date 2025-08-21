#### python
# filepath: /Users/nick/Documents/yale/科研/代码/hpyllama/hyperbolic_fully/image_fully_base.py

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image as PILImage

from utils.quantize.lookupFree import LFQ
from utils.hypformer_backup import HypFormer
from utils.improved_model import Encoder, Decoder
from utils.manifolds.hyp_layer import Optimizer
import shutil
from huggingface_hub import hf_hub_download

# ========== 替换掉 HypFormer、双曲优化器、manifolds等 ==========
from utils.improved_model import Encoder, Decoder
from utils.quantize.lookupFree import LFQ

# 标准PyTorch优化器
from torch.optim import Adam

# ===================== Argument Parser ======================
parser = argparse.ArgumentParser(description="Baseline: Transformer on CIFAR-like dataset")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--image_size', type=int, default=64, help='Resize images')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32, help='Transformer hidden dim')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Multi-head attention heads')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--vocab_size', type=int, default=2**18, help='Token vocabulary size')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate baseline')
parser.add_argument('--data_dir', type=str, default="/path/to/tiny-imagenet-200")
parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training set')
args = parser.parse_args()

# ============ Set random seed for reproducibility ===========
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# ========= Define CNN Encoder as LFQ input ==================
ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 18,  # 与embedding_dim对应
    "ch_mult": (1, 2, 2, 4),
    "resolution": args.image_size,
    "double_z": False,
}
cnn_encoder = Encoder(**ddconfig).to(device).eval()


# 根据自己的路径加载预训练好的codebook (如有)
lfq = LFQ(codebook_size=args.vocab_size, dim=18).to(device)
ckpt_path = hf_hub_download(repo_id="TencentARC/Open-MAGVIT2-Tokenizer-262144-Video", filename="video_128_262144.ckpt")
ckpt = torch.load(ckpt_path, map_location=device)
if "state_dict" in ckpt:
    lfq.load_state_dict(ckpt["state_dict"], strict=False)
elif "codebook" in ckpt:
    lfq.register_buffer("codebook", ckpt["codebook"])
lfq.eval()

# ========== Standard Transformer as baseline ================
class BaselineTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # 使用TransformerEncoder来模拟HypFormer逻辑
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=4*embed_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.linear_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_embed: torch.Tensor):
        # x_embed: (B, seq_len, embed_dim)
        # PyTorch nn.Transformer默认期望 (seq_len, B, embed_dim)，需要转置
        x_trans = x_embed.transpose(0, 1)  # -> (seq_len, B, embed_dim)
        # 通过transformer编码
        encoded = self.transformer_encoder(x_trans)  # (seq_len, B, embed_dim)
        # 转回 (B, seq_len, embed_dim)
        encoded = encoded.transpose(0, 1)
        # 最后映射到 vocab
        out = self.linear_out(encoded)  # (B, seq_len, vocab_size)
        return out

# ========== 实例化 BaselineTransformer ==========
model = BaselineTransformer(
    embed_dim=18,
    n_heads=args.num_heads,
    n_layers=args.num_layers,
    vocab_size=args.vocab_size,
    dropout=args.dropout
).to(device)

embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=18).to(device)
# ========= Load CIFAR-10 Dataset (torchvision) ==============
transform = transforms.Compose([
    transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),  # random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_size = int(args.train_ratio * len(cifar10_train))
val_size   = len(cifar10_train) - train_size
train_dataset, val_dataset = random_split(
    cifar10_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False)



# ========== Optimizer & Loss ==================
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# =========== Training & Validation ============
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in train_loader:
        images, _ = batch
        images = images.to(device)

        # 图像编码 + 量化
        with torch.no_grad():
            features = cnn_encoder(images)
            _, _, tokenized_x_raw = lfq(features)
            tokenized_x = tokenized_x_raw.view(images.size(0), -1).long()

            tokenized_x_embed = embedding_layer(tokenized_x)

        output = model(tokenized_x_embed)
        
        b, seq_len, vocab_size = output.shape
        output = output[:, :-1, :].reshape(b * (seq_len - 1), vocab_size)
        

        # shift 1 仅作演示
        target = tokenized_x[:, 1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    train_ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")

    # 验证
    model.eval()
    val_loss_sum = 0
    val_tokens_sum = 0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)
            features = cnn_encoder(images)
            _, _, tokenized_x = lfq(features)
            tokenized_x = tokenized_x.view(images.size(0), -1).long()

            tokenized_x_embed = embedding_layer(tokenized_x)
            out_val = model(tokenized_x_embed)
            b_v, seq_len_v, vsize_v = out_val.shape
            out_val = out_val [:, :-1, :].reshape(b * (seq_len - 1), vocab_size)

            tgt_val = tokenized_x[:, 1:].reshape(-1)
            loss_v = loss_fn(out_val, tgt_val)
            
            val_loss_sum += loss_v.item() * tgt_val.numel()
            val_tokens_sum += tgt_val.numel()

    val_avg_loss = val_loss_sum / val_tokens_sum
    val_ppl = math.exp(val_avg_loss) if val_avg_loss < 10 else float("inf")
    print(f"Epoch {epoch+1}: TrainLoss {avg_loss:.4f}, TrainPPL {train_ppl:.4f}, ValPPL {val_ppl:.4f}")

# =========== Test Loop =============
model.eval()
test_loss_sum = 0
test_tokens_sum = 0
with torch.no_grad():
    for batch in test_loader:
        images, _ = batch
        images = images.to(device)
        features = cnn_encoder(images)
        _, _, tokenized_x = lfq(features)
        tokenized_x = tokenized_x.view(images.size(0), -1).long()

        tokenized_x_embed = embedding_layer(tokenized_x)
        out_test = model(tokenized_x_embed)
        b_t, seq_len_t, vsize_t = out_test.shape
        out_test = out_test [:, :-1, :].reshape(b * (seq_len - 1), vocab_size)

        tgt_test = tokenized_x[:, 1:].reshape(-1)
        loss_t = loss_fn(out_test, tgt_test)

        test_loss_sum += loss_t.item() * tgt_test.numel()
        test_tokens_sum += tgt_test.numel()

test_avg_loss = test_loss_sum / test_tokens_sum
test_ppl = math.exp(test_avg_loss) if test_avg_loss < 10 else float("inf")
print(f"Test Perplexity: {test_ppl:.4f}")