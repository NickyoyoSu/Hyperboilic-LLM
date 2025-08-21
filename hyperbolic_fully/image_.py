import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image as PILImage

from utils.quantize.lookupFree import LFQ
from utils.hypformer import HypFormer
from utils.improved_model import Encoder, Decoder

# ========================
# **0️⃣ Set Random Seed for Reproducibility**
# ========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ========================
# **1️⃣ Define Encoder as LFQ Input (from improved_model)**
# ========================


ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 12,
    "ch_mult": (1, 2, 2, 4),
    "resolution": 64,  # 如果想保留32×32原尺寸，可改成32并相应调整transforms
    "double_z": False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = Encoder(**ddconfig).to(device).eval()

# ========================
# **2️⃣ Load CIFAR-10 Dataset (torchvision)**
# ========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整到 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# 手动拆分训练集，留一部分做验证集 (例如90% train, 10% val)
train_size = int(0.9 * len(cifar10_train))
val_size   = len(cifar10_train) - train_size
train_dataset, val_dataset = random_split(cifar10_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(cifar10_test)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(cifar10_test, batch_size=32, shuffle=False)

# ========================
# **3️⃣ Initialize LFQ and Embedding Layer**
# ========================

lfq = LFQ(codebook_size=2**12, dim=12).to(device).eval()

# 注意：这里将 embedding_dim 从原来的 19 改为 20，以保证能被注意力头 (4) 整除
embedding_layer = nn.Embedding(num_embeddings=2**12, embedding_dim=20).to(device)

# ========================
# **4️⃣ Define SimpleTransformer (Replace HypFormer) with Causal Mask and Positional Encoding**
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, in_channels=20, out_channels=2**12, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_len=512):
        super(SimpleTransformer, self).__init__()
        self.d_model = in_channels
        self.pos_encoder = PositionalEncoding(in_channels, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(in_channels, out_channels)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

    def forward(self, x):
        # x: (B, SeqLen, d_model)
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer_encoder(x, mask=tgt_mask)
        out = self.fc_out(out)
        return out  # (B, SeqLen, out_channels)

model = SimpleTransformer(
    in_channels=20,     # 输入维度必须与 embedding_layer 输出一致
    out_channels=2**12,
    nhead=4,
    num_layers=6,
    dim_feedforward=128,
    dropout=0.1,
    max_len=512
).to(device)

# ========================
# **5️⃣ Training Loop (Next Token Prediction) with Perplexity**
# ========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):  # 训练10个epoch
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in train_loader:
        images, _ = batch  # CIFAR-10 的标签暂不使用
        images = images.to(device)  # (B, 3, 64, 64)

        with torch.no_grad():
            features = cnn_encoder(images)         # (B, 18, H, W)
            _, _, tokenized_x = lfq(features)        # tokenized_x: (B, 18, H, W)
            # 展平为 (B, seq_len)
            tokenized_x = tokenized_x.view(images.size(0), -1).long()
            # 嵌入：输出 (B, seq_len, 20)
            tokenized_x_embed = embedding_layer(tokenized_x)

        # 模型前向输出，形状 (B, L_out, vocab_size)
        output = model(tokenized_x_embed)
        b, L_out, vocab_size = output.shape
        # 目标序列长度不能超过 tokenized_x 的长度-1
        L_target = min(L_out, tokenized_x.shape[1] - 1)
        output = output[:, :L_target, :].reshape(b * L_target, vocab_size)
        target = tokenized_x[:, 1:1+L_target].reshape(b * L_target)
        
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    train_perplexity = math.exp(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Perplexity: {train_perplexity:.4f}")

# ========================
    # **6️⃣ Validation Loop (Next Token Prediction)**
    # ========================
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)

            features = cnn_encoder(images)
            _, _, tokenized_x = lfq(features)
            tokenized_x = tokenized_x.view(images.size(0), -1).long()
            tokenized_x_embed = embedding_layer(tokenized_x)
            output = model(tokenized_x_embed)
            b, L_out, vocab_size = output.shape
            L_target = min(L_out, tokenized_x.shape[1] - 1)
            output = output[:, :L_target, :].reshape(b * L_target, vocab_size)
            target = tokenized_x[:, 1:1+L_target].reshape(b * L_target)
            
            loss = loss_fn(output, target)
            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

    val_avg_loss = total_loss / total_tokens
    val_perplexity = math.exp(val_avg_loss)
    print(f"Validation Perplexity: {val_perplexity:.4f}")

# ========================
# **7️⃣ Test Loop (Next Token Prediction) [Optional]**
# ========================
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch in test_loader:
        images, _ = batch
        images = images.to(device)

        features = cnn_encoder(images)
        _, _, tokenized_x = lfq(features)
        tokenized_x = tokenized_x.view(images.size(0), -1).long()
        tokenized_x_embed = embedding_layer(tokenized_x)
        output = model(tokenized_x_embed)
        b, L_out, vocab_size = output.shape
        L_target = min(L_out, tokenized_x.shape[1] - 1)
        output = output[:, :L_target, :].reshape(b * L_target, vocab_size)
        target = tokenized_x[:, 1:1+L_target].reshape(b * L_target)
        loss = loss_fn(output, target)
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

test_avg_loss = total_loss / total_tokens
test_perplexity = math.exp(test_avg_loss)
print(f"Test Perplexity: {test_perplexity:.4f}")
