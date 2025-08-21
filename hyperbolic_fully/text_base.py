#### python
#filepath: /Users/nick/Documents/yale/科研/代码/hpyllama/hyperbolic_fully/text_fully_baseline.py

import os
import math
import random
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# ...existing code...
parser = argparse.ArgumentParser(description="Baseline Transformer on WikiText-2")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--image_size', type=int, default=64, help='Resize images to this size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training set')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels for HypFormer')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--vocab_size', type=int, default=2**14, help='Vocabulary size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hyp_lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Euclidean optimizer type')
parser.add_argument('--hyp_optimizer_type', type=str, default='radam', help='Hyperbolic optimizer type')
parser.add_argument('--weight_decay', type=float, default=0.005, help='Euclidean weight decay')
parser.add_argument('--hyp_weight_decay', type=float, default=1e-4, help='Hyperbolic weight decay')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# =========== Load Dataset and Tokenizer ===========
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
val_dataset   = tokenized_datasets["validation"]
test_dataset  = tokenized_datasets["test"]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")

# ========== Baseline Transformer Model ==========
def generate_square_subsequent_mask(seq_len):
    """
    生成 (seq_len, seq_len) 的下三角boolean矩阵：
    True 表示屏蔽，False 表示可见。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

class BaselineTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads,
            dim_feedforward=4 * embed_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x_embed):
        """
        x_embed: 形状 [batch_size, seq_len, embed_dim]
        这里为自回归添加mask，让token只看到它自己以及更靠前的位置
        """
        b, seq_len, _ = x_embed.size()
        # 生成下三角形mask
        causal_mask = generate_square_subsequent_mask(seq_len).to(x_embed.device)

        # 在PyTorch 2.0+里，可以直接mask=causal_mask; 某些老版本里是src_mask=...
        encoded = self.transformer_encoder(x_embed, mask=causal_mask)
        out = self.linear_out(encoded)
        return out

# =========== Build Model ===========
vocab_size = tokenizer.vocab_size
embed_dim = 128  # 与 tokenizer 时的 max_length 无关，这里是嵌入的维度
model = BaselineTransformer(
    embed_dim = embed_dim,
    n_heads   = 2,
    n_layers  = 2,
    vocab_size= vocab_size,
    dropout   = 0.1
).to(device)

embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

from tqdm import tqdm

# =========== Training Loop with tqdm progress bar ============
for epoch in range(args.epochs):
    start_time = time.time()

    model.train()
    total_loss = 0
    total_tokens = 0

    # 使用tqdm包装train_loader，显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        input_seq = input_ids[:, :-1]
        embeddings = embedding_layer(input_seq)

        output = model(embeddings)
        b, seq_len, v_size = output.shape
        output = output.view(b * seq_len, v_size)

        target = input_ids[:, 1:].reshape(-1)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

        # 进度条上更新当前平均损失
        if total_tokens > 0:
            current_loss = total_loss / total_tokens
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")

    avg_loss = total_loss / total_tokens
    train_perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

    # ============== Validation after each epoch =================
    model.eval()
    val_loss_sum = 0
    val_tokens_sum = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            input_seq = input_ids[:, :-1]
            embeddings = embedding_layer(input_seq)
            
            output = model(embeddings)
            b_v, seq_len_v, vsize_v = output.shape
            output = output.view(b_v * seq_len_v, vsize_v)
            target = input_ids[:, 1:].reshape(-1)

            loss_v = loss_fn(output, target)
            val_loss_sum += loss_v.item() * target.numel()
            val_tokens_sum += target.numel()

    val_avg_loss = val_loss_sum / val_tokens_sum
    val_perplexity = math.exp(val_avg_loss) if val_avg_loss < 10 else float("inf")

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
          f"Train Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.4f}, "
          f"Val PPL: {val_perplexity:.4f}")

# =========== Test Loop (Next Token Prediction) ==========
test_start = time.time()
model.eval()
test_loss_sum = 0
test_tokens_sum = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        input_seq = input_ids[:, :-1]
        
        embeddings = embedding_layer(input_seq)
        output = model(embeddings)

        b_t, seq_len_t, vsize_t = output.shape
        output = output.view(b_t * seq_len_t, vsize_t)
        target = input_ids[:, 1:].reshape(-1)

        loss_t = loss_fn(output, target)
        test_loss_sum += loss_t.item() * target.numel()
        test_tokens_sum += target.numel()

test_avg_loss = test_loss_sum / test_tokens_sum
test_perplexity = math.exp(test_avg_loss) if test_avg_loss < 10 else float("inf")
test_time = time.time() - test_start
print(f"Test completed in {test_time:.2f}s | Test PPL: {test_perplexity:.4f}")

# ...existing code continues if any...