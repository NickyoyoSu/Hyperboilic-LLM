import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from utils.hypformer_backup import HypFormer
from utils.manifolds.hyp_layer import Optimizer
import shutil

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser(description="Train HypFormer on Audio Next-Token Prediction")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--audio_subset', type=str, default="train[:1%]", help='Subset of Common Voice to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for HypFormer')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--vocab_size', type=int, default=2**18, help='Vocabulary size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hyp_lr', type=float, default=0.001, help='Hyperbolic learning rate')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Euclidean optimizer type')
parser.add_argument('--hyp_optimizer_type', type=str, default='radam', help='Hyperbolic optimizer type')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Euclidean weight decay')
parser.add_argument('--hyp_weight_decay', type=float, default=0.0, help='Hyperbolic weight decay')
args = parser.parse_args()

# ----------------- Set Random Seed -----------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
device = torch.device(args.device)

# ----------------- Load Pretrained Audio Tokenizer -----------------
# We use Whisper's processor and model for tokenization.
processor = AutoProcessor.from_pretrained("openai/whisper-small")
tokenizer_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to(device)
tokenizer_model.eval()

def audio_to_tokens(sample):
    # sample['audio'] is expected to be a dict with keys "array" and "sampling_rate"
    waveform = torch.tensor(sample["audio"]["array"], dtype=torch.float32).unsqueeze(0).to(device)
    sr = sample["audio"]["sampling_rate"]
    # Resample if necessary (Whisper expects 16kHz)
    if sr != 16000:
        from torchaudio.transforms import Resample
        waveform = Resample(orig_freq=sr, new_freq=16000).to(device)(waveform)
    waveform_np = waveform.squeeze(0).cpu().numpy()
    inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        token_ids = tokenizer_model.generate(**inputs)
    tokens = token_ids.flatten().cpu().numpy().tolist()
    return tokens

# ----------------- Create Audio Dataset -----------------
class AudioTokenDataset(Dataset):
    def __init__(self, split, subset, max_length=512):
        # Load a small subset (e.g., 1%) of Common Voice English
        self.dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split=subset, num_proc=1)
        self.max_length = max_length  # maximum token sequence length (for truncation/padding)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        tokens = audio_to_tokens(sample)
        # Optionally truncate or pad tokens to fixed length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

# Use the same dataset for train/val/test splits for this prototype
full_audio_dataset = AudioTokenDataset(split="train", subset=args.audio_subset, max_length=256)
train_size = int(0.9 * len(full_audio_dataset))
val_size = len(full_audio_dataset) - train_size
train_dataset, val_dataset = random_split(full_audio_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
test_dataset = val_dataset  # for simplicity, use val as test too

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# ----------------- Initialize HypFormer Model -----------------
# (For audio, our tokens are already generated by the pre-trained tokenizer.)
# We use a simple embedding layer to convert token IDs to vectors.
embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=16).to(device)

# HypFormer expects input of shape (batch, token, embedding_dim)
# We assume our tokens are integers and will be embedded.
class HypArgs:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = "hyp"
        self.device = device
        self.add_positional_encoding = True
        self.attention_type = "full"
        self.power_k = 2
        self.trans_heads_concat = False

hyp_args = HypArgs()

model = HypFormer(
    in_channels=18,  # here, the embedding dimension is 16 (you can adjust)
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

# ----------------- Initialize Optimizer -----------------
optimizer = Optimizer(model, args)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------- Training Loop -----------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in train_loader:
        # Each batch is a tensor of shape (batch_size, seq_len)
        tokens = batch.to(device)  # tokens are already our discrete sequence
        # Get embedded tokens: shape (batch_size, seq_len, embedding_dim)
        token_embed = embedding_layer(tokens)

        # For next-token prediction, use the sequence as input and target shifted by one.
        # For simplicity, let’s assume our model takes the whole sequence.
        with torch.no_grad():
            output = model(token_embed)  # output shape (batch_size, seq_len, vocab_size)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size)
        # Target is the tokens shifted by one (ignoring the first token)
        target = tokens[:, 1:].reshape(-1)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    train_perplexity = math.exp(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.4f}")

    # Validation loop
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            tokens = batch.to(device)
            token_embed = embedding_layer(tokens)
            output = model(token_embed)
            b, seq_len, vocab_size = output.shape
            output = output.view(b * seq_len, vocab_size)
            target = tokens[:, 1:].reshape(-1)
            loss = loss_fn(output, target)
            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

    val_avg_loss = total_loss / total_tokens
    val_perplexity = math.exp(val_avg_loss)
    print(f"Epoch {epoch + 1}, Val PPL: {val_perplexity:.4f}")

# ----------------- Test Loop -----------------
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch in test_loader:
        tokens = batch.to(device)
        token_embed = embedding_layer(tokens)
        output = model(token_embed)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size)
        target = tokens[:, 1:].reshape(-1)
        loss = loss_fn(output, target)
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

test_avg_loss = total_loss / total_tokens
test_perplexity = math.exp(test_avg_loss)
print(f"Test Perplexity: {test_perplexity:.4f}")