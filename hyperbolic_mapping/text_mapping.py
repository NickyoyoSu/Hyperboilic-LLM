import sys
sys.path.append("/home/zy353/project/Hyperbolic-Multimodal")

import os
import time
import math
import argparse
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model  # LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def load_text_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_parser():
    parser = argparse.ArgumentParser(description="Train a language model with Hyperbolic or Euclidean mapping")
    # 超曲映射开关
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    # LoRA 开关，默认 False，不使用 LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--min_tokens", type=int, default=5, help="Minimum tokens required per line")
    parser.add_argument("--max_samples", type=int, default=150000, help="Maximum entity numbers for the dataset to load")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    return parser

args = get_parser().parse_args()

USE_HYPERBOLIC = args.use_hyperbolic
USE_LORA = args.use_lora
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BASE_LR = args.learning_rate
MAX_LENGTH = args.max_length
MIN_TOKENS = args.min_tokens
DEVICE = args.device
MAX_SAMPLES = args.max_samples

# ===================== Load LLaMA Model =====================
hf_token = "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"
model_name = "meta-llama/Llama-3.2-1B"

device = torch.device(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)
tokenizer.pad_token = tokenizer.eos_token

# ===== Apply LoRA Adaptation or Freeze Original Model =====
if USE_LORA:
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
else:
    # 冻结原模型的参数，只训练后续添加的层
    for param in model.parameters():
        param.requires_grad = False

# ============== Custom LM Head Class with Normalization =================
class MappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.vocab_size = base_model.config.vocab_size
        self.hidden_size = base_model.config.hidden_size
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # 第1层线性变换及归一化
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # 第二层（可选）
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # 超曲分类器：num_features 为 hidden_size+1（多出的1表示 time 分量）
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # 欧氏分类器
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        x = self.linear1(last_hidden_states)
        x = self.norm1(x)
        
        if self.num_layers == 2:
            x = torch.relu(x)
            x = self.linear2(x)
            x = self.norm2(x)
        
        if self.use_hyperbolic:
            # 添加 time 分量，再进行超曲映射
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# ====================== Load WikiText-103 Dataset ======================
wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
wiki_texts = [s["text"] for s in wiki_dataset["train"] if len(s["text"].split()) > 5][:MAX_SAMPLES]

train_texts, test_texts = train_test_split(wiki_texts, test_size=0.2, random_state=42)
train_texts, val_texts = train_test_split(train_texts, test_size=0.1, random_state=42)
print(f"Final dataset sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# =========== Instantiate Custom Head ===========
custom_lm_head = MappingHead(model, use_hyperbolic=USE_HYPERBOLIC).to(device)
learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))

all_params = list(model.parameters()) + list(custom_lm_head.parameters()) + [learnable_curvature]

# ========= Build Optimizer and Scheduler ===========
# 为 learnable_curvature 单独设定较小的 lr
curvature_lr = 1e-6
optimizer = torch.optim.AdamW([
    {"params": list(custom_lm_head.parameters()), "lr": BASE_LR},
    {"params": [learnable_curvature], "lr": curvature_lr}
] + ( [{"params": list(model.parameters()), "lr": BASE_LR}] if USE_LORA else []) )

# 计算训练步数（每个 epoch 的步数 * epoch 数）
num_training_steps = (len(train_texts) // BATCH_SIZE) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)  # warmup 10%
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_loss = float("inf")

# ========== Logging Helper ==========
def log_metrics(epoch, avg_loss, ppl, elapsed_time):
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    # 打印日志信息
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, LR={current_lr:.2e}, Time={elapsed_time:.2f}s, Curvature={learnable_curvature.item():.4f}")

# ====================== Save Model ======================
def save_model(epoch, loss, ppl):
    global best_loss
    model_type = "euc" if not USE_HYPERBOLIC else "hyp"
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = f"/home/zy353/project/Hyperbolic-Multimodal/hyperbolic_mapping/save_models/{timestamp}"

    os.makedirs(save_dir, exist_ok=True)
    if loss < best_loss:
        best_loss = loss
        filename = f"{model_type}_model_epoch_{epoch}_loss_{loss:.4f}_PPL_{ppl:.2f}.pth"
        save_path = os.path.join(save_dir, filename)
        torch.save({
            "model_state": model.state_dict(),
            "lm_head_state": custom_lm_head.state_dict(),
            "curvature": learnable_curvature.item(),
            "optimizer_state": optimizer.state_dict()
        }, save_path)
        print(f"Model checkpoint saved: {filename}")

# ==================== Data Loader Helpers ====================
def get_batches(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]

def prepare_labels(inputs):
    labels = inputs["input_ids"].clone()
    labels[..., :-1] = labels[..., 1:].clone()
    labels[..., -1] = -100
    return labels

def compute_lm_loss(logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

# ====================== Train Model ======================
def train_model():
    global best_loss
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0

        for batch_texts in tqdm(get_batches(train_texts, BATCH_SIZE), total=len(train_texts)//BATCH_SIZE, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                logits = custom_lm_head(outputs.hidden_states[-1], learnable_curvature)
                loss = compute_lm_loss(logits, labels)
            loss.backward()
            clip_grad_norm_(list(model.parameters()) + list(custom_lm_head.parameters()) + [learnable_curvature], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                learnable_curvature.clamp_(1e-3, 1e1)
            total_loss += loss.item() * len(batch_texts)
            count += len(batch_texts)

        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        evaluate_model(val_texts, phase="Validation")
        save_model(epoch, avg_loss, ppl)

# ====================== Evaluate Function ======================
def evaluate_model(texts, phase="Val"):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch_texts in get_batches(texts, BATCH_SIZE):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            logits = custom_lm_head(last_hidden_states=hidden_states, c_param=learnable_curvature)
            loss_val = compute_lm_loss(logits, labels)
            total_loss += loss_val.item() * len(batch_texts)
            count += len(batch_texts)
    avg_loss = total_loss / count if count > 0 else 9999.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print(f"{phase} - loss={avg_loss:.4f}, PPL={ppl:.2f}")
    return avg_loss

# ====================== Inference (Generation) ======================
def generate_text(prompt, max_len=100):
    model.eval()
    custom_lm_head.eval()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ====================== Main Execution ======================
if __name__ == "__main__":
    print(f"USE_HYPERBOLIC = {USE_HYPERBOLIC}")
    print(f"USE_LORA = {USE_LORA}")
    train_model()
    print("\n=== Testing ===")
    evaluate_model(test_texts, phase="Test")
    sample_out = generate_text("Donald Trump is the first president to")
    print("\nSample Generation:\n", sample_out)
