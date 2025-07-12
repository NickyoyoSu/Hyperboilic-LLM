from tqdm import tqdm
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch
import torch.nn as nn
import time
import os
from torch.optim import SGD, AdamW
from hypercore.nn import LoraModel, LoraConfig
from hypercore.models import LViT
import hypercore.nn as hnn
from hypercore.manifolds import Lorentz
import gc
import random
import logging

gc.collect()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Geodesic regularization
def geodesic_regularization(outputs, labels, manifold, lambda_reg=1e-4):
    dist_matrix = manifold.lorentzian_distance(outputs.unsqueeze(1), outputs.unsqueeze(0)).clamp_min(1e-6).sqrt()
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    reg_loss = ((1 - label_matrix) * dist_matrix).mean() 
    return lambda_reg * reg_loss

# Accuracy function
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Load pre-trained model checkpoint
def load_model(output_dir, model, epoch, device):
    model_checkpoint = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# LoRA Config
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.8,
    target_modules=["Wq", "Wk", "Wv", 'dense_1.linear', 'dense_2.linear'],
    lora_type="std",
    merge_weights=False,
    modules_to_save=["classifier"]
)

# Define manifolds
manifold_in = Lorentz(1.0)
manifold_hidden = Lorentz(1.0)
manifold_out = Lorentz(1.0)

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LViT(manifold_in, manifold_hidden, manifold_out)

# Load model checkpoint if exists
output_dir = '...'
checkpoint_epoch = 200
model_checkpoint = os.path.join(output_dir, f"checkpoint_epoch_{checkpoint_epoch}.pth")

if os.path.isfile(model_checkpoint):
    model = load_model(
        output_dir, model, checkpoint_epoch, device
    )
    logger.info("Model loaded successfully")
# Modify classifier
model.classifier = hnn.LorentzMLR(model.manifold_out, model.num_heads * model.hidden_channel, 200)
model = model.to(device)

# Wrap model in LoRA
lora_model = LoraModel(lora_config, model)
lora_model.train()

# Dataset & DataLoader
root_dir = "..."
train_dir = root_dir + "train/images"
test_dir = root_dir + "val/images"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(train_dir, train_transform)
test_set = datasets.ImageFolder(test_dir, test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=8, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=8, pin_memory=True, shuffle=False)

# Loss & Optimizer
criterion = torch.nn.CrossEntropyLoss()
base_lr = 1e-3
optimizer = AdamW(lora_model.parameters(), lr=base_lr, weight_decay=0.1)

# Training with Steps instead of Epochs
total_steps = 31250  # Total fine-tuning steps
warmup_steps = 500  # Warmup for 10% of total steps
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

# Training Loop (Step-based)
step = 0
while step < total_steps:
    lora_model.train()
    losses, acc1, acc5 = [], [], []
    t0 = time.time()

    for x, y in tqdm(train_loader, total=len(train_loader)):
        if step >= total_steps:
            break  # Stop training when total steps are reached
        
        x, y = x.to(device), y.to(device)
        logits = lora_model(x)
        loss = criterion(logits, y) # + geodesic_regularization(logits, y, lora_model.manifold_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Warmup phase
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale
        else:
            scheduler.step()

        with torch.no_grad():
            top1, top5 = accuracy(logits, y, topk=(1, 5))
            losses.append(loss.item())
            acc1.append(top1.item())
            acc5.append(top5.item())

        # Logging & Checkpointing
        if (step) % 1000 == 0:
            logger.info(f"Step {step}/{total_steps}: Loss={np.mean(losses):.4f}, Acc@1={np.mean(acc1):.4f}, Acc@5={np.mean(acc5):.4f}")
        if (step) % 3000 == 0:
            lora_model.eval()
            losses, acc1, acc5 = [], [], []

            with torch.no_grad():
                for x, y in tqdm(test_loader, total=len(test_loader)):
                    x, y = x.to(device), y.to(device)
                    logits = lora_model(x)
                    loss = criterion(logits, y)

                    top1, top5 = accuracy(logits, y, topk=(1, 5))
                    losses.append(loss.item())
                    acc1.append(top1.item())
                    acc5.append(top5.item())
            logger.info(f"Eval Results: Loss={np.mean(losses):.4f}, Acc@1={np.mean(acc1):.4f}, Acc@5={np.mean(acc5):.4f}")
        step += 1

# Final Evaluation
print("Final Evaluation...")
lora_model.eval()
losses, acc1, acc5 = [], [], []

with torch.no_grad():
    for x, y in tqdm(test_loader, total=len(test_loader)):
        x, y = x.to(device), y.to(device)
        logits = lora_model(x)
        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.append(loss.item())
        acc1.append(top1.item())
        acc5.append(top5.item())

print(f"Final Results: Loss={np.mean(losses):.4f}, Acc@1={np.mean(acc1):.4f}, Acc@5={np.mean(acc5):.4f}")