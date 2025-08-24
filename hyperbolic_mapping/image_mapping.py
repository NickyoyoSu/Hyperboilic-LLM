import sys
import os
import time
import math
import argparse
import traceback
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader, Dataset, random_split
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, MllamaForConditionalGeneration, get_linear_schedule_with_warmup
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from torchvision import transforms

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def get_parser():
    parser = argparse.ArgumentParser(description="Train a multimodal vision-language model with Hyperbolic mapping")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for the dataset to load")
    parser.add_argument("--image_size", type=int, default=224, help="Size of input images (448 for MLLama)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use (e.g., 'cuda', 'cuda:1', 'cpu')")
    parser.add_argument("--gpu_ids", type=str, default="0", 
                    help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--use_parallel", action="store_true",
                    help="Use DataParallel for multi-GPU training")
    return parser

args = get_parser().parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

USE_HYPERBOLIC = args.use_hyperbolic
USE_LORA = args.use_lora
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BASE_LR = args.learning_rate
MAX_LENGTH = args.max_length
DEVICE = args.device
MAX_SAMPLES = args.max_samples
IMAGE_SIZE = args.image_size



# ===================== Load the vision-language model =====================
# hf_token = "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"

model_name = "meta-llama/Llama-3.2-11B-Vision"
from huggingface_hub import HfFolder
# hf_token = HfFolder.get_token()
hf_token = "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"
device = torch.device(DEVICE)
processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

from inspect import signature
print("\nCheck the parameters of MLLama model's forward function:")
print(signature(model.forward))

# Define tokenizer (obtain from processor)
tokenizer = processor.tokenizer

# ===== Apply LoRA adaptation or freeze the base model =====
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
    # Freeze original model parameters
    for param in model.parameters():
        param.requires_grad = False




if args.use_parallel and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    # Do not set CUDA_VISIBLE_DEVICES; it conflicts with DataParallel
    
    # 1. Load the model first (do not move to device yet)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name, 
        token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 2. Apply LoRA (if needed)
    if USE_LORA:
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, config)
    
    # 3. Wrap with DataParallel
    model = nn.DataParallel(model)
    
    # 4. Finally move to device
    model = model.to(device)
    print(f"DataParallel enabled, using GPUs: {list(range(torch.cuda.device_count()))}")
else:
    print(f"Training on a single device: {DEVICE}")
    # Single-GPU mode
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name, 
        token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)

# ====================== Load Stable Diffusion for image generation ======================
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    scheduler=EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
    )
).to(device)

# ============== Custom multimodal mapping head =================
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        # Handle DataParallel-wrapped models
        if isinstance(base_model, nn.DataParallel):
            config = base_model.module.config
            print("Detected DataParallel model, accessing config from .module")
        else:
            config = base_model.config
        
        # Get the correct vocab_size for multimodal models
        if hasattr(config, 'text_config'):
            self.vocab_size = config.text_config.vocab_size
        elif hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            self.vocab_size = len(tokenizer)
            print(f"Could not find vocab_size in model config; using tokenizer vocab size: {self.vocab_size}")
        
        # Get the correct hidden_size
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        elif hasattr(config, 'text_config'):
            self.hidden_size = config.text_config.hidden_size
        else:
            # MLLama typically uses 4096 or 8192
            self.hidden_size = 4096
            print(f"Could not determine hidden_size from config; using default: {self.hidden_size}")
        
        # The rest of the code remains unchanged
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Multimodal-specific mapping layer
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 1st linear transform and normalization
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # Second layer (optional)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # Hyperbolic classifier: num_features = hidden_size + 1 (the extra 1 is the time component)
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # Euclidean classifier
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # Through multimodal adapter
        x = self.multimodal_adapter(last_hidden_states)
        
        # Standard processing
        x = self.linear1(x)
        x = self.norm1(x)
        
        if self.num_layers == 2:
            x = torch.relu(x)
            x = self.linear2(x)
            x = self.norm2(x)
        
        if self.use_hyperbolic:
            # Add time component, then perform hyperbolic mapping
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# Image transform definition - normalization suitable for MLLama
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# ====================== Load the COCO dataset ======================
print("Loading the COCO dataset with COCO API...")

train_loader = None
val_loader = None
test_loader = None
synthetic_dataset = False

from pycocotools.coco import COCO

# Set COCO dataset paths
data_dir = 'coco_dataset'
train_annotations = os.path.join(data_dir, 'annotations/captions_train2017.json')
val_annotations = os.path.join(data_dir, 'annotations/captions_val2017.json')
train_image_dir = os.path.join(data_dir, 'train2017')
val_image_dir = os.path.join(data_dir, 'val2017')

# Check that dataset files exist
if not os.path.exists(train_annotations) or not os.path.exists(train_image_dir):
    raise FileNotFoundError(f"COCO dataset files not found. Please ensure the dataset is downloaded to the {data_dir} directory")

# Load COCO API
print("Loading COCO training annotations...")
train_coco = COCO(train_annotations)
print("Loading COCO validation annotations...")
val_coco = COCO(val_annotations)

# Get all image IDs
train_ids = list(train_coco.imgs.keys())
val_ids = list(val_coco.imgs.keys())

# For the test set, split from the validation set
random.shuffle(val_ids)
val_split = int(len(val_ids) * 0.5)
new_val_ids = val_ids[:val_split]
test_ids = val_ids[val_split:]

print(f"COCO dataset loaded successfully! Train: {len(train_ids)} images, Val: {len(new_val_ids)} images, Test: {len(test_ids)} images")

# Create the COCO dataset class
class COCODataset(Dataset):
    def __init__(self, coco, img_ids, img_dir, transform=None, max_length=128):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Get image ID and path
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image and resize
        image = Image.open(img_path).convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Aspect-ratio-preserving resize
        w, h = image.size
        if w > h:
            new_w = IMAGE_SIZE
            new_h = int(h * IMAGE_SIZE / w)
        else:
            new_h = IMAGE_SIZE
            new_w = int(w * IMAGE_SIZE / h)
                
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Then create a square image and center the resized image
        square_img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        paste_x = (IMAGE_SIZE - new_w) // 2
        paste_y = (IMAGE_SIZE - new_h) // 2
        square_img.paste(image, (paste_x, paste_y))
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Randomly select one caption — this line of code may be missing on your server
        caption = random.choice([ann['caption'] for ann in anns]) if anns else "No description"
        
        return {
            "image": square_img,  # Should use square_img instead of image
            "caption": caption,
            "image_id": img_id
        }
# If MAX_SAMPLES is set, randomly select samples
# Create datasets
train_dataset = COCODataset(train_coco, train_ids[:MAX_SAMPLES] if MAX_SAMPLES else train_ids, 
                           train_image_dir, transform=None, max_length=MAX_LENGTH)
val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, 
                         transform=None, max_length=MAX_LENGTH)
test_dataset = COCODataset(val_coco, test_ids, val_image_dir, 
                          transform=None, max_length=MAX_LENGTH)

def collate_fn(batch):
    images = [item["image"] for item in batch]  # Keep as a list of PIL images
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
        
    return {
        "image": images,
        "caption": captions,
        "image_id": image_ids
    }

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    collate_fn=collate_fn
)

print(f"Data loaders created successfully! Batch size: {BATCH_SIZE}")

# =========== Initialize model components ===========
custom_lm_head = MultimodalMappingHead(model, use_hyperbolic=USE_HYPERBOLIC).to(device)
learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))

# ========= Build optimizer and learning-rate scheduler ===========
curvature_lr = 1e-6
optimizer = torch.optim.AdamW([
    {"params": list(custom_lm_head.parameters()), "lr": BASE_LR},
    {"params": [learnable_curvature], "lr": curvature_lr}
] + ([{"params": [p for p in model.parameters() if p.requires_grad], "lr": BASE_LR}] if USE_LORA else []))

# Compute training steps
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_loss = float("inf")

# ==================== Helper functions ====================

def prepare_multimodal_inputs(batch, processor):
    """Input processing optimized specifically for the MLLama-3.2-Vision model"""
    images = batch["image"]
    captions = batch["caption"]
    
    # Print shapes only for the first batch
    static_printed = getattr(prepare_multimodal_inputs, "printed", False)
    
    # Directly use the processor
    inputs = processor(
        images=images,
        text=captions,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True
    )
    
    # Print shapes only the first time
    if not static_printed:
        print("\nProcessed input shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}, dtype={v.dtype}")
        prepare_multimodal_inputs.printed = True
    
    # Force-remove cross_attention_mask, but avoid repeated prints
    if "cross_attention_mask" in inputs:
        if not static_printed:
            print("Removing problematic cross_attention_mask")
        inputs.pop("cross_attention_mask", None)
    
    # Ensure required fields are present
    required_fields = ["input_ids", "attention_mask", "pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"]
    for field in required_fields:
        if field not in inputs:
            raise ValueError(f"Processed inputs are missing a required field: {field}")
    
    return inputs

def prepare_labels(batch):
    """Prepare training labels"""
    input_ids = batch["input_ids"].clone()
    
    # Shift labels to the right by 1, fill the last position with -100 (ignore)
    labels = torch.roll(input_ids, shifts=1, dims=1)
    labels[:, 0] = -100  # Ignore the first position
    labels[:, -1] = -100  # Ignore the last position
    
    # Set the padding positions to -100
    padding_mask = (input_ids == tokenizer.pad_token_id)
    labels[padding_mask] = -100
    
    return labels

def log_metrics(epoch, avg_loss, ppl, elapsed_time):
    """Record training metrics"""
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, LR={current_lr:.2e}, Time={elapsed_time:.2f}s, Curvature={learnable_curvature.item():.4f}")

def save_model(epoch, loss, ppl):
    """Save the model"""
    global best_loss
    model_type = "hyp" if USE_HYPERBOLIC else "euc"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./checkpoints/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    if loss < best_loss:
        best_loss = loss
        filename = f"{model_type}_vision_epoch_{epoch}_loss_{loss:.4f}_PPL_{ppl:.2f}.pth"
        save_path = os.path.join(save_dir, filename)
        
        torch.save({
            "lm_head_state": custom_lm_head.state_dict(),
            "curvature": learnable_curvature.item(),
            "optimizer_state": optimizer.state_dict()
        }, save_path)
        
        if USE_LORA:
            lora_save_path = os.path.join(save_dir, f"lora_{model_type}_vision_epoch_{epoch}")
            model.save_pretrained(lora_save_path)
            
        print(f"Model checkpoint saved: {save_path}")

def compute_lm_loss(logits, labels):
    """Compute language model loss"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

# ====================== Train the model ======================
def get_gpu_memory_stats():
    """Get GPU memory usage"""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    stats = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        stats.append(f"GPU{i}: {allocated:.1f}GB/{reserved:.1f}GB")
    
    return " | ".join(stats)

# Enhanced training-metrics logging function
def log_metrics(epoch, avg_loss, ppl, elapsed_time):
    """Log detailed training metrics for each epoch"""
    current_lr = optimizer.param_groups[0]['lr']
    gpu_stats = get_gpu_memory_stats() if torch.cuda.is_available() else "N/A"
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch} training summary (elapsed: {elapsed_time:.2f}s)")
    print(f"{'='*50}")
    print(f"Training loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print(f"Learning rate: {current_lr:.2e}")
    print(f"Hyperbolic curvature: {learnable_curvature.item():.4f}")
    print(f"GPU memory usage: {gpu_stats}")
    print(f"Average time per batch: {elapsed_time/len(train_loader):.3f}s")
    print(f"Estimated time for next epoch: {elapsed_time/60:.1f} min")
    print(f"{'='*50}")

def train_model():
    global best_loss
    gradient_accumulation_steps = 4
    log_interval = 20  # Show loss every 20 batches
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0
        recent_losses = []  # Store recent loss values
        running_loss = 0.0  # For short-term average
        
        # Zero gradients outside the loop
        optimizer.zero_grad()
        
        # Clear memory before each epoch
        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", 
                                      bar_format='{l_bar}{bar:30}{r_bar}')):
            batch_start = time.time()
            
            # Process multimodal inputs
            inputs = prepare_multimodal_inputs(batch, processor)
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Prepare labels
            labels = prepare_labels(inputs)
            
            # Forward pass and loss computation
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                logits = custom_lm_head(outputs.hidden_states[-1], learnable_curvature)
                
                # Compute and record the raw (unscaled) loss
                batch_loss = compute_lm_loss(logits, labels)
                loss = batch_loss / gradient_accumulation_steps  # Scaled for gradient accumulation
            
            # Backpropagation
            loss.backward()
            
            # Record per-batch loss
            batch_size = len(batch["image"])
            item_loss = batch_loss.item()
            recent_losses.append(item_loss)
            running_loss += item_loss
            total_loss += item_loss * batch_size
            count += batch_size
            
            # Print average loss every log_interval batches
            if (i + 1) % log_interval == 0:
                avg_recent_loss = sum(recent_losses[-log_interval:]) / min(log_interval, len(recent_losses))
                lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {avg_recent_loss:.4f}, LR: {lr:.1e}, Curvature: {learnable_curvature.item():.4f}")
            
            # Update parameters after specified accumulation steps
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                # Gradient clipping
                clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + 
                    list(custom_lm_head.parameters()) + 
                    [learnable_curvature], 
                    max_norm=1.0
                )
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clamp curvature range
                with torch.no_grad():
                    learnable_curvature.clamp_(1e-3, 1e1)
            
            # Empty CUDA cache periodically to reduce fragmentation
            if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()

        # Compute and log metrics per epoch
        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        
        # Detailed epoch summary
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        
        # Validation
        print("\nStart validation...")
        val_loss = evaluate_model(val_loader, phase="Validation")
        save_model(epoch, val_loss, math.exp(val_loss) if val_loss < 20 else float("inf"))
# ====================== Evaluation function ======================
def evaluate_model(data_loader, phase="Val"):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {phase}", 
                         bar_format='{l_bar}{bar:30}{r_bar}'):
            # Process multimodal inputs
            inputs = prepare_multimodal_inputs(batch, processor)
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Prepare labels
            labels = prepare_labels(inputs)
            
            # Forward pass — use autocast to ensure dtype consistency
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1]
                logits = custom_lm_head(hidden_states, learnable_curvature)
                loss_val = compute_lm_loss(logits, labels)
            
            batch_size = len(batch["image"])
            total_loss += loss_val.item() * batch_size
            count += batch_size
        
        # Concise summary after evaluation
        eval_time = time.time() - start_time
        avg_loss = total_loss / count if count > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        print(f"{phase} results: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={eval_time:.2f}s")
        return avg_loss
# ====================== Generate text from image ======================
def generate_text_from_image(image_path, max_len=100):
    """Given an image, generate text"""
    model.eval()
    custom_lm_head.eval()
    
    # Load image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Process image
    image_outputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = image_outputs.pixel_values.to(device)
    
    # Compute aspect ratio ID
    original_aspect = image.width / image.height
    if 0.9 <= original_aspect <= 1.1:
        aspect_id = 0  # square
    elif original_aspect < 0.9:
        aspect_id = 1  # portrait
    else:
        aspect_id = 2  # landscape
    aspect_ratio_ids = torch.tensor([aspect_id], dtype=torch.long).to(device)
    
    # Create aspect_ratio_mask
    aspect_ratio_mask = torch.ones((1, 1), dtype=torch.long).to(device)
    
    # Prepare prompt text
    prompt = "Describe this image:"
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Prepare all required inputs
    inputs = {
        "pixel_values": pixel_values,
        "aspect_ratio_ids": aspect_ratio_ids,
        "aspect_ratio_mask": aspect_ratio_mask,
        "input_ids": input_ids
    }
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode output
    generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# ====================== Generate image from text ======================
def generate_image_from_text(text_prompt, output_path=None):
    """Given a text prompt, generate an image"""
    # Use Stable Diffusion to generate an image
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image = image_gen_model(text_prompt, guidance_scale=7.5).images[0]
    
    # Save image
    if output_path:
        image.save(output_path)
    
    return image

# ====================== Main execution ======================
if __name__ == "__main__":
    print(f"USE_HYPERBOLIC = {USE_HYPERBOLIC}")
    print(f"USE_LORA = {USE_LORA}")
    
    # Train the model
    train_model()
    
    # Evaluate on test set
    print("\n=== Testing ===")
    evaluate_model(test_loader, phase="Test")
    
    # # filepath: /gpfs/gibbs/project/ying_rex/sz583/hyperbolic_mapping/image_mapping.py
