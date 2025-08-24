import sys
import os
import time
import math
import argparse
import traceback
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, MllamaForConditionalGeneration, get_linear_schedule_with_warmup

from torchvision import transforms

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def get_parser():
    parser = argparse.ArgumentParser(description="Train a multimodal vision-language model with Hyperbolic mapping")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for the dataset to load")
    parser.add_argument("--image_size", type=int, default=224, help="Size of input images (224 for MLLama)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use (e.g., 'cuda', 'cuda:1', 'cpu')")
    parser.add_argument("--gpu_ids", type=str, default="0", 
                    help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--use_parallel", action="store_true",
                    help="Use DataParallel for multi-GPU training")
    parser.add_argument("--curvature_lr", type=float, default=1e-6, 
                    help="Learning rate for the curvature parameter")
    parser.add_argument("--save_interval", type=int, default=1, 
                    help="Save checkpoint every N epochs")
    parser.add_argument("--eval_steps", type=int, default=2000, 
                    help="Evaluate model every N steps")
    parser.add_argument("--resume", type=str, default="", 
                    help="Path to resume training from a checkpoint")
    parser.add_argument("--start_epoch", type=int, default=1,
                    help="Starting epoch when resuming training")
    parser.add_argument("--mode", type=str, default="train", 
                      choices=["train", "eval", "demo", "research"],
                      help="Run mode: train=training, eval=evaluation, demo=demonstration, research=research")
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
CURVATURE_LR = args.curvature_lr

# ===================== Load the vision-language model =====================
print(f"{'='*40}\nLoading model: Deepseek-AI/Janus-Pro\n{'='*40}")

model_name = "deepseek-ai/Janus-Pro-7B"
device = torch.device(DEVICE)

# Load processor and model
processor = VLChatProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

# Model loading and parallel processing
if args.use_parallel and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    # Load the model first (do not move to device yet)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Apply LoRA (if needed)
    if USE_LORA:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Janus attention module names
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        # Freeze original model parameters
        for param in model.parameters():
            param.requires_grad = False
    
    # Then wrap with DataParallel
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"DataParallel enabled, using GPUs: {list(range(torch.cuda.device_count()))}")
else:
    print(f"Training on a single device: {DEVICE}")
    # Single-GPU/CPU mode
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # Apply LoRA or freeze parameters
    if USE_LORA:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        # Freeze original model parameters
        for param in model.parameters():
            param.requires_grad = False

# ============== Custom multimodal mapping head =================
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        # Handle DataParallel-wrapped models
        if isinstance(base_model, nn.DataParallel):
            config = base_model.module.config
        else:
            config = base_model.config
        
        # Get vocab_size
        self.vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else len(tokenizer)
        
        # Get hidden_size
        self.hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 4096

        print(f"Mapping head config: vocab_size={self.vocab_size}, hidden_size={self.hidden_size}")
        
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Multimodal-specific mapping layer for fusion
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Linear transforms and normalization
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # Second layer (optional)
        if num_layers >= 2:
            self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(self.hidden_size)
        
        # Third layer (optional) - increases capacity
        if num_layers >= 3:
            self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(self.hidden_size)

        # Hyperbolic classifier: num_features = hidden_size + 1 (the extra 1 is the time component)
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # Euclidean classifier
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        print(f"Initialize {'hyperbolic' if use_hyperbolic else 'Euclidean'} mapping head, layers: {num_layers}")

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # Through multimodal adapter
        x = self.multimodal_adapter(last_hidden_states)
        
        # First layer
        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        
        # Second layer (if present)
        if self.num_layers >= 2:
            x = self.linear2(x)
            x = self.norm2(x)
            x = torch.relu(x)
        
        # Third layer (if present)
        if self.num_layers >= 3:
            x = self.linear3(x)
            x = self.norm3(x)
            x = torch.relu(x)
        
        # Choose classifier by geometry
        if self.use_hyperbolic:
            # Add time component then hyperbolic mapping
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# Image transforms — normalization suitable for MLLama
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ====================== Load the COCO dataset ======================
print(f"{'='*40}\nLoading the COCO dataset\n{'='*40}")

try:
    from pycocotools.coco import COCO
    
    # Set COCO dataset paths
    data_dir = 'coco_dataset'
    train_annotations = os.path.join(data_dir, 'annotations/captions_train2017.json')
    val_annotations = os.path.join(data_dir, 'annotations/captions_val2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')
    val_image_dir = os.path.join(data_dir, 'val2017')
    
    # Check dataset files exist
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
    
    # Split validation into val/test
    random.shuffle(val_ids)
    val_split = int(len(val_ids) * 0.5)
    new_val_ids = val_ids[:val_split]
    test_ids = val_ids[val_split:]
    
    if MAX_SAMPLES and MAX_SAMPLES > 0:
        train_ids = train_ids[:MAX_SAMPLES]
        new_val_ids = new_val_ids[:MAX_SAMPLES//5]
        test_ids = test_ids[:MAX_SAMPLES//5]
    
    print(f"Dataset size - train: {len(train_ids)} images, val: {len(new_val_ids)} images, test: {len(test_ids)} images")
    
    # COCO dataset class
    class COCODataset(Dataset):
        def __init__(self, coco, img_ids, img_dir, max_length=128):
            self.coco = coco
            self.img_ids = img_ids
            self.img_dir = img_dir
            self.max_length = max_length
            
        def __len__(self):
            return len(self.img_ids)
        
        def __getitem__(self, idx):
            # Get image ID and path
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                # Create a blank fallback image
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
            
            # Get captions
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Pick a random caption
            caption = random.choice([ann['caption'] for ann in anns]) if anns else "No description"
            
            # Make square while preserving aspect ratio
            w, h = image.size
            if w > h:
                new_w = IMAGE_SIZE
                new_h = int(h * IMAGE_SIZE / w)
            else:
                new_h = IMAGE_SIZE
                new_w = int(w * IMAGE_SIZE / h)
                    
            image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Paste centered on a square canvas
            square_img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
            paste_x = (IMAGE_SIZE - new_w) // 2
            paste_y = (IMAGE_SIZE - new_h) // 2
            square_img.paste(image, (paste_x, paste_y))
            
            return {
                "image": square_img,
                "caption": caption,
                "image_id": img_id
            }

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
            
        return {
            "image": images,
            "caption": captions,
            "image_id": image_ids
        }
    
    # Create datasets
    train_dataset = COCODataset(train_coco, train_ids, train_image_dir, max_length=MAX_LENGTH)
    val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, max_length=MAX_LENGTH)
    test_dataset = COCODataset(val_coco, test_ids, val_image_dir, max_length=MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    print(f"Data loaders created successfully! Batch size: {BATCH_SIZE}")
    
except Exception as e:
    print(f"Failed to load COCO dataset: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==================== Helper functions ====================
def prepare_multimodal_inputs_batch(batch, processor):
    """Process an entire batch of images and text"""
    batch_size = len(batch["image"])
    all_inputs = []
    
    # Ensure temp directory exists
    os.makedirs("temp_images", exist_ok=True)
    
    for i in range(batch_size):
        image = batch["image"][i]
        caption = batch["caption"][i]
        
        # Save image to temp file - use index to ensure a unique filename
        temp_img_path = f"temp_images/temp_{time.time()}_{i}.jpg"
        image.save(temp_img_path)
        
        # Build conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>\nDescribe this image",
                "images": [temp_img_path],
            },
            {"role": "<|Assistant|>", "content": caption},
        ]
        
        # Process images and conversation
        pil_images = load_pil_images(conversation)
        input_data = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        )
        
        # Append processed input
        all_inputs.append(input_data)
        
        # Clean up temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    
    # Combine processed inputs
    if len(all_inputs) == 0:
        raise ValueError("No valid inputs in the batch")
    
    return batch_combine_inputs(all_inputs)

def batch_combine_inputs(input_list):
    """Combine processed input list into one batch, handling variable-length sequences"""
    if len(input_list) == 1:
        return input_list[0]
    
    # Use the first input's keys as reference
    keys = input_list[0].keys()
    combined = {}
    
    for key in keys:
        if isinstance(input_list[0][key], torch.Tensor):
            # Handle different lengths
            if input_list[0][key].dim() > 1:
                tensors = [inp[key] for inp in input_list]
                shapes = [t.shape for t in tensors]
                
                # Check if shapes are the same beyond dim 0
                if not all(s[1:] == shapes[0][1:] for s in shapes):
                    # Need padding — find max per dimension
                    max_dims = []
                    for dim in range(1, len(shapes[0])):
                        max_dims.append(max(s[dim] for s in shapes))
                    
                    # Pad tensors
                    padded_tensors = []
                    for tensor in tensors:
                        pad_sizes = []
                        for dim in range(1, len(tensor.shape)):
                            pad_sizes.extend([0, max_dims[dim-1] - tensor.shape[dim]])
                        if any(p > 0 for p in pad_sizes):
                            padded = torch.nn.functional.pad(tensor, pad_sizes, 'constant', 0)
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    
                    combined[key] = torch.cat(padded_tensors, dim=0)
                else:
                    combined[key] = torch.cat(tensors, dim=0)
            else:
                combined[key] = torch.cat([inp[key] for inp in input_list], dim=0)
        elif key in ['pixel_values', 'images_emb_mask', 'images_seq_mask']:
            # Special handling for image-related tensors; also check shapes
            try:
                combined[key] = torch.cat([inp[key] for inp in input_list], dim=0)
            except RuntimeError as e:
                print(f"Image tensor shape mismatch: {e}, trying to pad...")
                tensors = [inp[key] for inp in input_list]
                shapes = [t.shape for t in tensors]
                print(f"Shapes of different samples: {shapes}")
                
        else:
            # Non-tensor fields: keep as list
            combined[key] = [inp[key] for inp in input_list]
    
    return combined

def prepare_labels(batch):
    """Prepare training labels"""
    input_ids = batch["input_ids"].clone()
    
    # Shift labels left by one; set the last position to -100 (ignore)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100  # Ignore the last position
    
    # Set padding positions to -100
    padding_mask = (input_ids == tokenizer.pad_token_id)
    labels[padding_mask] = -100
    
    return labels

def compute_lm_loss(logits, labels):
    """Compute language modeling loss"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

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

# =========== Initialize model components ===========
custom_lm_head = MultimodalMappingHead(model, use_hyperbolic=USE_HYPERBOLIC, num_layers=3).to(device)
learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))

# ========= Build optimizer and scheduler ===========
optimizer = torch.optim.AdamW([
    {"params": list(custom_lm_head.parameters()), "lr": BASE_LR},
    {"params": [learnable_curvature], "lr": CURVATURE_LR}
] + ([{"params": [p for p in model.parameters() if p.requires_grad], "lr": BASE_LR}] if USE_LORA else []))

# Compute training steps
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_loss = float("inf")

# Enhanced training metrics logger
def log_metrics(epoch, avg_loss, ppl, elapsed_time):
    """Log detailed metrics for each epoch"""
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
    print(f"Avg time per batch: {elapsed_time/len(train_loader):.3f}s")
    print(f"Estimated next epoch time: {elapsed_time/60:.1f} min")
    print(f"{'='*50}")

def save_model_for_hf(model, custom_lm_head, curvature, output_dir, processor=None):
    """Save model in a HuggingFace-compatible format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Save base model
    if isinstance(model, nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
    
    base_model.save_pretrained(output_dir)
    
    # Save processor if provided
    if processor is not None:
        processor.save_pretrained(output_dir)
    
    # 2) Update config with hyperbolic mapping info
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["hyperbolic_mapping"] = {
            "enabled": USE_HYPERBOLIC,
            "curvature": float(curvature.item()),
            "num_layers": custom_lm_head.num_layers,
            "model_type": "janus-hyperbolic" if USE_HYPERBOLIC else "janus-euclidean"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # 3) Save custom mapping head and curvature
    custom_head_path = os.path.join(output_dir, "hyperbolic_mapping_head.bin")
    torch.save({
        "lm_head_state": custom_lm_head.state_dict(),
        "curvature": curvature.item(),
        "use_hyperbolic": USE_HYPERBOLIC,
        "hidden_size": custom_lm_head.hidden_size,
        "vocab_size": custom_lm_head.vocab_size,
        "num_layers": custom_lm_head.num_layers
    }, custom_head_path)
    
    print(f"HuggingFace-format model saved to: {output_dir}")

    return output_dir

def save_model(epoch, loss, ppl):
    """Save model every epoch and also in HuggingFace format periodically"""
    global best_loss
    model_type = "hyp" if USE_HYPERBOLIC else "euc"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./checkpoints/{model_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best
    is_best = loss < best_loss
    if is_best:
        best_loss = loss
    
    # Always save current epoch (native format)
    filename = f"{model_type}_vision_epoch_{epoch}_loss_{loss:.4f}_PPL_{ppl:.2f}{'_best' if is_best else ''}.pth"
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        "lm_head_state": custom_lm_head.state_dict(),
        "curvature": learnable_curvature.item(),
        "epoch": epoch,
        "loss": loss,
        "ppl": ppl,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }, save_path)
    
    if USE_LORA:
        lora_save_path = os.path.join(save_dir, f"lora_{model_type}_epoch_{epoch}")
        if isinstance(model, nn.DataParallel):
            model.module.save_pretrained(lora_save_path)
        else:
            model.save_pretrained(lora_save_path)
    
    # Save HuggingFace format for best or every 5 epochs
    if is_best or (epoch % 5 == 0):
        hf_dir = f"./hf_models/{model_type}_epoch_{epoch}"
        save_model_for_hf(
            model=model,
            custom_lm_head=custom_lm_head,
            curvature=learnable_curvature,
            output_dir=hf_dir,
            processor=processor
        )
        print(f"Also saved in HuggingFace format: {hf_dir}")
    
    print(f"Checkpoint saved: {save_path}")
    return save_path


def load_checkpoint(checkpoint_path):
    """Load checkpoint to resume training"""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore mapping head
    custom_lm_head.load_state_dict(checkpoint["lm_head_state"])
    
    # Restore curvature parameter
    with torch.no_grad():
        learnable_curvature.copy_(torch.tensor(checkpoint["curvature"], device=device))
    
    # Restore optimizer
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # Ensure tensors are on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Restore scheduler
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    # Restore LoRA weights (if present)
    if USE_LORA and os.path.dirname(checkpoint_path):
        lora_dir = os.path.dirname(checkpoint_path)
        lora_dirs = [d for d in os.listdir(lora_dir) if d.startswith("lora_")]
        if lora_dirs:
            newest_lora = sorted(lora_dirs)[-1]
            lora_path = os.path.join(lora_dir, newest_lora)
            print(f"Loading LoRA weights: {lora_path}")
            if isinstance(model, nn.DataParallel):
                model.module.load_adapter(lora_path)
            else:
                model.load_adapter(lora_path)
    
    # Return resume info
    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))


# ====================== Train the model ======================
gradient_accumulation_steps = 4
def train_model():
    global best_loss
    latest_checkpoint_path = None
    log_interval = 1000  # Show loss every N batches
    
    start_epoch = args.start_epoch
    if args.resume:
        loaded_epoch, loaded_loss = load_checkpoint(args.resume)
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1
            best_loss = min(best_loss, loaded_loss)
            print(f"Successfully resumed to epoch {loaded_epoch}; continuing from epoch {start_epoch}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0
        recent_losses = []  # Store recent loss values
        
        # Zero grads outside the loop
        optimizer.zero_grad()
        
        # Clear memory before each epoch
        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}", 
                                      bar_format='{l_bar}{bar:30}{r_bar}')):

            # Process multimodal inputs
            inputs = prepare_multimodal_inputs_batch(batch, processor)
            
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            labels = prepare_labels(inputs)
            
            # Internal model call with autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs_embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
                
                # Call internal model
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Last hidden state
                hidden_states = outputs.last_hidden_state
                
                # Custom mapping head
                logits = custom_lm_head(hidden_states, learnable_curvature)
                raw_loss = compute_lm_loss(logits, labels)
                loss = raw_loss / gradient_accumulation_steps
            # Backprop
            loss.backward()
            
            # Track batch loss
            batch_size = len(batch["image"])
            item_loss = raw_loss.item()
            recent_losses.append(item_loss)
            total_loss += item_loss * batch_size
            count += batch_size
            
            # Log every log_interval
            if (i + 1) % log_interval == 0:
                avg_recent_loss = sum(recent_losses[-log_interval:]) / min(log_interval, len(recent_losses))
                lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i+1}/{len(train_loader)}, loss: {avg_recent_loss:.4f}, LR: {lr:.1e}, curvature: {learnable_curvature.item():.4f}")
            
            # Optimizer step on accumulation
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                # Gradient clipping
                clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + 
                    list(custom_lm_head.parameters()) + 
                    [learnable_curvature], 
                    max_norm=1.0
                )
                
                # Step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clamp curvature
                with torch.no_grad():
                    learnable_curvature.clamp_(1e-3, 1e1)
            
            # Periodically empty CUDA cache
            if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
                
            # Periodic evaluation
            if (i + 1) % args.eval_steps == 0:
                print("\nRunning intermediate evaluation...")
                interim_loss = evaluate_model(val_loader, phase="intermediate validation", max_batches=50)
                print(f"Intermediate validation loss: {interim_loss:.4f}")
                # Back to train mode
                model.train()
                custom_lm_head.train()

        # Epoch metrics
        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        
        # Detailed epoch summary
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        
        # Validation
        print("\nStarting full validation...")
        val_loss = evaluate_model(val_loader, phase="validation")
        latest_checkpoint_path = save_model(epoch, val_loss, math.exp(val_loss) if val_loss < 20 else float("inf"))
        # Saved
        print(f"Saved checkpoint for epoch {epoch}: {latest_checkpoint_path}")

# ====================== Evaluation ======================
def evaluate_model(data_loader, phase="validation", max_batches=None):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {phase}", 
                                     bar_format='{l_bar}{bar:30}{r_bar}')):
            if max_batches and i >= max_batches:
                break
                
            # Process multimodal inputs
            inputs = prepare_multimodal_inputs_batch(batch, processor)
            
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}            
            # Prepare labels
            labels = prepare_labels(inputs)
            
            # Forward with autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs_embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
                
                # Internal model
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # Last hidden state
                hidden_states = outputs.last_hidden_state
                
                # Custom head
                logits = custom_lm_head(hidden_states, learnable_curvature)
                loss = compute_lm_loss(logits, labels) 
            
            batch_size = len(batch["image"])
            total_loss += loss.item() * batch_size
            count += batch_size
        
        # Summary
        eval_time = time.time() - start_time
        avg_loss = total_loss / count if count > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        print(f"{phase} results: loss={avg_loss:.4f}, PPL={ppl:.2f}, time={eval_time:.2f}s")
        return avg_loss


# ====================== Main execution ======================
# ====================== Main execution ======================
if __name__ == "__main__":
    print(f"{'='*50}")
    print(f"Config: Janus-Pro model, hyperbolic={USE_HYPERBOLIC}, LoRA={USE_LORA}, batch={BATCH_SIZE}")
    print(f"LR={BASE_LR}, curvature LR={CURVATURE_LR}, device={DEVICE}")
    print(f"{'='*50}")
    
    # Use args.mode; remove duplicate parsing
    if args.mode == "train":
        # Training mode
        print("Start training model...")
        train_model()
    
    elif args.mode == "eval":
        # Evaluation mode
        print("Start evaluating model...")
        evaluate_model(test_loader, phase="test")
