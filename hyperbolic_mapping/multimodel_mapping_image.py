import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, PreTrainedModel
from janus.models import VLChatProcessor, MultiModalityCausalLM # Ensure to import MultiModalityCausalLM
from janus.models.vq_model import VQModel, ModelArgs # Import VQModel from Janus
from janus.models.image_processing_vlm import VLMImageProcessor, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD # Import from Janus
from PIL import Image
from tqdm import tqdm
import math
import random
import numpy as np
from peft import LoraConfig, get_peft_model
import json
import traceback
from pycocotools.coco import COCO # Used to load the COCO dataset

# Import HyperLib components
from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz

# --- 1. Argument parsing ---
def get_parser():
    parser = argparse.ArgumentParser(description="Train hyperbolic mapping head from text to image visual tokens")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Janus model")
    parser.add_argument("--coco_data_dir", type=str, default="./coco_dataset", help="Path to COCO dataset directory (containing annotations/ and train2017/)")
    parser.add_argument("--output_dir", type=str, default="./t2i_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping head")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for the language model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the head/LoRA")
    parser.add_argument("--curvature_lr", type=float, default=1e-6, help="Learning rate for curvature")
    parser.add_argument("--max_text_length", type=int, default=77, help="Max length for text prompts")
    parser.add_argument("--image_size", type=int, default=384, help="Target image size for VQModel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Use pin memory for DataLoader")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Computation data type")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--head_num_layers", type=int, default=3, help="Number of layers in the mapping head")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from specified checkpoint file")
    # Add LoRA target modules parameter
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Target modules for LoRA")

    return parser

# --- 2. Set random seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 3. Custom generation head (HyperbolicGenHead) ---
class HyperbolicGenHead(nn.Module):
    def __init__(self, hidden_size, codebook_size, use_hyperbolic=True, num_layers=3):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size # Visual codebook size (e.g., 16384)
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Similar to the image-to-text generation head, but the final output dimension is codebook_size
        # Note: There's no multimodal_adapter here, starting directly from hidden_states
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        if num_layers >= 2:
            self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(hidden_size)
        if num_layers >= 3:
            self.linear3 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(hidden_size)
        # More layers can be added if needed

        # Classifier
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=hidden_size + 1, # Lorentz requires a time component
            num_classes=self.codebook_size
        )
        self.euc_cls = nn.Linear(hidden_size, self.codebook_size, bias=False)

        print(f"Initialized {'Hyperbolic' if use_hyperbolic else 'Euclidean'} GenHead, output dimension: {self.codebook_size}, layers: {self.num_layers}")

    def lorentz_map(self, x, c_param):
        k_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
        if k_tensor.dim() == 0:
            k_tensor = k_tensor.unsqueeze(0)
        return expmap0(x, k=k_tensor, dim=-1)

    def forward(self, hidden_states, c_param=None):
        target_dtype = self.linear1.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        x = hidden_states # Directly use hidden_states

        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        if self.num_layers >= 2:
            x = self.linear2(x)
            x = self.norm2(x)
            x = torch.relu(x)
        if self.num_layers >= 3:
            x = self.linear3(x)
            x = self.norm3(x)
            x = torch.relu(x)
        # More layers can be added if needed

        if self.use_hyperbolic:
            if c_param is None:
                raise ValueError("Curvature parameter 'c_param' is required for hyperbolic head.")
            x = self.manifold.add_time(x)
            c_param_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
            hyper_embs = self.lorentz_map(x, c_param_tensor)
            if hasattr(self.hyp_cls, 'z'):
                if hyper_embs.dtype != self.hyp_cls.z.dtype:
                    hyper_embs = hyper_embs.to(self.hyp_cls.z.dtype)
                # If 'z' doesn't exist, try 'a' or throw an error
                # print("Warning: LorentzMLR has no 'z' attribute, checking 'a'")
                # if hasattr(self.hyp_cls, 'a') and hyper_embs.dtype != self.hyp_cls.a.dtype:
                #      hyper_embs = hyper_embs.to(self.hyp_cls.a.dtype)
            else:
                raise AttributeError("LorentzMLR object has no 'z' or 'a' attribute to check dtype against.")
            # --- End modification ---
                 
            logits = self.hyp_cls(hyper_embs)
        else:
            if x.dtype != self.euc_cls.weight.dtype:
                x = x.to(self.euc_cls.weight.dtype)
            logits = self.euc_cls(x)

        return logits # Output logits of visual tokens


def evaluate(model, val_loader, loss_fn, device, compute_dtype, args, learnable_curvature, codebook_size):
    model.eval() # Set to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            if batch is None: continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_visual_tokens = batch["target_visual_tokens"].to(device)

            # --- Forward pass similar to training loop ---
            with torch.autocast(device_type=args.device, dtype=compute_dtype, enabled=(compute_dtype != torch.float32)):
                text_embeds = model.language_model.get_input_embeddings()(input_ids)
                visual_bos_id = 0
                visual_bos_ids = torch.full((target_visual_tokens.size(0), 1), visual_bos_id, dtype=torch.long, device=device)
                input_visual_ids = torch.cat([visual_bos_ids, target_visual_tokens[:, :-1]], dim=1)
                visual_embeds = model.prepare_gen_img_embeds(input_visual_ids)
                combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)
                visual_attention_mask = torch.ones_like(input_visual_ids)
                combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

                outputs = model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                visual_hidden_states = hidden_states[:, input_ids.shape[1]:, :]
                current_curvature = learnable_curvature.to(visual_hidden_states.dtype) if learnable_curvature is not None else None
                logits = model.gen_head(visual_hidden_states, current_curvature) # Use model.gen_head

                loss = loss_fn(logits.view(-1, codebook_size), target_visual_tokens.view(-1))
                total_val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# --- 4. COCO Dataset definition ---
class COCODataset(Dataset):
    def __init__(self, data_dir, processor, vq_model, image_processor, max_text_length, split='train'):
        self.data_dir = data_dir
        self.processor = processor
        self.vq_model = vq_model # Pass in VQModel instance (should be in eval mode and frozen)
        self.image_processor = image_processor # Pass in VLMImageProcessor instance
        self.max_text_length = max_text_length
        self.split = split

        # Set COCO paths
self.img_dir = os.path.join(data_dir, f'{split}2017')
self.ann_file = os.path.join(data_dir, 'annotations', f'captions_{split}2017.json')

if not os.path.exists(self.img_dir) or not os.path.exists(self.ann_file):
    raise FileNotFoundError(f"COCO dataset files not found in {data_dir}. Please make sure annotations/ and {split}2017/ exist.")

print(f"Loading COCO {split} annotations: {self.ann_file}")
self.coco = COCO(self.ann_file)
self.ids = list(sorted(self.coco.imgs.keys()))
print(f"Found {len(self.ids)} images.")

# Ensure VQModel is in eval mode
self.vq_model.eval()

def __len__(self):
    return len(self.ids)

def __getitem__(self, index):
    img_id = self.ids[index]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    img_info = self.coco.loadImgs(img_id)[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])

    # Randomly select one caption as input
    caption = random.choice(anns)['caption']

    # 1. Process text
    # Use DeepSeek's template to format text input
    # Note: Here we assume we only use one User turn as input
    conv = self.processor.new_chat_template() # Get dialogue template
    conv.append_message(conv.roles[0], caption) # Add user message
    conv.append_message(conv.roles[1], None) # Add empty assistant message to trigger generation format
    text_prompt = conv.get_prompt() # Get formatted text

    text_inputs = self.processor.tokenizer(
        text_prompt,
        max_length=self.max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.squeeze(0) # (seq_len)
    attention_mask = text_inputs.attention_mask.squeeze(0) # (seq_len)

    # 2. Process image and obtain target visual tokens
    try:
        image = Image.open(img_path).convert('RGB')
        # Use VLMImageProcessor to process the image
        image_inputs = self.image_processor.preprocess([image], return_tensors="pt")
        pixel_values = image_inputs['pixel_values'] # (1, 3, H, W)

        # Use VQModel to obtain target visual tokens (inside no_grad context)
        with torch.no_grad():
            # Move pixel_values to the same device and dtype as vq_model
            vq_device = next(self.vq_model.parameters()).device
            vq_dtype = next(self.vq_model.parameters()).dtype
            pixel_values = pixel_values.to(vq_device, vq_dtype)

            # Call encode to obtain visual tokens
            _, _, info = self.vq_model.encode(pixel_values)
            # info = (perplexity, min_encodings, min_encoding_indices)
            target_visual_tokens = info[2].squeeze(0) # (latent_h * latent_w) or (num_tokens)

    except Exception as e:
        print(f"Error processing image {img_path} or obtaining visual tokens: {e}")
        traceback.print_exc()
        # Return None or raise an error, to be handled by collate_fn
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_visual_tokens": target_visual_tokens, # (num_visual_tokens)
    }

# --- 5. Collate function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None] # Filter out failed samples
    if not batch:
        return None

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Process target_visual_tokens (they should have the same length)
    target_visual_tokens = torch.stack([item["target_visual_tokens"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_visual_tokens": target_visual_tokens,
    }

# ... (other imports) ...
import os
import json
import torch
import torch.nn as nn
from peft import PeftModel # Ensure PeftModel is imported

# ... (other code, including HyperbolicGenHead, COCODataset, main, etc.) ...

def save_model_for_hf(model, hyperbolic_gen_head, learnable_curvature, output_dir, processor, args, hidden_size, codebook_size):
    """
    Save the model (including base model config, LoRA adapters, and custom generation head) in HuggingFace-compatible format.

    Args:
        model: Trained model (possibly a PeftModel instance).
        hyperbolic_gen_head: Trained custom generation head (HyperbolicGenHead).
        learnable_curvature: Trained curvature parameter (torch.Tensor).
        output_dir: Directory to save the model.
        processor: VLChatProcessor instance used.
        args: argparse Namespace containing configs (e.g., args.use_hyperbolic, args.head_num_layers, args.use_lora).
        hidden_size: Model hidden size.
        codebook_size: Visual codebook size (vocabulary size of the VQ model).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Start saving HuggingFace-compatible model to: {output_dir}")

       # 1. Save Processor
    if processor is not None:
        try:
            processor.save_pretrained(output_dir)
            print("  - Processor saved.")
        except Exception as e:
            print(f"  - Error: Failed to save Processor: {e}")
    else:
        print("  - Warning: Processor not provided, unable to save.")

    # 2. Save LoRA adapter (if LoRA is used)
    if args.use_lora:
        if isinstance(model, PeftModel):
            try:
                model.save_pretrained(output_dir) # Save adapter_config.json and adapter_model.bin
                print("  - LoRA adapter saved.")
            except Exception as e:
                print(f"  - Error: Failed to save LoRA adapter: {e}")
        else:
            print("  - Warning: LoRA is enabled but the model is not a PeftModel instance, unable to save adapter separately.")

    # 3. Save base model config (always required)
    # Get base model instance (handle PeftModel and DataParallel)
    if isinstance(model, PeftModel):
        base_model_instance = model.base_model.model # Get underlying MultiModalityCausalLM
    elif isinstance(model, nn.DataParallel):
         base_model_instance = model.module
    else:
        base_model_instance = model

    try:
        # Save base model's config.json
        base_model_instance.config.save_pretrained(output_dir)
        print(f"  - Base model config saved.")
    except Exception as e:
        print(f"  - Error: Failed to save base model config: {e}")

    # 4. Save full base model weights if LoRA is not used
    if not args.use_lora:
        try:
            base_model_instance.save_pretrained(output_dir)
            print(f"  - Full base model weights saved (LoRA not used).")
        except Exception as e:
            print(f"  - Error: Failed to save full base model weights: {e}")

    # 5. Modify config file to include custom generation head information
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Add custom generation head info
            config_data["custom_gen_head_config"] = {
                "is_hyperbolic": args.use_hyperbolic,
                "curvature": float(learnable_curvature.item()) if args.use_hyperbolic else None,
                "num_layers": args.head_num_layers,
                "hidden_size": hidden_size,
                "codebook_size": codebook_size,
                "head_type": "HyperbolicGenHead", # Record head type
                "weights_file": "mapping_gen_head.bin" # Path to weights file
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print("  - Main config file updated with custom generation head info.")

        except Exception as e:
            print(f"  - Error: Failed to modify main config file: {e}")
    else:
        print(f"  - Warning: Main config file {config_path} not found, unable to add custom head info.")

    # 6. Save custom generation head weights
    head_state_dict_path = os.path.join(output_dir, "mapping_gen_head.bin") # Use a different filename
    try:
        torch.save(hyperbolic_gen_head.state_dict(), head_state_dict_path)
        print(f"  - Custom generation head weights saved to: {os.path.basename(head_state_dict_path)}")
    except Exception as e:
        print(f"  - Error: Failed to save custom generation head weights: {e}")

    print(f"HuggingFace-compatible model saved: {output_dir}")
    return output_dir


# --- 6. Main training function ---
def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

       # --- Set computation data type ---
    if args.dtype == "float16":
        compute_dtype = torch.float16
    elif args.dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # --- Load model and Processor ---
    print("Loading model and Processor...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    # Load Janus model, ensure trust_remote_code=True
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype, # Use selected data type
        trust_remote_code=True,
        attn_implementation="eager" # Keep consistent
    ).to(device)
    print("Checking gen_embed and codebook size...")
    try:
        # Try to get image_token_size from config (more reliable)
        gen_embed_size = model.config.gen_head_config.params.get('image_token_size', model.gen_embed.num_embeddings) # Get from gen_head config, fallback to embedding layer
        codebook_size = None
        if hasattr(model, 'gen_vision_model') and hasattr(model.gen_vision_model, 'config') and hasattr(model.gen_vision_model.config, 'codebook_size'):
            codebook_size = model.gen_vision_model.config.codebook_size
            print("  - Obtained codebook_size from model.gen_vision_model.config")
        elif hasattr(model, 'config') and hasattr(model.config, 'gen_vision_config'):
            # Try alternative path just in case
            if hasattr(model.config.gen_vision_config, 'params') and hasattr(model.config.gen_vision_config.params, 'codebook_size'):
                 codebook_size = model.config.gen_vision_config.params.codebook_size
                 print("  - Obtained codebook_size from model.config.gen_vision_config.params")
            elif hasattr(model.config.gen_vision_config, 'codebook_size'): # Check if directly inside gen_vision_config
                 codebook_size = model.config.gen_vision_config.codebook_size
                 print("  - Obtained codebook_size from model.config.gen_vision_config")
        if codebook_size is None:
             # If still not found, print debug info
             print("  - Warning: Unable to automatically determine codebook_size. Please check model config.")
             print("    - model.config:", model.config)
             if hasattr(model, 'gen_vision_model'):
                 print("    - model.gen_vision_model.config:", getattr(model.gen_vision_model, 'config', 'N/A'))
             raise AttributeError("Unable to locate codebook_size")
        print(f"  - model.gen_embed.num_embeddings (or image_token_size from config): {gen_embed_size}")
        print(f"  - codebook_size: {codebook_size}")

    except AttributeError as e:
        print(f"  - Unable to check size: {e}")
    # --- Verify required components exist ---
    if not hasattr(model, 'gen_vision_model') or not isinstance(model.gen_vision_model, VQModel):
        raise AttributeError("Model instance does not contain 'gen_vision_model' or its type is not the expected VQModel. Please check model loading and code.")
    if not hasattr(model, 'gen_head'):
        raise AttributeError("Model instance does not contain 'gen_head'.")
    if not hasattr(model, 'prepare_gen_img_embeds'):
        raise AttributeError("Model instance does not contain 'prepare_gen_img_embeds' method.")
    # --- Verify codebook_size was successfully obtained ---
    if codebook_size is None: # Ensure above logic succeeded
         raise AttributeError("Unable to obtain codebook_size during verification step.")

    # --- Prepare VLMImageProcessor ---
    # Use Janus default or inference script values
    image_processor = VLMImageProcessor(
        image_size=args.image_size,
        image_mean=IMAGENET_INCEPTION_MEAN, # Assume using Inception normalization
        image_std=IMAGENET_INCEPTION_STD,
    )

    # --- Prepare custom head and replace ---
    print("Preparing and replacing generation head...")
    # Get hidden_size and codebook_size
    hidden_size = model.language_model.config.hidden_size
    # --- End FIX 1 ---
    hyperbolic_gen_head = HyperbolicGenHead(
        hidden_size=hidden_size,
        codebook_size=codebook_size,
        use_hyperbolic=args.use_hyperbolic,
        num_layers=args.head_num_layers
    ).to(device).to(compute_dtype) # Keep dtype consistent

    # Replace original gen_head
    model.gen_head = hyperbolic_gen_head

    # --- Freeze parameters ---
    print("Freezing original model parameters...")
    for name, param in model.named_parameters():
        if "gen_head" not in name: # Do not freeze the new gen_head
            param.requires_grad = False

    # --- Apply LoRA (optional) ---
    if args.use_lora:
        print(f"Applying LoRA to modules: {args.lora_target_modules}")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules # Use modules specified by arguments
        )
        # Apply LoRA only to language_model
        model.language_model = get_peft_model(model.language_model, config)
        model.language_model.print_trainable_parameters()
    
        # --- Prepare curvature parameter ---
    learnable_curvature = None
    if args.use_hyperbolic:
        # Initialize with a small positive value, use float32 for stability
        learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
        learnable_curvature.requires_grad = True
        print(f"Using learnable curvature (float32), initial value: {learnable_curvature.item():.4f}")

    # --- Print trainable parameters ---
    print("Trainable parameters:")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name} ({param.numel()}) - dtype: {param.dtype}")
            total_trainable_params += param.numel()
    if learnable_curvature is not None:
         print(f"- learnable_curvature ({learnable_curvature.numel()}) - dtype: {learnable_curvature.dtype}")
         total_trainable_params += learnable_curvature.numel()
    print(f"Total number of trainable parameters: {total_trainable_params / 1e6:.2f} M")


    # --- Prepare dataset and DataLoader ---
    print("Loading COCO dataset...")
    # Ensure vq_model is on the correct device and frozen
    model.gen_vision_model.to(device).eval()
    train_dataset = COCODataset(
        args.coco_data_dir, processor, model.gen_vision_model, image_processor, args.max_text_length, split='train'
    )
    # You can add validation dataset loading logic here
    # val_dataset = COCODataset(args.coco_data_dir, ..., split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    # val_loader = DataLoader(val_dataset, ...)

    # --- Prepare optimizer and scheduler ---
    print("Preparing optimizer and scheduler...")
    # --- FIX 2: Optimize original parameters ---
    # params_to_optimize_head = [p.float() for p in model.gen_head.parameters() if p.requires_grad] # Original problematic line
    params_to_optimize_head = [p for p in model.gen_head.parameters() if p.requires_grad] # Pass original parameters
    # --- End FIX 2 ---
    params_to_optimize_lora = []
    if args.use_lora:
        # LoRA parameters are usually already float32
        params_to_optimize_lora = [p for p in model.language_model.parameters() if p.requires_grad]

    optimizer_grouped_parameters = []
    if params_to_optimize_head:
        # Optimizer will handle mixed precision based on the parameter's dtype
        optimizer_grouped_parameters.append({"params": params_to_optimize_head, "lr": args.learning_rate})
    if params_to_optimize_lora:
         optimizer_grouped_parameters.append({"params": params_to_optimize_lora, "lr": args.learning_rate}) # LoRA uses the same LR

    if learnable_curvature is not None:
        # Curvature uses a separate learning rate (keep as float32 if intended)
        optimizer_grouped_parameters.append({"params": [learnable_curvature], "lr": args.curvature_lr})

    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_training_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # --- Load checkpoint (if specified) ---
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

            # Load model state
            # Use strict=False if model structure might differ, or only partial weights were saved
            # Use strict=True if the full state_dict was saved
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Warning: Missing keys when loading model state: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading model state: {unexpected_keys}")

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load GradScaler state
            if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load curvature
            if learnable_curvature is not None and 'learnable_curvature' in checkpoint and checkpoint['learnable_curvature'] is not None:
                 # Directly set Parameter data
                 with torch.no_grad():
                      learnable_curvature.data.fill_(checkpoint['learnable_curvature'])
                 print(f"Restored curvature: {learnable_curvature.item():.4f}")

            # Restore epoch and step
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']

            print(f"Successfully resumed from epoch {start_epoch}, global_step {global_step}.")

            # Optional: Verify parameter mismatch
            # loaded_args = checkpoint.get('args', {})
            # current_args = vars(args)
            # for key, value in loaded_args.items():
            #     if key != 'resume_from_checkpoint' and current_args.get(key) != value:
            #         print(f"Warning: Checkpoint argument '{key}' ({value}) does not match current argument ({current_args.get(key)})")

        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            global_step = 0
    else:
        print("No checkpoint found or not specified, starting training from scratch.")

    # --- Training loop ---
    print("Starting training...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype == torch.float16)) # Only needed for float16

    for epoch in range(args.num_epochs):
        model.train() # Ensure model is in training mode (especially for LoRA and Dropout)
        if args.use_lora:
            model.language_model.train()
        model.gen_head.train() # Ensure custom head is in training mode

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", initial=0)

        for step, batch in enumerate(progress_bar):
            if batch is None: continue # Skip failed batches

            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_visual_tokens = batch["target_visual_tokens"].to(device) # (B, num_visual_tokens)

            # --- Mixed precision training context ---
            with torch.autocast(device_type=args.device, dtype=compute_dtype, enabled=(compute_dtype != torch.float32)):
                # 1. Get text embeddings
                text_embeds = model.language_model.get_input_embeddings()(input_ids) # (B, text_seq_len, D)

                # 2. Prepare input visual Token IDs (shift right + BOS)
                # TODO: Verify start Token ID visual_bos_id
                visual_bos_id = 0 # Assuming 0, needs verification!
                visual_bos_ids = torch.full((target_visual_tokens.size(0), 1), visual_bos_id,
                                            dtype=torch.long, device=device)
                # input_visual_ids: (B, visual_seq_len)
                input_visual_ids = torch.cat([visual_bos_ids, target_visual_tokens[:, :-1]], dim=1)

                # 3. Get embeddings for input visual Tokens
                try:
                    # Use model's prepare_gen_img_embeds method
                    # visual_embeds: (B, visual_seq_len, D)
                    visual_embeds = model.prepare_gen_img_embeds(input_visual_ids)
                    # Ensure data type matches text_embeds (autocast will handle)
                    # visual_embeds = visual_embeds.to(text_embeds.dtype)
                except Exception as e:
                    print(f"Error getting visual_embeds: {e}")
                    traceback.print_exc()
                    continue # Skip this batch

                                # 4. Concatenate embedding sequences
                # combined_embeds: (B, text_len + visual_seq_len, D)
                combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)

                # 5. Build attention mask
                visual_attention_mask = torch.ones_like(input_visual_ids) # Visual part fully visible
                # combined_attention_mask: (B, text_len + visual_seq_len)
                combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

                # --- Forward pass ---
                outputs = model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True # Ensure hidden states are returned
                )
                # hidden_states: (B, text_len + visual_seq_len, D)
                hidden_states = outputs.hidden_states[-1]

                # Take only the visual part of hidden_states for prediction
                # Visual part starts from text_len
                # visual_hidden_states: (B, visual_seq_len, D)
                visual_hidden_states = hidden_states[:, input_ids.shape[1]:, :]

                # Use custom head for prediction
                # Convert curvature to current computation dtype
                current_curvature = learnable_curvature.to(visual_hidden_states.dtype) if learnable_curvature is not None else None
                # logits: (B, visual_seq_len, codebook_size)
                logits = model.gen_head(visual_hidden_states, current_curvature)

                # --- Compute loss ---
                # Ensure logits are float32 for loss calculation
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.float().reshape(-1, codebook_size),
                                target_visual_tokens.reshape(-1))

            # --- Backpropagation ---
            loss_value = loss.item() # Get loss value
            loss = loss / args.gradient_accumulation_steps

            # Use GradScaler for scaling
            scaler.scale(loss).backward()

            epoch_loss += loss_value # Accumulate unscaled loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                 # Clip gradients before unscale (optional but recommended)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_norm=1.0 # Can adjust max_norm if needed
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True) # Use set_to_none for efficiency
                global_step += 1

                # Update progress bar
                lr_value = scheduler.get_last_lr()[0]
                curve_value = learnable_curvature.item() if learnable_curvature is not None else float('nan')
                progress_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{lr_value:.2e}",
                    "curve": f"{curve_value:.3f}" if not math.isnan(curve_value) else "N/A",
                    "scale": f"{scaler.get_scale():.1f}" if compute_dtype == torch.float16 else "N/A"
                })

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # --- Validation (optional) ---
        if (epoch + 1) % args.eval_interval == 0:
             avg_val_loss = evaluate(...)
             print(f"Epoch {epoch+1} validation average loss: {avg_val_loss:.4f}")
             # Optionally save the best model based on avg_val_loss

                # --- Save the model ---

        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}") # Use epoch number to name the directory
        save_model_for_hf(
            model=model,
            hyperbolic_gen_head=hyperbolic_gen_head,
            learnable_curvature=learnable_curvature if args.use_hyperbolic else torch.tensor(0.0),
            output_dir=epoch_output_dir, # Use specific directory for this epoch
            processor=processor,
            args=args,
            hidden_size=hidden_size,
            codebook_size=codebook_size
        )

        checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pt") # Save to a fixed file name or epoch-based file name
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True) # Ensure directory exists

        # Prepare the state to save
        # Note: For LoRA, we usually only save adapter weights, but to restore optimizer state,
        # saving the entire model.state_dict() is simpler, or you can choose to save only trainable parameters.
        # Here, we save the entire model state. When loading, you may need to handle non-strict matching if the model structure changes.
        model_state_to_save = model.state_dict()
        # If you only want to save trainable parameters, you can do:
        # model_state_to_save = {
        #     name: param.cpu().clone() for name, param in model.named_parameters() if param.requires_grad
        # }

        checkpoint_data = {
            'epoch': epoch + 1, # Save the *next* epoch to start from
            'global_step': global_step,
            'model_state_dict': model_state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'learnable_curvature': learnable_curvature.item() if learnable_curvature is not None else None,
            'args': vars(args) # Save command-line arguments for reference
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

        # --- Validation (optional) ---
        # evaluate(...)

if __name__ == "__main__":
    # --- Add this before calling main ---
    try:
        # Set the start method *before* any CUDA calls or DataLoader creation
        import platform
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # Might happen if it was already set or in certain environments
        print(f"Note: Could not set start method to 'spawn': {e}")
    # --- End of addition ---
    main()
