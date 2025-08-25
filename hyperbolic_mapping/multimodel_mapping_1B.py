import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, PreTrainedModel
from janus.models import VLChatProcessor, MultiModalityCausalLM  # Ensure MultiModalityCausalLM is imported
from janus.models.vq_model import VQModel, ModelArgs  # Import VQModel from Janus
from janus.models.image_processing_vlm import VLMImageProcessor, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD  # Import from Janus
from PIL import Image
from tqdm import tqdm
import math
import random
import numpy as np
from peft import LoraConfig, get_peft_model
import json
import traceback
from pycocotools.coco import COCO  # To load COCO dataset

# Import HyperLib components
from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz

# --- 1. Argument parsing ---
def get_parser():
    parser = argparse.ArgumentParser(description="Train Text-to-Image with Hyperbolic Mapping Head using COCO")
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
    # Add LoRA target module parameter
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
        self.codebook_size = codebook_size  # visual codebook size (e.g., 16384)
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Similar to image-to-text head, but final output dim is codebook_size
        # Note: no multimodal_adapter here; start directly from hidden_states
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        if num_layers >= 2:
            self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(hidden_size)
        if num_layers >= 3:
            self.linear3 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(hidden_size)
        # Add more layers if needed

        # Classifiers
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=hidden_size + 1,  # Lorentz requires time component
            num_classes=self.codebook_size
        )
        self.euc_cls = nn.Linear(hidden_size, self.codebook_size, bias=False)

        print(f"Initialized {'hyperbolic' if use_hyperbolic else 'euclidean'} generation head (GenHead), output dim: {self.codebook_size}, layers: {self.num_layers}")

    def lorentz_map(self, x, c_param):
        k_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
        if k_tensor.dim() == 0:
            k_tensor = k_tensor.unsqueeze(0)
        return expmap0(x, k=k_tensor, dim=-1)

    def forward(self, hidden_states, c_param=None):
        target_dtype = self.linear1.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        x = hidden_states  # use hidden_states directly

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
        # Add more layers if needed

        if self.use_hyperbolic:
            if c_param is None:
                raise ValueError("Curvature parameter 'c_param' is required for hyperbolic head.")
            x = self.manifold.add_time(x)
            c_param_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
            hyper_embs = self.lorentz_map(x, c_param_tensor)
            logits = self.hyp_cls(hyper_embs)
        else:
            if x.dtype != self.euc_cls.weight.dtype:
                x = x.to(self.euc_cls.weight.dtype)
            logits = self.euc_cls(x)

        return logits  # logits over visual tokens

# --- 4. COCO dataset definition ---
class COCODataset(Dataset):
    def __init__(self, data_dir, processor, vq_model, image_processor, max_text_length, split='train'):
        self.data_dir = data_dir
        self.processor = processor
        self.vq_model = vq_model  # pass VQModel instance (should be eval and frozen)
        self.image_processor = image_processor  # pass VLMImageProcessor instance
        self.max_text_length = max_text_length
        self.split = split

        # Set COCO paths
        self.img_dir = os.path.join(data_dir, f'{split}2017')
        self.ann_file = os.path.join(data_dir, 'annotations', f'captions_{split}2017.json')

        if not os.path.exists(self.img_dir) or not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"COCO dataset files not found in {data_dir}. Please ensure annotations/ and {split}2017/ exist.")

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

        # 1) Process text
        # Use DeepSeek-style template to format input text
        # Note: assume we only use one user turn as input
        conv = self.processor.new_chat_template()  # get dialog template
        conv.append_message(conv.roles[0], caption)  # add user message
        conv.append_message(conv.roles[1], None)  # add empty assistant turn to trigger generation format
        text_prompt = conv.get_prompt()  # get formatted text

        text_inputs = self.processor.tokenizer(
            text_prompt,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.squeeze(0)  # (seq_len)
        attention_mask = text_inputs.attention_mask.squeeze(0)  # (seq_len)

        # 2) Process image and get target visual tokens
        try:
            image = Image.open(img_path).convert('RGB')
            # Use VLMImageProcessor to process image
            image_inputs = self.image_processor.preprocess([image], return_tensors="pt")
            pixel_values = image_inputs['pixel_values']  # (1, 3, H, W)

            # Get target visual tokens via VQModel (no_grad)
            with torch.no_grad():
                # Move pixel_values to the same device/dtype as vq_model
                vq_device = next(self.vq_model.parameters()).device
                vq_dtype = next(self.vq_model.parameters()).dtype
                pixel_values = pixel_values.to(vq_device, vq_dtype)

                # Call encode to get visual tokens
                _, _, info = self.vq_model.encode(pixel_values)
                # info = (perplexity, min_encodings, min_encoding_indices)
                target_visual_tokens = info[2].squeeze(0)  # (latent_h * latent_w) or (num_tokens)

        except Exception as e:
            print(f"Error processing image {img_path} or obtaining visual tokens: {e}")
            traceback.print_exc()
            # Return None and let collate_fn drop the sample
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_visual_tokens": target_visual_tokens,  # (num_visual_tokens)
        }

# --- 5. Collate function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # filter failed samples
    if not batch:
        return None

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # target_visual_tokens should have consistent length
    target_visual_tokens = torch.stack([item["target_visual_tokens"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_visual_tokens": target_visual_tokens,
    }

# --- 6. Main training function ---
def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Set compute dtype ---
    if args.dtype == "float16":
        compute_dtype = torch.float16
    elif args.dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # --- Load model and processor ---
    print("Loading model and processor...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    # Load Janus model, ensure trust_remote_code=True
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype,  # use chosen dtype
        trust_remote_code=True,
        attn_implementation="eager"  # keep consistent
    ).to(device)

    print("Checking gen_embed and codebook sizes...")
    try:
        # Try to get image_token_size from config (more reliable)
        gen_embed_size = model.config.gen_head_config.params.get('image_token_size', model.gen_embed.num_embeddings)  # from gen_head config or fallback to embedding layer
        codebook_size = None
        if hasattr(model, 'gen_vision_model') and hasattr(model.gen_vision_model, 'config') and hasattr(model.gen_vision_model.config, 'codebook_size'):
            codebook_size = model.gen_vision_model.config.codebook_size
            print("  - Retrieved codebook_size from model.gen_vision_model.config")
        elif hasattr(model, 'config') and hasattr(model.config, 'gen_vision_config'):
            # Try the original path just in case
            if hasattr(model.config.gen_vision_config, 'params') and hasattr(model.config.gen_vision_config.params, 'codebook_size'):
                codebook_size = model.config.gen_vision_config.params.codebook_size
                print("  - Retrieved codebook_size from model.config.gen_vision_config.params")
            elif hasattr(model.config.gen_vision_config, 'codebook_size'):  # check if directly present
                codebook_size = model.config.gen_vision_config.codebook_size
                print("  - Retrieved codebook_size from model.config.gen_vision_config")
        if codebook_size is None:
            # Still not found — add debug info
            print("  - Warning: Unable to automatically determine codebook_size. Please check model config.")
            print("    - model.config:", model.config)
            if hasattr(model, 'gen_vision_model'):
                print("    - model.gen_vision_model.config:", getattr(model.gen_vision_model, 'config', 'N/A'))
            raise AttributeError("Could not find codebook_size")
        print(f"  - model.gen_embed.num_embeddings (or image_token_size from config): {gen_embed_size}")
        print(f"  - codebook_size: {codebook_size}")

    except AttributeError as e:
        print(f"  - Could not inspect sizes: {e}")

    # --- Sanity check required components ---
    if not hasattr(model, 'gen_vision_model') or not isinstance(model.gen_vision_model, VQModel):
        raise AttributeError("Could not find 'gen_vision_model' in the model instance, or its type is not VQModel. Please verify model loading and code.")
    if not hasattr(model, 'gen_head'):
        raise AttributeError("Could not find 'gen_head' in the model instance.")
    if not hasattr(model, 'prepare_gen_img_embeds'):
        raise AttributeError("Could not find 'prepare_gen_img_embeds' method in the model instance.")
    # --- Ensure codebook_size was obtained ---
    if codebook_size is None:
        raise AttributeError("Failed to obtain codebook_size during validation.")

    # --- Prepare VLMImageProcessor ---
    # Use Janus defaults (Inception normalization)
    image_processor = VLMImageProcessor(
        image_size=args.image_size,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
    )

    # --- Prepare and replace custom head ---
    print("Preparing and replacing generation head...")
    # Get hidden_size and codebook_size
    hidden_size = model.language_model.config.hidden_size
    hyperbolic_gen_head = HyperbolicGenHead(
        hidden_size=hidden_size,
        codebook_size=codebook_size,
        use_hyperbolic=args.use_hyperbolic,
        num_layers=args.head_num_layers
    ).to(device).to(compute_dtype)  # keep dtype consistent

    # Replace original gen_head
    model.gen_head = hyperbolic_gen_head

    # --- Freeze parameters ---
    print("Freezing original model parameters...")
    for name, param in model.named_parameters():
        if "gen_head" not in name:  # do not freeze the new gen_head
            param.requires_grad = False

    # --- Apply LoRA (optional) ---
    if args.use_lora:
        print(f"Applying LoRA to modules: {args.lora_target_modules}")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules  # use user-specified target modules
        )
        # Apply LoRA only to language_model
        model.language_model = get_peft_model(model.language_model, config)
        model.language_model.print_trainable_parameters()

    # --- Prepare curvature parameter ---
    learnable_curvature = None
    if args.use_hyperbolic:
        # Initialize to a small positive value; use float32 for stability
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
    print(f"Total trainable params: {total_trainable_params / 1e6:.2f} M")

    # --- Prepare dataset and DataLoader ---
    print("Loading COCO dataset...")
    # Ensure vq_model is on the correct device and in eval mode
    model.gen_vision_model.to(device).eval()
    train_dataset = COCODataset(
        args.coco_data_dir, processor, model.gen_vision_model, image_processor, args.max_text_length, split='train'
    )
    # You can add validation dataset loading here as needed
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
    # Optimize original parameters (do not cast to float() here; keep original dtypes)
    params_to_optimize_head = [p for p in model.gen_head.parameters() if p.requires_grad]
    params_to_optimize_lora = []
    if args.use_lora:
        # LoRA params are typically float32 already
        params_to_optimize_lora = [p for p in model.language_model.parameters() if p.requires_grad]

    optimizer_grouped_parameters = []
    if params_to_optimize_head:
        # Optimizer will handle mixed precision based on each param's dtype
        optimizer_grouped_parameters.append({"params": params_to_optimize_head, "lr": args.learning_rate})
    if params_to_optimize_lora:
        optimizer_grouped_parameters.append({"params": params_to_optimize_lora, "lr": args.learning_rate})  # use same LR for LoRA

    if learnable_curvature is not None:
        # Curvature uses its own LR (keep as float32 if intended)
        optimizer_grouped_parameters.append({"params": [learnable_curvature], "lr": args.curvature_lr})

    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_training_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # --- Training loop ---
    print("Starting training...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype == torch.float16))  # scaler is only needed for float16

    for epoch in range(args.num_epochs):
        model.train()  # ensure train mode (especially for LoRA & dropout)
        if args.use_lora:
            model.language_model.train()
        model.gen_head.train()  # ensure custom head is in train mode

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue  # skip failed batches

            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_visual_tokens = batch["target_visual_tokens"].to(device)  # (B, num_visual_tokens)

            # --- Mixed precision context ---
            with torch.autocast(device_type=args.device, dtype=compute_dtype, enabled=(compute_dtype != torch.float32)):
                # 1) Text embeddings
                text_embeds = model.language_model.get_input_embeddings()(input_ids)  # (B, text_seq_len, D)

                # 2) Prepare input visual token IDs (shift-right + BOS)
                # TODO: confirm the start token id for visual sequence: visual_bos_id
                visual_bos_id = 0  # assume 0 for now; verify!
                visual_bos_ids = torch.full((target_visual_tokens.size(0), 1), visual_bos_id,
                                            dtype=torch.long, device=device)
                # input_visual_ids: (B, visual_seq_len)
                input_visual_ids = torch.cat([visual_bos_ids, target_visual_tokens[:, :-1]], dim=1)

                # 3) Get embeddings for input visual tokens
                try:
                    # Use model-provided prepare_gen_img_embeds
                    # visual_embeds: (B, visual_seq_len, D)
                    visual_embeds = model.prepare_gen_img_embeds(input_visual_ids)
                    # Autocast will keep dtype consistent with text_embeds
                except Exception as e:
                    print(f"Error obtaining visual_embeds: {e}")
                    traceback.print_exc()
                    continue  # skip this batch

                # 4) Concatenate embedding sequences
                # combined_embeds: (B, text_len + visual_seq_len, D)
                combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)

                # 5) Build attention mask
                visual_attention_mask = torch.ones_like(input_visual_ids)  # visual part is fully visible
                # combined_attention_mask: (B, text_len + visual_seq_len)
                combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

                # --- Forward pass through language model ---
                outputs = model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True  # ensure hidden states are returned
                )
                # hidden_states: (B, text_len + visual_seq_len, D)
                hidden_states = outputs.hidden_states[-1]

                # Only take the visual portion to predict next visual tokens
                # visual part starts at text_len
                # visual_hidden_states: (B, visual_seq_len, D)
                visual_hidden_states = hidden_states[:, input_ids.shape[1]:, :]

                # Use custom head
                current_curvature = learnable_curvature.to(visual_hidden_states.dtype) if learnable_curvature is not None else None
                # logits: (B, visual_seq_len, codebook_size)
                logits = model.gen_head(visual_hidden_states, current_curvature)

                # --- Loss ---
                # Ensure float32 for loss computation
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.float().reshape(-1, codebook_size),
                                target_visual_tokens.reshape(-1))

            # --- Backprop ---
            loss_value = loss.item()
            loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            epoch_loss += loss_value

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip grads before unscale (recommended)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_norm=1.0  # can be tuned
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # more efficient
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

        # --- End of epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # --- Save model ---
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)

            # Save custom head and config
            head_state_dict = model.gen_head.state_dict()
            # Cast params back to float32 for saving
            for k, v in head_state_dict.items():
                head_state_dict[k] = v.float()

            save_data = {
                'gen_head_state': head_state_dict,
                'use_hyperbolic': args.use_hyperbolic,
                'codebook_size': codebook_size,
                'hidden_size': hidden_size,
                'num_layers': args.head_num_layers,
                'curvature': learnable_curvature.item() if learnable_curvature is not None else None
            }
            torch.save(save_data, os.path.join(save_path, "hyperbolic_gen_head.bin"))

            # Save LoRA weights
            if args.use_lora:
                model.language_model.save_pretrained(os.path.join(save_path, "lora_adapters"))

            # Save tokenizer/processor configs (optional but recommended)
            processor.save_pretrained(save_path)

            print(f"Model components saved to: {save_path}")

        # --- Validation (optional) ---
        # evaluate(...)

if __name__ == "__main__":
    # Set start method before any CUDA calls or DataLoader creation
    try:
        import platform
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # May happen if already set or in certain environments
        print(f"Note: Could not set start method to 'spawn': {e}")
    main()
