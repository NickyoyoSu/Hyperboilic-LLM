import os
import sys
import torch
import argparse
from PIL import Image
import requests
from io import BytesIO
import torch.nn as nn
from transformers import AutoProcessor, MllamaForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz

# Argument parser
parser = argparse.ArgumentParser(description="Perform multimodal inference using a trained checkpoint")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (.pth)")
parser.add_argument("--lora_dir", type=str, default="", help="Path to the LoRA directory (if using LoRA)")
parser.add_argument("--image", type=str, default="", help="Path or URL to the input image")
parser.add_argument("--text", type=str, default="", help="Prompt text for generating an image")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--image_size", type=int, default=224, help="Input image size")
args = parser.parse_args()

# Device setup
device = torch.device(args.device)

# Custom multimodal mapping head
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=3):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        # Get config
        if isinstance(base_model, nn.DataParallel):
            config = base_model.module.config
        else:
            config = base_model.config
        
        # Get vocab_size
        if hasattr(config, 'text_config'):
            self.vocab_size = config.text_config.vocab_size
        elif hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            self.vocab_size = 32000  # Default
        
        # Get hidden_size
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        elif hasattr(config, 'text_config'):
            self.hidden_size = config.text_config.hidden_size
        else:
            self.hidden_size = 4096  # Default
        
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # Multimodal adapter
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Linear layers and normalization
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        if num_layers >= 2:
            self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(self.hidden_size)
        
        if num_layers >= 3:
            self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(self.hidden_size)

        # Classifiers
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # Multimodal adaptation
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
        
        # Choose classifier based on geometry
        if self.use_hyperbolic:
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

def load_model():
    """Load the MLLama-3.2-Vision model"""
    print("Loading MLLama-3.2-Vision model...")
    hf_token = "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"
    model_name = "meta-llama/Llama-3.2-11B-Vision"
    
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    
    try:
        # Load MLLama model
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name, 
            token=hf_token,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"Error: Failed to load the original model {model_name}")
        print(f"Details: {e}")
        print("Please ensure you have sufficient GPU memory and a properly configured environment.")
        sys.exit(1)
    
    # Load Stable Diffusion for image generation
    print("Loading Stable Diffusion model...")
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
        )
    ).to(device)
    
    return processor, model, image_gen_model

def load_checkpoint(checkpoint_path, model, lora_dir=""):
    """Load the trained checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a hyperbolic model
    is_hyperbolic = "hyp" in os.path.basename(checkpoint_path)
    
    # Create and load custom head
    custom_lm_head = MultimodalMappingHead(model, use_hyperbolic=is_hyperbolic, num_layers=3).to(device)
    custom_lm_head.load_state_dict(checkpoint["lm_head_state"])
    
    # Load curvature parameter
    curvature = torch.tensor(checkpoint["curvature"], device=device)
    
    # Load LoRA weights (if specified)
    if lora_dir and os.path.exists(lora_dir):
        print(f"Loading LoRA weights: {lora_dir}")
        if isinstance(model, nn.DataParallel):
            model.module.load_adapter(lora_dir)
        else:
            model.load_adapter(lora_dir)
    elif "lora" in checkpoint_path.lower():
        # Try to locate LoRA in the checkpoint path
        parent_dir = os.path.dirname(checkpoint_path)
        lora_dirs = [d for d in os.listdir(parent_dir) if d.startswith("lora_")]
        if lora_dirs:
            newest_lora = sorted(lora_dirs)[-1]
            lora_path = os.path.join(parent_dir, newest_lora)
            print(f"Automatically loading LoRA weights: {lora_path}")
            if isinstance(model, nn.DataParallel):
                model.module.load_adapter(lora_path)
            else:
                model.load_adapter(lora_path)
    
    return custom_lm_head, curvature, is_hyperbolic

def generate_text_from_image(image_path, model, processor, custom_lm_head, curvature, max_len=100):
    """Generate text from image"""
    model.eval()
    custom_lm_head.eval()
    
    # Load image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Process image
    inputs = processor(images=image, text="Describe this image:", return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode output
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text

def generate_image_from_text(text_prompt, image_gen_model, output_path=None):
    """Generate image from text"""
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                      dtype=torch.float16):
        image = image_gen_model(text_prompt, guidance_scale=7.5).images[0]
    
    if output_path:
        image.save(output_path)
    
    return image

# Main function
def main():
    # Load models
    processor, model, image_gen_model = load_model()
    
    # Load checkpoint
    custom_lm_head, curvature, is_hyperbolic = load_checkpoint(
        args.checkpoint, model, args.lora_dir
    )
    
    print(f"{'='*50}")
    print(f"Model loaded - Geometry: {'Hyperbolic' if is_hyperbolic else 'Euclidean'}, Curvature: {curvature.item():.4f}")
    print(f"{'='*50}")
    
    # Image-to-text
    if args.image:
        print(f"\n=== Generating text from image ===")
        generated_text = generate_text_from_image(
            args.image, model, processor, custom_lm_head, curvature
        )
        print(f"Image path: {args.image}")
        print(f"Generated description: {generated_text}")
    
    # Text-to-image
    if args.text:
        print(f"\n=== Generating image from text ===")
        output_path = "generated_image.jpg"
        generate_image_from_text(args.text, image_gen_model, output_path)
        print(f"Prompt text: {args.text}")
        print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    main()
