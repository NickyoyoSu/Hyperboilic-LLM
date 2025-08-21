import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
import os
import time
import random
import json
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from PIL import ImageFile
import torch.nn.functional as F
import kornia
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from models.vqvae_hyp import VQVAE_HYP as VQVAE 
from geoopt.optim import RiemannianAdam
from geoopt.manifolds import Lorentz as GeooptLorentz

# import helper function
from utils_hyp import (
    VGGPerceptualLoss,
    calc_loss_with_regularization,
    visualize_reconstructions,
    plot_cumulative_losses,
    analyze_embeddings,
    monitor_gradients,
    monitor_model_weights,
    plot_codebook_usage,
    plot_latent_space_distribution,
    plot_angular_distribution_at_radius
)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced VQVAE training")
    
    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=128, help="effective batch size")
    parser.add_argument("--n_updates", type=int, default=4000080)
    parser.add_argument("--learning_rate", type=float, default=3e-4) # slightly adjusted LR
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log_interval", type=int, default=100) # log interval
    parser.add_argument("--val_interval", type=int, default=2000, help="validation interval")
    parser.add_argument("--dataset", type=str, default='IMAGENET')
    
    # Model parameters
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--n_embeddings", type=int, default=16384)
    parser.add_argument("--beta", type=float, default=1, help="commitment loss weight")
    
    # Regularization & stability
    parser.add_argument("--weight_decay", type=float, default=1e-5) # reduced weight decay
    parser.add_argument("--grad_clip_value", type=float, default=3.0)
    
    # LR scheduling
    parser.add_argument("--use_lr_scheduler", action="store_true", default=True) # enabled by default
    parser.add_argument("--lr_scheduler", type=str, default="plateau", choices=["cosine", "plateau"])
    parser.add_argument("--min_lr", type=float, default=1e-6) # min learning rate
    
    # Mixed precision
    parser.add_argument("--fp16", action="store_true")
    
    # Resume & save
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./new_results")
    
    # Hyperbolic-specific
    parser.add_argument('--adaptive_c', action='store_true')
    parser.add_argument('--initial_c', type=float, default=1.0)
    
    # Loss/training params (borrowed from AE)
    parser.add_argument('--pretrained_path', type=str, default=None)
    # Key fix: add loss weights and increase default weights for perceptual and color losses
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="reconstruction loss weight")
    parser.add_argument("--lambda_p", type=float, default=1.0, help="perceptual loss weight (increased from 0.1)")
    parser.add_argument("--lambda_color", type=float, default=0.5, help="color loss weight (increased from 0.1)")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="codebook diversity loss weight")
    parser.add_argument("--reg_warmup_steps", type=int, default=10000, help="warm-up steps for regularization")
    parser.add_argument("--final_lambda_reg", type=float, default=0.005, help="final regularization weight")
    parser.add_argument("--visualization_interval", type=int, default=2000, help="visualization interval")
    parser.add_argument("--monitor_interval", type=int, default=100, help="monitor interval")
    # New: codebook regularization
    parser.add_argument("--lambda_cb_reg", type=float, default=1.0, help="codebook geometric regularization loss weight")
    parser.add_argument("--target_cb_radius", type=float, default=5.0, help="target average radius for codebook")
    # Fix: add missing CLI args
    parser.add_argument("--quantizer_warmup_steps", type=int, default=5000, help="number of steps to train decoder with z_e before switching to z_q.")
    parser.add_argument("--reset_codes_interval", type=int, default=2500, help="reset dead codes every N steps; 0 disables")
    
    return parser.parse_args()

def calc_codebook_regularization_loss(codebook_vectors, manifold, target_mean_radius, lambda_std=1.0, lambda_mean=1.0):
    """
    Compute geometric regularization loss for the codebook itself.
    Encourage well-spread codebook vectors in hyperbolic space.
    """
    if codebook_vectors.numel() == 0:
        return torch.tensor(0.0, device=codebook_vectors.device), {}

    radii = manifold.dist0(codebook_vectors) # [n_embeddings]
    
    # 1) Maximize radius std (encourage spread). We minimize its negative.
    radius_std_reg = -lambda_std * torch.std(radii)

    # 2) Penalize mean radius (encourage moving away from origin toward target radius)
    mean_radius_reg = lambda_mean * F.mse_loss(radii.mean(), torch.tensor(target_mean_radius, device=radii.device))
    
    total_reg_loss = radius_std_reg + mean_radius_reg
    
    loss_dict = {
        'cb_radius_std': radius_std_reg.item(),
        'cb_mean_radius': mean_radius_reg.item()
    }
    return total_reg_loss, loss_dict

def train(args):
    set_seed(args.seed)

    gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16' if args.fp16 else 'no'
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"config_{utils.readable_timestamp()}.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
    
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using device: {device}")
    
    # Load data
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)

    # Initialize model (no .to(device) here)
    model = VQVAE(
        h_dim=args.n_hiddens,
        n_res_layers=args.n_residual_layers,
        n_embeddings=args.n_embeddings,
        beta=args.beta,
        c=args.initial_c,
        adaptive_c=args.adaptive_c
    )
    # Pass quantizer warmup steps to model (support decoder bypass)
    if hasattr(model, 'quantizer_warmup_steps'):
        model.quantizer_warmup_steps = max(0, int(args.quantizer_warmup_steps))
    
    # Load pretrained weights
    if args.pretrained_path:
        if accelerator.is_main_process:
            print(f"Loading pretrained AE weights from: {args.pretrained_path}")
        # All processes need to load weights; only main prints
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.encoder.load_state_dict(checkpoint['encoder'], strict=False)
        model.decoder.load_state_dict(checkpoint['decoder'], strict=False)
        if args.adaptive_c and 'curvature' in checkpoint:
            if isinstance(checkpoint['curvature'], dict):
                model.curvature.load_state_dict(checkpoint['curvature'])
            else:
                c_val = checkpoint['curvature']
                model.manifold.c.data.fill_(c_val)
                model.encoder.manifold.c.data.fill_(c_val)
                model.decoder.manifold.c.data.fill_(c_val)
                model.vq.manifold.c.data.fill_(c_val)
        if accelerator.is_main_process:
            print("Pretrained weights loaded!")
    
    optimizer = RiemannianAdam(
        model.parameters(), 
        lr=args.learning_rate, 
        stabilize=10,
        weight_decay=args.weight_decay,
    )
    
    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )
    
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=args.min_lr)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=args.n_updates, eta_min=args.min_lr)

    start_step = 0
    best_val_loss = float('inf')
    
    # Loss functions & history (borrowed from AE)
    perceptual_loss = VGGPerceptualLoss().to(device)
    history = {
        'total': [], 'recon': [], 'perceptual': [], 'color': [], 'reg': [], 'embed': [], 'perplexity': [], 'diversity': [], 'cb_reg': []
    }
    
    if args.resume and args.checkpoint:
        # Simplified resume logic
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint.get('step', 0) + 1
        if accelerator.is_main_process:
            print(f"Successfully resumed at step {start_step}")
    
    model.train()
    if accelerator.is_main_process:
        print(f"Start training, total {args.n_updates} steps...")
    
    train_iterator = iter(training_loader)
    
    for step in range(start_step, args.n_updates):
        try:
            x, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(training_loader)
            x, _ = next(train_iterator)
        
        # Periodically clear cache
        if step % 500 == 0:
            torch.cuda.empty_cache()
        
        x = x.to(device)
        
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # Decide whether to reset dead codes at this step
            reset_dead_codes_this_step = (
                args.reset_codes_interval > 0 and 
                step > 0 and 
                step % args.reset_codes_interval == 0
            )

            embedding_loss, x_hat, perplexity, diversity_loss, z_e_hyp, codebook_usage = model(
                x, 
                global_step=step, 
                reset_dead_codes=reset_dead_codes_this_step
            )
            
            # --- compute all losses using helper (borrowed from AE) ---
            total_loss, loss_dict = calc_loss_with_regularization(
                x, x_hat, z_e_hyp, embedding_loss, diversity_loss, perceptual_loss, args, step
            )
            
            # New: add codebook regularization loss
            unwrapped_model = accelerator.unwrap_model(model)
            # be robust if some quantizer impls don't expose embedding weights
            cb_reg_loss = torch.tensor(0.0, device=device)
            cb_reg_loss_dict = {}
            vq = getattr(unwrapped_model, 'vq', None)
            embedding_weight = getattr(vq, 'embedding', None)
            embedding_weight = getattr(embedding_weight, 'weight', None) if embedding_weight is not None else None
            if embedding_weight is not None:
                cb_reg_loss, cb_reg_loss_dict = calc_codebook_regularization_loss(
                    embedding_weight,
                    unwrapped_model.manifold,
                    args.target_cb_radius
                )
            total_loss = total_loss + args.lambda_cb_reg * cb_reg_loss
            loss_dict.update(cb_reg_loss_dict) # merge codebook reg items
            
            accelerator.backward(total_loss)
            
            if args.grad_clip_value > 0:
                # Clip only when gradients are synced (i.e., at the end of accumulation)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            
            optimizer.step()

        # --- logging / monitoring / visualization ---
        if accelerator.is_main_process:
            # update history dict with loss_dict
            for key, value in loss_dict.items():
                if key in history:
                    history[key].append(value)
            history['perplexity'].append(perplexity.item())
            # history['cb_reg'] handled above
            
            if step > 0 and step % args.log_interval == 0:
                # when warmup, codebook_usage may be a scalar placeholder
                if codebook_usage.dim() == 0:
                    usage_percent = 0.0
                else:
                    used_codes = torch.sum(codebook_usage > 1e-7).item()
                    total_codes = codebook_usage.numel()
                    usage_percent = (used_codes / total_codes) * 100

                log_str = f"\n[Step {step}/{args.n_updates}] | "
                for key, value in loss_dict.items():
                    log_str += f"{key.capitalize()}: {value:.4f} | "
                log_str += f"Perplexity: {perplexity.item():.2f} | "
                log_str += f"CB_Reg: {loss_dict.get('cb_radius_std', 0) + loss_dict.get('cb_mean_radius', 0):.4f} | "
                log_str += f"Usage: {usage_percent:.2f}% | "
                log_str += f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                print(log_str)
            
            if step > 0 and step % args.monitor_interval == 0:
                monitor_gradients(accelerator.unwrap_model(model), step)

            if step > 0 and step % args.visualization_interval == 0 or step == 100 or step == 200 or step == 300 or step == 400 or step == 500 or step == 600 or step == 700 or step == 800 or step == 900 or step == 1000:
                print("--- Generating visualization ---")
                unwrapped_model = accelerator.unwrap_model(model)
                plot_cumulative_losses(history, args.output_dir)
                visualize_reconstructions(unwrapped_model, validation_loader, device, step, args.output_dir)
                analyze_embeddings(unwrapped_model, x, step, args.output_dir)
                plot_codebook_usage(codebook_usage, step, args.output_dir)
                plot_latent_space_distribution(unwrapped_model, z_e_hyp, step, args.output_dir)
                plot_angular_distribution_at_radius(unwrapped_model, z_e_hyp, step, args.output_dir)

        # --- validation loop ---
        if step > 0 and step % args.val_interval == 0:
            model.eval()
            val_losses = {'total': [], 'recon': [], 'embed': []}
            val_perplexities = []
            print("\n--- Start validation ---")
            with torch.no_grad():
                for i, (val_x, _) in enumerate(validation_loader):
                    if i >= 50: break # limit samples to speed up
                    val_x = val_x.to(device)
                    val_embed_loss, val_x_hat, val_perplexity, _, _, _ = model(val_x)
                    val_recon_loss = F.mse_loss(val_x_hat, val_x)

                    # Simplified total loss during validation
                    val_total_loss = val_recon_loss + val_embed_loss 
                    
                    val_losses['total'].append(val_total_loss.item())
                    val_losses['recon'].append(val_recon_loss.item())
                    val_losses['embed'].append(val_embed_loss.item())
                    val_perplexities.append(val_perplexity.item())
            
            avg_val_loss = np.mean(val_losses['total'])
            avg_perplexity = np.mean(val_perplexities)
            print(f"--- Validation done (Step {step}) ---")
            print(f"Avg Total Loss: {avg_val_loss:.6f} | "
                  f"Avg Recon: {np.mean(val_losses['recon']):.6f} | "
                  f"Avg Embed: {np.mean(val_losses['embed']):.6f} | "
                  f"Avg Perplexity: {avg_perplexity:.2f}")
            
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if args.save:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_path = os.path.join(args.output_dir, 'best_model.pth')
                    torch.save({'model': unwrapped_model.state_dict(), 'step': step, 'loss': best_val_loss}, save_path)
                    if accelerator.is_main_process:
                        print(f"*** Saved new best model to: {save_path} ***")
            model.train()
        
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        if args.save and step > 0 and step % args.save_interval == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f'checkpoint_{step}.pth')
            torch.save({'model': unwrapped_model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, save_path)
            if accelerator.is_main_process:
                print(f"--- Saved checkpoint to: {save_path} ---")
            
    print("Training finished!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
