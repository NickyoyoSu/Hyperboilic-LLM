import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import utils
import torchvision.models as models
from models.AE import StandardAutoencoder
from geoopt.optim import RiemannianSGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Import Accelerate library
from accelerate import Accelerator
from accelerate.utils import set_seed
from matplotlib.lines import Line2D
import kornia

# Define VGGPerceptualLoss after imports and before first function
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Freeze all VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract features from the following VGG layers
        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:16]  # relu3_3
        self.slice4 = vgg[16:23] # relu4_3
        
        # Normalization parameters for ImageNet pretrained models
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        # Normalize inputs and targets to match VGG expected inputs
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        # Extract features
        def get_feats(x):
            h = self.slice1(x)
            h_relu1_2 = h
            h = self.slice2(h)
            h_relu2_2 = h
            h = self.slice3(h)
            h_relu3_3 = h
            h = self.slice4(h)
            h_relu4_3 = h
            return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

        feats_input = get_feats(input)
        feats_target = get_feats(target)

        loss = 0.0
        for f_in, f_t in zip(feats_input, feats_target):
            loss = loss + F.l1_loss(f_in, f_t) # Use L1 loss over feature differences
            
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Hypersurface autoencoder pretraining (Accelerate)")
    
    # Basic training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--effective_batch_size", type=int, default=256, 
                        help="Effective batch size via gradient accumulation")
    parser.add_argument("--epochs", type=int, default=30)  
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='IMAGENET')
    
    # Model args
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--n_res_layers", type=int, default=2)
    parser.add_argument("--lambda_p", type=float, default=0.1, help="Weight of perceptual loss")
    parser.add_argument("--reg_warmup_batches", type=int, default=10000, help="Warmup batches for geometric regularization")
    
    # Misc args
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--adaptive_c", action="store_true", help="Use adaptive curvature")
    parser.add_argument("--initial_c", type=float, default=1.0, help="Initial curvature value")
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./pretrained_autoencoder")
    parser.add_argument("--log_with", type=str, default=None, 
                        choices=["tensorboard", "wandb", None], help="Logger backend")
    
    return parser.parse_args()

def save_tensor_as_images(tensor, prefix):
    """Save a batch of images as PNG with auto-normalization"""
    os.makedirs("debug_images", exist_ok=True)
    
    # Print range for debugging
    print(f"Tensor {prefix} range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    for i in range(min(len(tensor), 8)):  # save at most 8
        img = tensor[i].detach().cpu().permute(1, 2, 0).numpy()
        
        # Auto-normalize
        if img.min() < -0.01 or img.max() > 1.01:
            print(f"  Normalize image {i}, original range: [{img.min():.4f}, {img.max():.4f}]")
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img = np.clip(img, 0.0, 1.0)  # ensure in [0,1]
        img = (img * 255).astype('uint8')
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"{prefix} sample {i}")
        plt.axis('off')
        plt.savefig(f"debug_images/{prefix}_{i}.png")
        plt.close()

def monitor_model_weights(model, batch_count):
    """Monitor model weights for NaN/Inf and extreme values"""
    if 2890 <= batch_count <= 2910:
        print(f"\n===== Batch {batch_count} weight status check =====")
        problem_found = False
        problem_layers = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                problem_found = True
                problem_layers.append(name)
                print(f"Warning: parameter {name} contains NaN!")
            elif torch.isinf(param.data).any():
                problem_found = True
                problem_layers.append(name)
                print(f"Warning: parameter {name} contains Inf!")
            elif param.data.abs().max() > 1e3:
                print(f"Note: parameter {name} has very large values: {param.data.abs().max().item():.4e}")
        
        if problem_found:
            print(f"Problematic layers: {problem_layers}")
            return False
        else:
            print(f"Batch {batch_count}: all weights are OK")
            return True


def monitor_gradients(model, batch_count):
    """Monitor gradients and detect explosions"""
    if 2890 <= batch_count <= 2910:
        print(f"\n===== Batch {batch_count} gradient check =====")
        max_grad_norm = 0
        problem_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                if torch.isnan(param.grad).any():
                    problem_grads.append((name, "NaN"))
                    print(f"Warning: {name} gradient contains NaN!")
                elif torch.isinf(param.grad).any():
                    problem_grads.append((name, "Inf"))
                    print(f"Warning: {name} gradient contains Inf!")
                elif grad_norm > 100:
                    problem_grads.append((name, f"{grad_norm:.2e}"))
                    print(f"Warning: {name} gradient norm too large: {grad_norm:.2e}")
        
        print(f"Max gradient norm: {max_grad_norm:.4e}")
        return problem_grads
def visualize_reconstructions(model, validation_loader, epoch, output_dir):
    model.eval()
    with torch.no_grad():
        # Get a validation batch
        batch = next(iter(validation_loader))
        
        # Use parameter device instead of model.device
        device = next(model.parameters()).device
        images = batch[0][:8].to(device)  # take first 8 images on the correct device
        
        # Get reconstructions
        reconstructions, _ = model(images)
        
        # Print reconstruction tensor range for debugging
        print(f"Debug [Epoch {epoch}]: reconstruction tensor range [{reconstructions.min().item():.4f}, {reconstructions.max().item():.4f}]")

        # Concatenate original and reconstructed images and normalize for display
        # normalize=True maps [-1, 1] to [0, 1] for safe display
        comparison = torch.cat([images.cpu(), reconstructions.cpu()])
        grid = make_grid(comparison, nrow=8, normalize=True)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Epoch {epoch}: Original (top) vs Reconstructed (bottom)")
        
        # Save image
        recon_dir = os.path.join(output_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        plt.savefig(os.path.join(recon_dir, f"recon_epoch_{epoch}.png"), bbox_inches='tight')
        plt.close()
    
    model.train()

# Helper functions below

def plot_cumulative_losses(total_losses, recon_losses, perceptual_losses, color_losses, output_dir, window_size=2000):
    """
    Plot cumulative loss curves and update legend with latest averages.
    This function overwrites the previous image to keep a single up-to-date figure.
    """
    plt.figure(figsize=(15, 8))
    
    # Plot four loss curves
    plt.plot(total_losses, 'r-', alpha=0.7, linewidth=1.5, label='Total Loss')
    plt.plot(recon_losses, 'b-', alpha=0.6, linewidth=1.0, label='Recon Loss')
    plt.plot(perceptual_losses, 'g-', alpha=0.6, linewidth=1.0, label='Perceptual Loss')
    plt.plot(color_losses, 'm-', alpha=0.6, linewidth=1.0, label='Color Loss')
    
    # Compute last-N averages for legend
    if len(recon_losses) >= window_size:
        avg_total = np.mean(total_losses[-window_size:])
        avg_recon = np.mean(recon_losses[-window_size:])
        avg_perceptual = np.mean(perceptual_losses[-window_size:])
        avg_color = np.mean(color_losses[-window_size:])
        legend_text_total = f'Total Loss (last {window_size} avg: {avg_total:.4f})'
        legend_text_recon = f'Recon Loss (last {window_size} avg: {avg_recon:.4f})'
        legend_text_perceptual = f'Perceptual Loss (last {window_size} avg: {avg_perceptual:.4f})'
        legend_text_color = f'Color Loss (last {window_size} avg: {avg_color:.4f})'
    else:
        # If not enough data, do not show averages
        legend_text_total = 'Total Loss'
        legend_text_recon = 'Recon Loss'
        legend_text_perceptual = 'Perceptual Loss'
        legend_text_color = 'Color Loss'

    # Create custom legend
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='m', lw=2)]
    
    plt.title('Cumulative Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Use custom legend and text
    plt.legend(custom_lines, [legend_text_total, legend_text_recon, legend_text_perceptual, legend_text_color], loc='upper right')
    # plt.yscale('log')  # <-- removed log scale
    
    # Save image, overwrite previous file
    save_path = os.path.join(output_dir, "latest_loss.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150) # lower DPI to speed up saving
    plt.close()


def calc_loss_with_regularization(x, x_hat, z_hyp, manifold, x_train_var, perceptual_loss_fn, lambda_reg=5.0, lambda_p=0.1):
    """Multi-objective geometric regularization loss"""
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x)
    
    # Perceptual loss
    p_loss = perceptual_loss_fn(x_hat, x)

    x_norm = (x + 1.0)/2.0
    x_hat_norm = (x_hat + 1.0)/2.0
    
    x_hsv = kornia.color.rgb_to_hsv(x_norm)
    x_hat_hsv = kornia.color.rgb_to_hsv(x_hat_norm)
    
    hue_loss = F.l1_loss(x_hsv[:,0,:,:], x_hat_hsv[:,0,:,:])
    sat_loss = F.l1_loss(x_hsv[:,1,:,:], x_hat_hsv[:,1,:,:])
    color_loss = hue_loss + sat_loss
    
    # Extract radius info
    radius = torch.acosh(torch.clamp(z_hyp[..., 0], min=1.0+1e-7))
    
    # 1) Encourage radius dispersion: penalize small std (maximize std)
    # Minimizing -std is equivalent to maximizing std
    radius_std_reg = -torch.sqrt(torch.var(radius) + 1e-8)
    
    # 2) Encourage mean radius near target: penalize deviation
    mean_target = 5.0
    mean_radius_reg = (radius.mean() - mean_target)**2

    # 3) Encourage uniformity (optional, more reasonable):
    # Penalize KS statistic; larger value => further from uniform
    sorted_radius, _ = torch.sort(radius.reshape(-1))
    target_cdf = (sorted_radius - sorted_radius.min()) / (sorted_radius.max() - sorted_radius.min() + 1e-8)
    uniform_cdf = torch.linspace(0, 1, len(sorted_radius), device=z_hyp.device)
    uniformity_reg = torch.max(torch.abs(target_cdf - uniform_cdf))

    # Combine regularizers (positive penalty)
    # Weights are tunable; example given
    reg_loss = lambda_reg * (0.5 * radius_std_reg + 0.3 * mean_radius_reg + 0.2 * uniformity_reg)
    
    lambda_color = 0.1
    total_loss = recon_loss + lambda_p * p_loss + reg_loss + lambda_color * color_loss
    
    # Return values (including color_loss_val)
    return total_loss, recon_loss, p_loss, recon_loss.item(), reg_loss.item(), p_loss.item(), color_loss.item()
    

def analyze_embeddings(z_hyp, epoch, batch_idx, output_dir):
    """Analyze hyperbolic embedding distribution"""
    with torch.no_grad():
        # Extract radius and ensure 1D
        radius = torch.acosh(torch.clamp(z_hyp[:, 0], min=1.0+1e-5))
        
        # Flatten tensor to 1D regardless of input dimension
        radius_flat = radius.reshape(-1).cpu().numpy()
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(radius_flat, bins=30)
        plt.title(f"Epoch {epoch}, Batch {batch_idx} - hyp radius distribution")
        plt.xlabel("hyp radius")
        plt.ylabel("frequency")
        
        # Add statistics (numpy functions instead of tensor methods)
        plt.axvline(np.mean(radius_flat), color='r', linestyle='--')
        stats_text = (f"mean: {np.mean(radius_flat):.4f}\n"
                     f"std: {np.std(radius_flat):.4f}\n"
                     f"min: {np.min(radius_flat):.4f}\n"
                     f"max: {np.max(radius_flat):.4f}")
        plt.annotate(stats_text, xy=(0.7, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
        plt.savefig(f"{output_dir}/embeddings/radius_dist_e{epoch}_b{batch_idx}.png")
        plt.close()

def train_autoencoder():
    args = parse_args()
    train_losses = []  # average loss per epoch
    val_losses = []    # validation loss per epoch
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute gradient accumulation steps
    gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    print(f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {args.effective_batch_size})")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=args.log_with
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Use accelerator.print to avoid duplicate prints in distributed
    accelerator.print(f"Using device: {accelerator.device}")
    
    # Load dataset
    accelerator.print("Loading dataset...")
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    accelerator.print(f"Dataset sizes: {len(training_data)} train, {len(validation_data)} val")
    
    # Create model
    accelerator.print("Creating autoencoder model...")
    model = StandardAutoencoder(
        h_dim=args.h_dim,
        n_res_layers=args.n_res_layers,
        c=args.initial_c,
        adaptive_c=args.adaptive_c

    )
    nan_grad_detected = False
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            if 2880 <= batch_count <= 2910:
                print(f"Zeroing NaN/Inf gradients for parameter {name}")
            param.grad.zero_()
            nan_grad_detected = True

    # Reset optimizer momentum if NaN gradients detected
    if nan_grad_detected and 2880 <= batch_count <= 2910:
        print("Detected NaN gradients, resetting optimizer momentum...")
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].zero_()
    # Optimizer
    optimizer = RiemannianSGD(
        model.parameters(), 
        lr=args.learning_rate, 
        stabilize=1000,
        weight_decay=args.weight_decay,
        momentum=0.95
    )
    
    # LR scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True,
        min_lr=1e-6
    )
    
    # Prepare with accelerator
    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )
    
    # Track best
    best_loss = float('inf')
    
    # Training loop
    batches_per_epoch = len(training_data) // args.batch_size
    visualization_interval = max(1, batches_per_epoch // 100)
    
    accelerator.print(f"~{batches_per_epoch} batches per epoch")
    accelerator.print(f"Visualize every {visualization_interval} batches (~every 1/100 epoch)")
    
    # Start training
    accelerator.print(f"Starting training for {args.epochs} epochs...")
    
    # Setup logging
    if accelerator.is_main_process and args.log_with is not None:
        accelerator.init_trackers("autoencoder_training")
    
    # Define variables for special checkpoint
    if accelerator.is_main_process:
        best_loss_in_first_fifth = float('inf')
        # histories
        history_total_losses = []
        history_recon_losses = []
        history_perceptual_losses = []
        history_color_losses = []
        
    # Instantiate VGG perceptual loss
    perceptual_loss = VGGPerceptualLoss().to(accelerator.device)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Final target for dynamic lambda_reg
        final_lambda_reg = 0.005

        # Compute target batch count for special checkpoint
        if epoch == 0 and accelerator.is_main_process:
            target_batch_for_checkpoint = batches_per_epoch // 5
            accelerator.print(f"--- [Epoch 0] Special monitoring: will look for best within first {target_batch_for_checkpoint} batches. ---")
        
        # Store per-batch losses for current epoch
        current_epoch_batch_losses = []
        # Temp list for plotting every 2000 batches
        interval_batch_losses = []
        last_plot_batch = 0
        
        # Let errors surface (no try/except)
        for batch_idx, batch_data in enumerate(training_loader):
            current_batch = batch_count
            

            with accelerator.accumulate(model):
                if batch_count % 500 == 0:
                    torch.cuda.empty_cache()
                x, _ = batch_data
                
                # Forward - reconstruction only
                x_hat, z_hyp = model(x, batch_count=batch_count)
                
                # Warmup logic
                if batch_count < args.reg_warmup_batches:
                    current_lambda_reg = 0.0
                else:
                    current_lambda_reg = final_lambda_reg
                
                # Compute loss (target is x)
                loss, recon_loss, p_loss, recon_loss_val, reg_loss_val, p_loss_val, color_loss_val = calc_loss_with_regularization(
                    x, x_hat, z_hyp, model.manifold, x_train_var, perceptual_loss,
                    lambda_reg=current_lambda_reg,
                    lambda_p=args.lambda_p
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: loss is {loss.item()}, skipping this batch")
                    continue
                # Backward
                accelerator.backward(loss)


                # Gradient monitoring in window
                if 5770  <= batch_count <= 5800:
                    problem_grads = monitor_gradients(model, batch_count)
                    if problem_grads:
                        print(f"\n======= Batch {batch_count} abnormal gradients detected! =======")
                        print(f"Problem gradients: {len(problem_grads)} items, e.g.: {problem_grads[:3]}")
                
                # Weight monitoring in window
                if 5770 <= current_batch <= 5800:
                    try:
                        weights_ok = monitor_model_weights(model, current_batch)
                        if not weights_ok:
                            print(f"\n======= Batch {current_batch} abnormal weights detected! =======")
                            # Diagnose each module
                            for name, module in model.named_modules():
                                if hasattr(module, 'weight') and module.weight is not None:
                                    try:
                                        if isinstance(module.weight, torch.Tensor):
                                            w = module.weight.data
                                            print(f"Layer {name}: weight range [{w.min().item():.4e}, {w.max().item():.4e}], "
                                                  f"NaN count: {torch.isnan(w).sum().item()}, "
                                                  f"Inf count: {torch.isinf(w).sum().item()}")
                                    except Exception as e:
                                        print(f"Error checking layer {name} weights: {str(e)}")
                                
                                if hasattr(module, 'bias') and module.bias is not None:
                                    try:
                                        if isinstance(module.bias, torch.Tensor):
                                            b = module.bias.data
                                            print(f"Layer {name}: bias range [{b.min().item():.4e}, {b.max().item():.4e}], "
                                                  f"NaN count: {torch.isnan(b).sum().item()}, "
                                                  f"Inf count: {torch.isinf(b).sum().item()}")
                                        else:
                                            print(f"Layer {name}: bias type is {type(module.bias)}, skip")
                                    except Exception as e:
                                        print(f"Error checking layer {name} bias: {str(e)}")
                    except Exception as e:
                        print(f"Error during weight monitoring: {str(e)}")
                # Gradient clipping
            if args.grad_clip_value > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_value)

                nan_grad_detected = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        if 5770 <= batch_count <= 5800:
                            print(f"Zeroing NaN/Inf gradients for parameter {name}")
                        param.grad.zero_()
                        nan_grad_detected = True

                # Reset optimizer momentum if many NaN gradients
                if nan_grad_detected and 5770 <= batch_count <= 5800:
                    print("Detected NaN gradients, resetting optimizer momentum...")
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p in optimizer.state:
                                param_state = optimizer.state[p]
                                if 'momentum_buffer' in param_state:
                                    param_state['momentum_buffer'].zero_()
                    
                optimizer.step()
                optimizer.zero_grad()
                
                
                # Periodic logging (main process only)
                if accelerator.is_main_process and batch_count % 100 == 0:
                    accelerator.print(
                        f"Epoch {epoch}, Batch {batch_count}/{batches_per_epoch}, "
                        f"Total Loss: {loss.item():.6f}, Recon Loss: {recon_loss_val:.6f}, "
                        f"Perceptual Loss: {p_loss_val:.6f}, Reg Loss: {reg_loss_val:.6f}, "
                        f"Lambda_Reg: {current_lambda_reg:.6f}, Color Loss: {color_loss_val:.6f}"
                )
            # Update losses only on main process
            if accelerator.is_main_process:
                current_loss = loss.item()
                epoch_loss += current_loss

                # Append to histories
                history_total_losses.append(current_loss)
                history_recon_losses.append(recon_loss_val)
                history_perceptual_losses.append(p_loss_val)
                history_color_losses.append(color_loss_val)
                
                # Save best within first 1/5 of epoch 0
                if epoch == 0 and batch_count < target_batch_for_checkpoint:
                    if current_loss < best_loss_in_first_fifth:
                        best_loss_in_first_fifth = current_loss
                        
                        accelerator.print(
                            f"\n[Epoch 0] New best loss: {best_loss_in_first_fifth:.6f} at batch {batch_count}. Saving model..."
                        )
                        
                        # Save via accelerator
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_path = os.path.join(args.output_dir, "best_model_first_fifth_epoch.pth")
                        
                        accelerator.save({
                            'encoder': unwrapped_model.encoder.state_dict(),
                            'decoder': unwrapped_model.decoder.state_dict(),
                            'epoch': epoch,
                            'batch': batch_count,
                            'loss': best_loss_in_first_fifth,
                            'adaptive_c': args.adaptive_c,
                            'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                            'args': vars(args)
                        }, save_path)
                        
                        accelerator.print(f"  -> Saved to: {save_path}\n")

                batch_count += 1
                # Record current batch loss
                current_epoch_batch_losses.append(current_loss)
                interval_batch_losses.append(current_loss)   
            # Plot cumulative loss every 2000 batches (main process only)
            if accelerator.is_main_process and batch_count > 0 and batch_count % 2000 == 0:
                plot_cumulative_losses(
                    history_total_losses,
                    history_recon_losses,
                    history_perceptual_losses,
                    history_color_losses,
                    args.output_dir
                )
                accelerator.print(f"Epoch {epoch}, updated cumulative loss figure: latest_loss.png")
            
            # Visualization (main process only)
            if accelerator.is_main_process and (batch_count % visualization_interval == 0 or batch_count == 100):
                visualize_reconstructions(model, validation_loader, f"{epoch}_batch{batch_count}", args.output_dir + "/intermediate")
                analyze_embeddings(z_hyp, epoch, batch_count, args.output_dir)
                accelerator.print(f"Epoch {epoch}, Batch {batch_count}: finished intermediate visualization and embedding analysis")
                



        if accelerator.is_main_process and interval_batch_losses and batch_count - last_plot_batch >= 100:
            # Not needed now since the loss figure is cumulative
            pass
        # Save this epoch's batch losses for final detailed loss plot
        batch_losses=[]
        if accelerator.is_main_process:
            batch_losses.append(current_epoch_batch_losses)
        
        
        
        # Compute average loss
        if accelerator.is_main_process:
            avg_loss = epoch_loss / batch_count
            accelerator.print(f"Epoch {epoch} done, average loss: {avg_loss:.6f}")
            train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
            with torch.no_grad():
            for val_x, _ in validation_loader:
                val_x_hat, _ = model(val_x)
                val_batch_loss = F.mse_loss(val_x_hat, val_x) / x_train_var
                val_loss += val_batch_loss.item()
                val_count += 1
        
        # Synchronize validation results
        val_loss = accelerator.gather(torch.tensor(val_loss, device=accelerator.device)).sum().item()
        val_count = accelerator.gather(torch.tensor(val_count, device=accelerator.device)).sum().item()
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        accelerator.print(f"Validation loss: {avg_val_loss:.6f}")
        if accelerator.is_main_process:
            val_losses.append(avg_val_loss)

        # Scheduler step
        scheduler.step(avg_val_loss)
        # Log scalars
        if accelerator.is_main_process and args.log_with is not None:
            logs = {"train_loss": avg_loss, "val_loss": avg_val_loss, "learning_rate": optimizer.param_groups[0]['lr']}
            accelerator.log(logs, step=epoch)

        # Periodic logging
        if accelerator.is_main_process and batch_count % 1000 == 0:
            # Get current curvature
            c_value = model.manifold.c.item()
            print(f"Batch {batch_count} current curvature: {c_value}")
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print summary status
            accelerator.print("\n" + "="*50)
            accelerator.print(f"Status summary [Epoch {epoch}, Batch {batch_count}]")
            accelerator.print(f"Curvature: {c_value}")
            accelerator.print(f"Learning rate: {current_lr}")
            accelerator.print(f"Last 10 batches avg loss: {np.mean(current_epoch_batch_losses[-10:]):.6f}")
            accelerator.print(f"Last 1000 batches loss std: {np.std(current_epoch_batch_losses[-1000:]):.6f}")
            accelerator.print("="*50 + "\n")
                    if accelerator.is_main_process:
            # Save checkpoint each epoch
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_path = f"{args.output_dir}/epoch_{epoch}_autoencoder.pth"
            torch.save({
                'encoder': unwrapped_model.encoder.state_dict(),
                'decoder': unwrapped_model.decoder.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
                'adaptive_c': args.adaptive_c,
                'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                'args': vars(args)
            }, checkpoint_path)
            accelerator.print(f"Saved epoch {epoch} checkpoint to: {checkpoint_path}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                checkpoint_path = f"{args.output_dir}/best_autoencoder.pth"
                torch.save({
                    'encoder': unwrapped_model.encoder.state_dict(),
                    'decoder': unwrapped_model.decoder.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'adaptive_c': args.adaptive_c,
                    'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
                    'args': vars(args)
                }, checkpoint_path)
                accelerator.print(f"Saved best model to: {checkpoint_path}")
            
            # End-of-epoch visualization
            visualize_reconstructions(unwrapped_model, validation_loader, epoch, args.output_dir)
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = f"{args.output_dir}/final_autoencoder.pth"
        torch.save({
            'encoder': unwrapped_model.encoder.state_dict(),
            'decoder': unwrapped_model.decoder.state_dict(),
            'epoch': args.epochs - 1,
            'loss': avg_val_loss,
            'adaptive_c': args.adaptive_c,
            'curvature': unwrapped_model.curvature.state_dict() if args.adaptive_c else args.initial_c,
            'args': vars(args)
        }, final_path)
        accelerator.print(f"Saved final model to: {final_path}")
        
        accelerator.print(f"Pretraining complete! Best validation loss: {best_loss:.6f}")
    
    # End logging
    if accelerator.is_main_process and args.log_with is not None:
        accelerator.end_training()

if __name__ == "__main__":
    train_autoencoder()
