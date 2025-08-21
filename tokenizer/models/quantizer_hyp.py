import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import geoopt
sys.path.append('/ext/work')
from hyplib.manifolds.lmath import poincare_to_lorentz, lorentz_to_poincare, dist
from hyplib.manifolds.lorentzian import Lorentz
import numpy as np

def from_polar(r, w):
    """
    r: (...), hyperbolic radius
    w: (..., e_dim), unit vector in tangent/Euc space
    returns x in Poincaré ball: x = tanh(r/2) * w
    """
    # Ensure r has dimensionality matched with w to avoid unintended broadcasting
    if r.dim() == 1 and w.dim() > 1:
        r = r.view(-1, 1)
    return torch.tanh(r / 2.0) * w

def check_tensor(tensor, name, step, detach=True):
    """Check if a tensor contains NaN or Inf values"""
    if detach:
        tensor = tensor.detach()
            
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
            
    if has_nan or has_inf:
        print(f"[Step {step}] {name} contains NaN: {has_nan}, Inf: {has_inf}")
                
        # Get statistics for valid values
        if not (has_nan and tensor.isnan().all()):
            tensor_valid = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
            if len(tensor_valid) > 0:
                print(f"  Valid values - min: {tensor_valid.min().item():.4f}, max: {tensor_valid.max().item():.4f}")
        return False
    return True

def check_on_manifold(x_hyp, name, manifold, step, tolerance=1e-5):
    """Check whether a point lies on the hyperbolic manifold"""
    t = x_hyp[..., 0:1]  # time component
    x = x_hyp[..., 1:]   # spatial components
            
    # Riemannian metric constraint: t^2 - ||x||^2 = 1
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
    t_sq = t**2
    constraint = t_sq - x_norm_sq
            
    deviation = torch.abs(constraint - 1.0)
    max_deviation = deviation.max().item()
            
    if max_deviation > tolerance:
        print(f"[Step {step}] {name} violates geometric constraint, max deviation: {max_deviation:.6f}")
        max_idx = torch.argmax(deviation).item()
        flat_t = t.flatten()
        flat_norm = torch.sqrt(x_norm_sq).flatten()
        print(f"  Problematic point t value: {flat_t[max_idx].item():.4f}, spatial norm: {flat_norm[max_idx].item():.4f}")
        return False
    return True


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, radial_bins=16, max_radius=1.1, 
                use_ema=False, ema_decay=0.99, c=1.0, initial_temp=5.0):
        super().__init__()
        assert n_e % radial_bins == 0, "n_e must be divisible by radial_bins"
        self.n_e          = n_e
        self.e_dim        = e_dim           # Poincaré ball spatial dimension
        self.beta         = beta
        self.radial_bins  = radial_bins
        self.angular_bins = n_e // radial_bins
        self.max_radius   = max_radius
        self.manifold     = Lorentz(c=c)
        self.initial_temp = initial_temp
        self.current_temp = initial_temp  
        self.register_buffer('temp', torch.tensor(initial_temp)) 
        
        # Add EMA support
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Radial centers (hyperbolic radius r)
        if radial_bins > 0:
            # Set a new radial center distribution to match the encoder output range
            # Set a radial distribution that matches the encoder's extended range
            r_min = 0.15
            r_max = 1.2 # Slightly larger than the encoder's maximum radius (0.6074)
            # Method 1: use linear spacing (recommended here given the range)
            r_values = torch.linspace(r_min, r_max, radial_bins).tolist()
            
            # Method 2: piecewise linear distribution (alternative)
            # r_values = []
            # r_values.extend(torch.linspace(r_min, 1.0, radial_bins//4).tolist())  # 25% bins for [0.33-1.0]
            # r_values.extend(torch.linspace(1.0, 3.0, radial_bins//4).tolist())    # 25% bins for [1.0-3.0]
            # r_values.extend(torch.linspace(3.0, 7.0, radial_bins//4).tolist())    # 25% bins for [3.0-7.0]
            # r_values.extend(torch.linspace(7.0, r_max, radial_bins-len(r_values)).tolist()) # remaining bins for [7.0-10.0]
            
            self.r_centres = nn.Parameter(torch.tensor(r_values))

        # Angular centers: unit vectors in R^(e_dim)
        self.angular_codebook = nn.Embedding(self.angular_bins, self.e_dim- 1)
        # Ensure angular vectors are initialized across different radii
        with torch.no_grad():
            # Re-initialize radial centers to ensure a more uniform distribution
            r_span = r_max - r_min
            for i, r in enumerate(self.r_centres):
                # Slightly perturb radius to avoid perfectly uniform distribution
                noise = torch.randn(1).item() * 0.05 * r_span
                self.r_centres.data[i] = r_min + (i / (self.radial_bins-1)) * r_span + noise
                self.r_centres.data[i] = torch.clamp(self.r_centres.data[i], min=r_min, max=r_max)
            
            # --- Fix: correctly initialize the angular codebook ---
            # The angular codebook is independent of the radial component and only needs to be initialized once.
            # The previous complex loop was incorrect as it attempted to access non-existent indices in the codebook.
            v = torch.randn(self.angular_bins, self.e_dim - 1)
            self.angular_codebook.weight.data.copy_(F.normalize(v, dim=-1))
            
        # Add buffers for EMA updates
        if self.use_ema:
            # Radial EMA buffers
            self.register_buffer('r_centres_ema', self.r_centres.clone().detach())
            self.register_buffer('r_cluster_size', torch.zeros(radial_bins))
            
            # Angular EMA buffers
            self.register_buffer('angular_ema', self.angular_codebook.weight.clone().detach())
            self.register_buffer('angular_cluster_size', torch.zeros(self.angular_bins))
            
            # EMA state tracking
            self.register_buffer('ema_initialized', torch.tensor(0))

        # Add to VectorQuantizer class
    def temp_adjusted_dist(self, x, y, temp=None):
        """Temperature-adjusted hyperbolic distance computation"""
        temp = temp if temp is not None else self.temp
        
        # Base hyperbolic distance
        hyp_dist = self.manifold.dist(x, y)
        
        # Compute radius difference penalty
        r_x = torch.acosh(torch.clamp(x[:, 0], min=1.0+1e-5))
        r_y = torch.acosh(torch.clamp(y[:, 0], min=1.0+1e-5))
        radius_diff = torch.abs(r_x - r_y)
        
        # Add radius difference penalty term
        radius_weight = 1  # Increase the importance of radius difference
        return hyp_dist + radius_weight * radius_diff

    def forward(self, u_hyp, debug_step=None, features_for_clustering=None):
        """
        Vector quantization directly on hyperbolic inputs.
        Input: u_hyp [B, C+1, H, W] - points on the Lorentz model (including time component)
        """
        if debug_step is not None and debug_step < 100:
            check_tensor(u_hyp, "quantizer input", debug_step)
        
        u_hyp_shape = u_hyp.shape
        u_hyp_flat = u_hyp.reshape(-1, u_hyp_shape[-1])

        # --- Decomposition ---
        u_time = u_hyp_flat[:, 0:1]
        u_space = u_hyp_flat[:, 1:]
        r = torch.acosh(u_time.clamp(min=1.0 + 1e-2))
        w = F.normalize(u_space, dim=1)
        
        # --- Improved quantization process ---
        r_centres = torch.clamp(self.r_centres, min=1e-2, max=self.max_radius)
        
        # 1. Pre-filter the most likely candidates
        top_k_r = min(3, self.radial_bins)   # select nearest radii
        top_k_w = min(5, self.angular_bins)  # select most similar directions
        
        # Find top-k nearest radial values
        dist_r = torch.abs(r - r_centres)  # use absolute difference instead of squared
        _, top_r_indices = torch.topk(-dist_r, k=top_k_r, dim=-1)   # negative to rank smaller distances first
        
        # Find top-k most similar angular directions
        sim = torch.matmul(w, self.angular_codebook.weight.t())
        _, top_w_indices = torch.topk(sim, k=top_k_w, dim=-1)
        
        # 2. Among filtered candidates, find the truly nearest
        batch_size = u_hyp_flat.size(0)
        best_dists = torch.full((batch_size,), float('inf'), device=u_hyp_flat.device)
        best_r_idx = torch.zeros((batch_size,), dtype=torch.long, device=u_hyp_flat.device)
        best_w_idx = torch.zeros((batch_size,), dtype=torch.long, device=u_hyp_flat.device)
        
        # Efficient computation: batch processing to avoid loops where possible
        for i in range(top_k_r):
            r_idx_batch = top_r_indices[:, i]
            r_vals = r_centres[r_idx_batch].unsqueeze(-1)  # [batch, 1]
            
            for j in range(top_k_w):
                w_idx_batch = top_w_indices[:, j]
                w_vals = self.angular_codebook(w_idx_batch)  # [batch, e_dim-1]
                
                # Compute hyperbolic distance for this (r, w) combination
                candidate_poinc = from_polar(r_vals, w_vals)
                candidate_hyp = poincare_to_lorentz(candidate_poinc, k=self.manifold.k)
                candidate_hyp = self.manifold.projx(candidate_hyp)
                
                # Distance to the input
                curr_dists = self.manifold.dist(u_hyp_flat, candidate_hyp)
                
                # Update best match
                update_mask = curr_dists < best_dists
                best_dists[update_mask] = curr_dists[update_mask]
                best_r_idx[update_mask] = r_idx_batch[update_mask]
                best_w_idx[update_mask] = w_idx_batch[update_mask]
        
        # Get best-matching codes
        r_hard = r_centres[best_r_idx]
        w_hard = self.angular_codebook(best_w_idx)
        
        # --- Reconstruction ---
        x_q_poinc = from_polar(r_hard.unsqueeze(-1), w_hard)
        x_q_hyp_flat = poincare_to_lorentz(x_q_poinc, k=self.manifold.k)
        x_q_hyp_flat = self.manifold.projx(x_q_hyp_flat)

        # --- Loss calculation ---
        # VQ loss: encourage encoder outputs to be close to codebook vectors
        # Use detach() to stop gradients flowing to the codebook
        if not self.use_ema:
            # Use temperature-adjusted distance
            codebook_loss = self.temp_adjusted_dist(u_hyp_flat.detach(), x_q_hyp_flat, self.temp).mean()
            commitment_loss = self.temp_adjusted_dist(u_hyp_flat, x_q_hyp_flat.detach(), self.temp).mean()
            loss = codebook_loss + self.beta * commitment_loss
        else:
            # With EMA, only commitment loss is needed
            loss = self.beta * self.temp_adjusted_dist(u_hyp_flat, x_q_hyp_flat.detach(), self.temp).mean()

        # --- Straight-through gradient ---
        with torch.no_grad():
            if self.temp > 1.5:  # high-temperature regime
                # Simple linear interpolation
                direction = x_q_hyp_flat.detach() - u_hyp_flat
                z_q_hyp = u_hyp_flat + direction
            else:
                # Standard hyperbolic geodesic
                direction = self.manifold.logmap(x_q_hyp_flat.detach(), u_hyp_flat)
                z_q_hyp = self.manifold.expmap(u_hyp_flat, direction)

        # Ensure geometric constraint
        z_q_hyp = self.manifold.projx(z_q_hyp.view(u_hyp_shape))


        # --- Compute perplexity and usage ---
        r_idx_flat = best_r_idx.flatten()
        w_idx_flat = best_w_idx.flatten()
        combined_idx = r_idx_flat * self.angular_bins + w_idx_flat
        e_mean = F.one_hot(combined_idx, self.n_e).float().mean(0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # Fix: codebook_usage should return the e_mean vector itself for visualization
        # rather than returning a scalar
        codebook_usage = e_mean

        # --- Add: codebook diversity loss ---
        # Goal is to maximize the entropy of e_mean, equivalent to minimizing negative entropy
        diversity_loss = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        
        # Clean up return interface, remove extra None
        return loss, z_q_hyp, perplexity, diversity_loss, codebook_usage
