# Polar: Hyperbolic VQ-VAE and Tokenizers (Lorentz Manifold)

This directory contains a compact research codebase for training autoencoders and VQ-VAE models in hyperbolic geometry (Lorentz model). It includes:

- A Euclidean encoder/decoder with hyperbolic mappings
- Two flavors of hyperbolic tokenizers (simple distance-based and polar/cluster-aware)
- A hyperbolic VQ-VAE training loop with curvature scheduling, visualization, and codebook analytics
- An AE pretraining pipeline and tools to train/evaluate tokenizers using the AE

If you are new here, start with AE pretraining, then train the hyperbolic VQ-VAE.

## Repository Structure

```text
polar/
  main_hyp.py                  # Train Hyperbolic VQ-VAE (Accelerate-ready)
  train_autoencoder.py         # Pretrain AE (Accelerate-ready)
  evaluate_autoencoder.py      # Evaluate AE recon quality (PSNR/SSIM/MSE/FID)
  reconstruct_images.py        # Utility to reconstruct images using a trained model
  reconstraction.py            # (legacy helper)
  utils.py                     # Dataset loading (CIFAR-10 / ImageNet), robust ImageFolder
  utils_hyp.py                 # Losses, VGG-perceptual, plots, monitoring utilities
  requirements.txt             # Minimal baseline requirements (see below for modern stack)

  models/
    encoder_hyp.py             # Encoder (Euclid->Lorentz), geometric residual/downsampling
    decoder_hyp.py             # Decoder (Lorentz->Euclid)
    AE.py                      # Standard autoencoder wrapper
    vqvae_hyp.py               # Hyperbolic VQ-VAE model (enc + tokenizer + dec)

    # Hyperbolic tokenizers (two families):
    quantizer_simple.py        # StandardHyperbolicQuantizer (distance-based; robust baseline)
    quantizer_hyp.py           # VectorQuantizer (polar: radial + angular bins)
    cluster_aware_quantizer.py # ClusterAwareVectorQuantizer (polar + hierarchical clustering)

    # Other components and variants
    residual.py, metrics.py, adaptive_curvature.py, ...

  datasets/                    # (placeholder) if you add custom datasets
  new_results/, results_balanced/  # Output/example result folders
```

## Key Ideas

- Lorentz model hyperbolic geometry is used throughout. Encoder lifts Euclidean features to Lorentz points; decoder maps back.
- Tokenization happens directly in hyperbolic space:
  - StandardHyperbolicQuantizer: a single codebook of Lorentz points; nearest neighbor by true hyperbolic distance (robust, simple, good baseline).
  - VectorQuantizer (polar) and ClusterAwareVectorQuantizer (polar + clustering): decompose into radius r and unit direction w, then reconstruct a Lorentz point. Offers finer control of radial/angular usage and optional EMA/cluster-aware updates.
- Training utilities include perceptual/color losses, curvature adaptation, diversity/codebook regularization, dead-code reset, and rich plotting.

## Installation

Recommended (modern stack; Python 3.10+):

```bash
conda create -n polar python=3.10 -y
conda activate polar

# Core DL stack (choose versions matching your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # example

# Project deps
pip install -r requirements.txt  # baseline
pip install accelerate geoopt kornia scikit-learn matplotlib pillow tqdm torchmetrics clean-fid
```

Notes:
- The provided `requirements.txt` pins some legacy packages. Prefer using modern versions as shown above where possible.
- Geoopt is used for Riemannian optimization and utility stabilization.
- Accelerate is used for mixed-precision and multi-GPU orchestration.

## Datasets

- CIFAR-10: automatically downloaded to `./data` when selected.
- ImageNet: by default, the code expects
  - `/ext/imagenet/train`
  - `/ext/imagenet/val`

You can customize loaders in `utils.py` if your layout differs. For quick functional checks, prefer `--dataset CIFAR10` first.

## Quick Start

### 1) Pretrain the Autoencoder (AE)

```bash
accelerate launch /ext/work/polar/train_autoencoder.py \
  --dataset IMAGENET \
  --batch_size 32 \
  --effective_batch_size 256 \
  --epochs 30 \
  --learning_rate 3e-3 \
  --adaptive_c \
  --initial_c 1.0 \
  --output_dir /ext/work/pretrained_autoencoder
```

This produces an AE checkpoint (encoder/decoder weights and curvature when enabled) used to warm-start the VQ-VAE training.

### 2) Train the Hyperbolic VQ-VAE

Standard single-GPU (use absolute paths):

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch /ext/work/polar/main_hyp.py \
  --dataset IMAGENET \
  --batch_size 8 \
  --effective_batch_size 256 \
  --n_hiddens 128 \
  --n_residual_layers 2 \
  --n_embeddings 8192 \
  --beta 1.0 \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --grad_clip_value 3.0 \
  --initial_c 1.0 \
  --use_lr_scheduler \
  --lr_scheduler plateau \
  --output_dir /ext/work/vqvae_results_with_ae \
  --save \
  --save_interval 50000 \
  --val_interval 2000 \
  --log_interval 100 \
  --monitor_interval 400 \
  --visualization_interval 2000 \
  --pretrained_path /ext/work/pretrained_autoencoder/epoch_1_autoencoder.pth \
  --fp16
```

Tips:
- Prefer `--learning_rate 3e-4` to start; `1e-2` is usually too high and can diverge.
- Use `--fp16` for speed/memory unless it destabilizes your setup.
- You can expose quantizer warmup to defer using z_q early on: add `--quantizer_warmup_steps 5000`.

Outputs (by default under `--output_dir`):
- Periodic reconstructions, loss curves, codebook usage, latent distribution plots
- Best checkpoints and periodic training checkpoints

### 3) (Optional) Reconstruct Images

```bash
python -u /ext/work/polar/reconstruct_images.py \
  --checkpoint /path/to/checkpoint.pth \
  --input_dir /path/to/images \
  --output_dir /path/to/recons
```

### 4) Evaluate AE Reconstruction Quality

```bash
python -u /ext/work/polar/evaluate_autoencoder.py \
  --checkpoint /ext/work/pretrained_autoencoder/best.pth \
  --val_dir /ext/imagenet/val \
  --batch_size 8 \
  --num_samples 10000 \
  --output_dir /ext/work/eval_ae
```

Metrics include PSNR, SSIM, MSE, and FID (via clean-fid). The script saves real/reconstruction images to compute FID.

## Choosing the Hyperbolic Tokenizer

The VQ-VAE (`models/vqvae_hyp.py`) currently instantiates the robust baseline:

- `StandardHyperbolicQuantizer` (in `models/quantizer_simple.py`):
  - Single codebook of Lorentz points
  - True hyperbolic nearest-neighbor in chunks to reduce memory
  - Dead-code reset and diversity metrics

Two alternative polar variants are available:

- `VectorQuantizer` (in `models/quantizer_hyp.py`): radial bins + angular codebook, temperature-adjusted distance
- `ClusterAwareVectorQuantizer` (in `models/cluster_aware_quantizer.py`): polar + hierarchical/EMA-style cluster guidance

To switch tokenizers, edit `models/vqvae_hyp.py` where the tokenizer is instantiated and select the desired class and arguments (the code already contains commented examples showing how to plug them in). Re-run training.

## Important Arguments (VQ-VAE)

- Model: `--n_hiddens`, `--n_residual_layers`, `--n_embeddings`, `--beta`, `--initial_c`, `--adaptive_c`
- Training: `--batch_size`, `--effective_batch_size`, `--learning_rate`, `--weight_decay`, `--grad_clip_value`, `--fp16`
- Scheduler: `--use_lr_scheduler`, `--lr_scheduler {plateau,cosine}`, `--min_lr`
- AE warm-start: `--pretrained_path`
- Quantizer warmup: `--quantizer_warmup_steps`
- Logging/IO: `--output_dir`, `--log_interval`, `--val_interval`, `--save`, `--save_interval`, `--visualization_interval`, `--monitor_interval`
- Loss weights: `--lambda_recon`, `--lambda_p`, `--lambda_color`, `--lambda_diversity`, `--reg_warmup_steps`, `--final_lambda_reg`, `--lambda_cb_reg`, `--target_cb_radius`

## Visualization and Monitoring

`utils_hyp.py` provides rich diagnostics invoked by `main_hyp.py`:
- Reconstruction grids over time
- Cumulative loss plots (with smoothing)
- Codebook usage histograms (log scale)
- Latent distribution (PCA 3D + polar plots) and angular distribution near typical radii
- Gradient and weight monitors (NaN/Inf, extremes)

Artifacts are written under `--output_dir` in subfolders like `reconstructions/`, `latent_space_dist/`, `angular_dist/`.

## Training Tips

- Start small (CIFAR-10) to validate end-to-end plumbing: `--dataset CIFAR10 --n_updates 10000`.
- Use gradient accumulation via `--effective_batch_size` when memory is limited.
- If OOM occurs in tokenization, reduce the chunk size in `quantizer_simple.py` (default 1024).
- Curvature adaptation (`--adaptive_c`) can improve stability; otherwise keep `--initial_c 1.0`.
- Dead-code reset is supported in the tokenizer; set `--reset_codes_interval` in `main_hyp.py`.

## Troubleshooting

- ImageNet path errors: adjust the hard-coded path in `utils.py` or mirror your dataset to `/ext/imagenet/{train,val}`.
- Divergence early in training: lower `--learning_rate` (e.g., `1e-4`), enable `--fp16`, ensure AE warm-start is valid.
- NaNs/Inf: monitors will report layers/gradients. Consider smaller LR, stronger grad clipping, or enabling curvature adaptation.
- Very low code usage: increase diversity weight (`--lambda_diversity`) or consider polar tokenizers.

## Acknowledgements

This code leverages geoopt for Riemannian optimization, torchvision for VGG perceptual features, and standard scientific Python tools for analysis/visualization.
