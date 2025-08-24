## Overview

This repository implements and prototypes hyperbolic-space (Hyperbolic) modeling across image, text, audio, video, and multimodal tasks. Core components include:
- **HypFormer**: a hyperbolic Transformer backbone (`utils/hypformer*.py`).
- **HyperLib**: manifolds, geometry utilities, and layers (`HyperLib/`).
- **LFQ** (Lookup-Free Quantization): a training-free codebook tokenizer (`utils/quantize/lookupFree.py`).

Most training scripts expose `argparse` flags to support quick experiments from the command line.

## Repository Layout

- `HyperLib/`: Manifolds (Sphere/Siegel/Lorentz, etc.), geometry utils, layers, and optimizers.
- `utils/`:
  - `hypformer*.py`: HypFormer backbones and multimodal variants.
  - `manifolds/`: hyperbolic layers and the custom optimizer (`hyp_layer.py`).
  - `quantize/`: codebooks and quantization (`lookupFree.py`).
  - additional improved models and video models.
- Training scripts (single- and multi-modal):
  - Image: `image_fully_minor.py`, `image_fully_backup.py`, `image_fully copy.py`, `image_fully_prototype.py`
  - Text: `text_base.py`, `text_fully.py`
  - Audio: `sound_fully.py`
  - Video: `vedio_fully.py`
  - Multimodal: `multimodal_fully.py`, `multimodal_hypformer.py`
  - Baselines/Prototypes: `base.py`, `image_mapping.py`

## Environment & Dependencies

Recommended: Python 3.9+ and PyTorch 2.0+.

```bash
# Choose the right PyTorch index/CUDA build for your system (or CPU wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install datasets transformers huggingface_hub peft diffusers accelerate
pip install pillow matplotlib scikit-learn tqdm
pip install pycocotools  # when using COCO datasets
```

- Several scripts pull weights from the Hugging Face Hub (e.g., Open-MAGVIT2 tokenizer ckpt). If needed, authenticate first:
```bash
huggingface-cli login
```
- `image_mapping.py` relies on `StableDiffusionPipeline` and generally requires a GPU with sufficient VRAM (bf16/fp16 recommended).

## Data Preparation

- Image (Tiny-ImageNet or CIFAR-10):
  - CIFAR-10 is auto-downloaded to `./data`.
  - Tiny-ImageNet must be downloaded manually and extracted as `/path/to/tiny-imagenet-200/{train,val,test}`.
- Text (WikiText-2): automatically downloaded via `datasets`.
- Audio (Common Voice): automatically downloaded via `datasets`; `sound_fully.py` uses Whisper to tokenize audio into discrete tokens.
- Video (UCF101): prepare extracted frames and annotations (`--data_dir`, `--annotation_path`).
- COCO (for `multimodal_hypformer.py` and `image_mapping.py` examples):
  - expected `coco_dataset/` with `annotations/*.json`, `train2017/`, and `val2017/`.

## Quick Start

Run these from the project root (if your path contains spaces or non-ASCII characters, prefer launching the shell in that directory).

### Image (Tiny-ImageNet/CIFAR-10)
```bash
python image_fully_minor.py --data_dir /path/to/tiny-imagenet-200 --image_size 64 --batch_size 32 --epochs 10
```
Baseline (CIFAR-10 auto-download):
```bash
python base.py --image_size 64 --batch_size 32 --epochs 10
```

### Text (WikiText-2)
```bash
python text_fully.py --epochs 20 --batch_size 64 --lr 5e-3
```
Lightweight baseline:
```bash
python text_base.py --epochs 10 --batch_size 32 --lr 1e-3
```

### Audio (Common Voice + Whisper tokenizer)
```bash
python sound_fully.py --audio_subset "train[:1%]" --batch_size 8 --epochs 5
```

### Video (UCF101)
```bash
python vedio_fully.py \
  --data_dir ./ucf101_frames \
  --annotation_path ./ucf101_annotations \
  --num_frames 8 --batch_size 4 --epochs 10
```

### Multimodal (Image + Text, simplified)
```bash
python multimodal_fully.py --data_dir /path/to/tiny-imagenet-200 --image_size 64 --epochs 3
```

### Multimodal (COCO, larger model config; dataset required)
```bash
python multimodal_hypformer.py --image_size 224 --batch_size 1 --epochs 20
```

### Multimodal Mapping & Generation (Llama 3.2 Vision + Stable Diffusion)
```bash
python image_mapping.py --use_hyperbolic --use_lora --num_epochs 5 --batch_size 2
```
Note: the script loads sizable vision-language and text-to-image models; ensure sufficient GPU and a valid Hugging Face token.

## Common Arguments (examples)

- `--device`: `cuda:0` or `cpu`; scripts auto-detect CUDA availability.
- `--image_size`: encoder/decoder spatial resolution (e.g., 64 or 224).
- `--vocab_size` / `--image_vocab_size`: discrete codebook size for LFQ.
- `--num_layers`, `--num_heads`, `--dropout`: HypFormer Transformer hyperparameters.
- `--lr`, `--weight_decay`, `--epochs`, `--batch_size`: optimization and training controls.

## Notes & Recommendations

- Some scripts are prototypes and contain TODOs or places where target shifting/shape alignment may need attention. Start with small-scale runs.
- If remote downloads are slow, pre-download assets and point the paths to local files.
- Logs typically report cross-entropy and perplexity (PPL). Multimodal examples include simple generation demos.

## Summary

- This project is an unfinished one, a prototype only. We plan to wait until the Hyperbolic Tokenizer is complete before moving forward. Alternatively, we can conduct small-scale experiments using the Euclid Tokenizer.
