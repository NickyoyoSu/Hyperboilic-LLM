# Hyperbolic Foundation Models Repository

**A comprehensive codebase for ongoing research in hyperbolic geometry for deep learning and foundational models across multiple modalities**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


This repository provides a (in progress) ecosystem for building, training, and deploying foundation models in hyperbolic space. It includes frameworks, training scripts, tokenizers, and tools for working with hyperbolic geometry across text, image, audio, video, and multimodal tasks.

##  Key Features

- **HyperCore Framework**: Comprehensive PyTorch-based framework for hyperbolic neural networks
- **HypFormer Architecture**: Hyperbolic Transformer backbone for multimodal learning
- **Multi-Modal Support**: Text, image, audio, video, and cross-modal applications
- **Hyperbolic Tokenizers**: VQ-VAE and autoencoder implementations in hyperbolic space
- **Model Integration**: Support for popular models (Llama, Stable Diffusion, CLIP, etc.)
- **Efficient Training**: LoRA fine-tuning, multi-GPU support, and optimized implementations

## Repository Structure

```
├── HyperCore/           # Core hyperbolic deep learning framework
├── hyperbolic_fully/    # Full hyperbolic modeling implementations
├── hyperbolic_mapping/  # Hyperbolic mapping for language and multimodal models
└── tokenizer/          # Hyperbolic VQ-VAE and tokenizers
```

###  Component Overview

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **[HyperCore](HyperCore/)** | PyTorch framework for hyperbolic foundation models | Manifolds, optimizers, layers, comprehensive modules |
| **[hyperbolic_fully](hyperbolic_fully/)** | Complete implementations across modalities | HypFormer, LFQ tokenization, training scripts |
| **[hyperbolic_mapping](hyperbolic_mapping/)** | Mapping heads for existing models | Text/image alignment, LoRA fine-tuning, multi-GPU |
| **[tokenizer](tokenizer/)** | Hyperbolic tokenization and compression | VQ-VAE, autoencoders, Lorentz manifold operations |

## Quick Start

### Prerequisites

- Python 3.9+ 
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/NickyoyoSu/Hyperboilic-LLM.git
cd Hyperboilic-LLM
```

2. **Install HyperCore framework:**
```bash
cd HyperCore
pip install -r requirements.txt
export PYTHONPATH="${PWD}:${PYTHONPATH}"
cd ..
```

3. **Install additional dependencies for specific components:**
```bash
# For hyperbolic_mapping
pip install transformers datasets peft diffusers accelerate pillow scikit-learn

# For tokenizer (if using specific requirements)
pip install matplotlib tqdm opencv-python

# For hyperbolic_fully 
pip install whisper librosa torch-audio
```

### Basic Usage Examples

#### 🔧 HyperCore Framework
```python
import torch
import hypercore.nn as hnn
from hypercore.manifolds import Lorentz

# Create hyperbolic manifold
manifold = Lorentz()

# Build hyperbolic Transformer block
class HyperbolicTransformer(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = hnn.LorentzMultiheadAttention(manifold, d_model//n_heads, d_model//n_heads, n_heads)
        self.norm = hnn.LorentzLayerNorm(manifold, d_model-1)
        self.mlp = hnn.LorentzLinear(manifold, d_model, d_model*4-1)
```

#### Image Classification (CIFAR-10)
```bash
cd hyperbolic_fully
python image_fully_minor.py --dataset cifar10 --image_size 64 --batch_size 32 --epochs 10
```

#### Text Modeling (WikiText-2)
```bash
cd hyperbolic_fully  
python text_fully.py --epochs 20 --batch_size 64 --lr 5e-3
```

#### Audio Processing
```bash
cd hyperbolic_fully
python sound_fully.py --audio_subset "train[:1%]" --batch_size 8 --epochs 5
```

#### Video Understanding (UCF101)
```bash
cd hyperbolic_fully
python vedio_fully.py --data_dir ./ucf101_frames --num_frames 8 --batch_size 4 --epochs 10
```

#### Multimodal Learning
```bash
cd hyperbolic_mapping
python multimodal_mapping.py --use_hyperbolic --use_lora --num_epochs 5 --batch_size 2
```

#### Hyperbolic Tokenizer Training
```bash
cd tokenizer
# Pretrain autoencoder
python train_autoencoder.py --dataset CIFAR10 --epochs 50

# Train hyperbolic VQ-VAE
python main_hyp.py --dataset CIFAR10 --n_embeddings 1024 --n_updates 10000
```

## Detailed Documentation

Each component has comprehensive documentation:

- **[HyperCore Framework](HyperCore/README.md)**: Complete guide to the hyperbolic deep learning framework
- **[Hyperbolic Fully](hyperbolic_fully/README.md)**: Multi-modal implementations and training scripts  
- **[Hyperbolic Mapping](hyperbolic_mapping/README.md)**: Language and multimodal mapping with existing models
- **[Tokenizer](tokenizer/README.md)**: Hyperbolic VQ-VAE and tokenization systems

## Research Applications

This repository enables research in:

- **Foundation Models**: Hyperbolic Transformers, Vision Transformers, and hybrid architectures
- **Multimodal Learning**: Cross-modal alignment and generation in hyperbolic space
- **Geometric Deep Learning**: Leveraging hyperbolic geometry for hierarchical representations
- **Efficient Training**: Parameter-efficient fine-tuning with hyperbolic mappings
- **Tokenization**: Novel compression schemes using hyperbolic embeddings

## Supported Models & Integrations

- **Language Models**: Llama 3.2 (1B, 11B), GPT-style architectures
- **Vision Models**: ViT, ResNet variants, multimodal transformers
- **Multimodal**: CLIP, Janus-Pro-7B, MLLama, Stable Diffusion
- **Audio**: Whisper tokenizer integration
- **Datasets**: CIFAR-10, ImageNet, WikiText, COCO, Common Voice, UCF101

## Hyperbolic Geometry Benefits

Hyperbolic geometry offers unique advantages for deep learning:

- **Natural Hierarchies**: Better representation of tree-like and hierarchical data
- **Improved Embeddings**: More efficient use of embedding space
- **Geometric Inductive Bias**: Built-in understanding of hierarchical relationships
- **Scale Invariance**: Better handling of multi-scale features

## Development & Contribution

### Testing Your Installation

```bash
# Test HyperCore installation
cd HyperCore
python -c "import hypercore; print('HyperCore imported successfully')"

# Run a quick training example
cd ../hyperbolic_fully
python base.py --epochs 1 --batch_size 8  # Quick baseline test
```

### Common Issues & Solutions

- **CUDA Issues**: Ensure PyTorch CUDA version matches your CUDA installation
- **Memory Issues**: Reduce batch size, use gradient checkpointing, or enable mixed precision
- **Import Errors**: Verify `PYTHONPATH` includes HyperCore directory
- **Dataset Issues**: Check data paths and ensure sufficient disk space

##  Acknowledgments

This project builds upon and integrates with:

- **PyTorch & Hugging Face Ecosystem**: Core deep learning infrastructure
- **Geoopt**: Optimized manifolds and optimizers for hyperbolic geometry  
- **Popular Models**: Meta Llama, Stability AI, DeepSeek, OpenAI Whisper
- **Datasets**: MS COCO, ImageNet, WikiText, Common Voice, UCF101
- **Research Community**: Contributions from hyperbolic deep learning researchers

##  License

This project is licensed under the MIT License - see the [HyperCore/LICENSE](HyperCore/LICENSE) file for details.

##  Related Work

- [Hyperbolic Neural Networks](https://proceedings.neurips.cc/paper/2018/hash/dbab2adc8f9d078009ee3fa810265-Abstract.html)
- [Lorentzian Distance Learning](https://arxiv.org/abs/2006.10160) 
- [Hyperbolic Attention Networks](https://arxiv.org/abs/2005.00749)
- [Poincaré Embeddings](https://proceedings.neurips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229-Abstract.html)

