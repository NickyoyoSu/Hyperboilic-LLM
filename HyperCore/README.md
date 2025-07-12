# HyperCore

**HyperCore** is a framework built upon PyTorch to easily write and train hyperbolic foundation models for a wide range of applications in diverse modalities. [PAPER LINK](https://arxiv.org/abs/2504.08912)

We provide the core modules and functionalities that makes this process simple for users of all backgrounds in differential geometry. These include methods and algorithms for hyperbolic neural networks and foundation models, optimization techiniques, and manifold operations. These come together to enable intuitive constructions of hyperbolic foundation models architectures and pipelines, e.g. [hyperbolic Transformer encoder](example_usage/Hyperbolic_Transformers/lorentz_transformer.py), [hyperbolic ViT](example_usage/Hyperbolic_Transformers/vision_transformer.py), [hyperbolic fine-tuning](example_usage/Fine-Tuning/fine_tuning_example.py), [hyperbolic GraphRAG](example_usage/Hyperbolic_GraphRAG/graphRAG_example.py) and much more (see our paper and tutorials below!). 

- [Framework Highlights](#framework-highlights)
- [Installation](#installation)
- [Quick Start: Build Hyperbolic Foundation Models](#quick-start-build-hyperbolic-foundation-models)
- [Framework Overview](#framework-overview)
- [Implemented Modules and Details](#implemented-modules-and-details)


## Framework Highlights
HyperCore is accessible to experts in hyperbolic deep learning, the more general AI audience, and first-time user of deep learning toolkits alike. Here are some reasons you might want to use HyperCore for building, training, or using foundation models in hyperbolic space!

- **Flexible and Intuitive Foundation Model Support:** HyperCore it is capable of doing much more than reproducing existing models—its components can be effortlessly combined to construct novel hyperbolic foundation models that have yet to be proposed. See [example usage](example_usage) for extensive examples of how to build hyperbolic foundation models with HyperCore!
- **Easy-to-use Modules:** As the API is designed almost identically to Euclidean counterparts, users only need a high-level understanding of foundation models to build hyperbolic ones using HyperCore. All it takes is a few lines to create hyperbolic foundation models components like fully hyperbolic Transformer encoder blocks (see [here](#quick-start-build-hyperbolic-foundation-models) for a quick start guide)!
- **Comprehensive Modules and Model Support:** Unlike anything out there, hyperCore provides a comprehensive and extensive list of essential hyperbolic modules for building a wide range of hyperbolic foundation models for learning across diverse modalities. 

## Installation
For now, the dependencies for HyperCore can be installed via 
```
pip install -r requirements.txt
```
Installation via pip directly is ***coming soon...***

## Quick Start: Build Hyperbolic Foundation Models
In this quick start guide, we highlight the ease of creating and training a hyperbolic foundation model model with HyperCore.
### Creating Your Own Hyperbolic Transformer Encoder Block
In the first glimpse of HyperCore, we build the encoder block of a hyperbolic Transformer, using hyperbolic-tailored modules such as [LorentzMultiheadAttention](hypercore/nn/attention/lorentz_former_conv.py) for multi-head attention and [LResNet](hypercore/nn/conv/conv_util_layers.py) for residual connection. 

```python
import torch
import torch.nn as nn
import hypercore.nn as hnn

class LTransformerBlock(nn.Module):
    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        out_dim = d_model - 1
        mlp_dim = d_model * 4 - 1
        self.manifold = manifold
        self.attn = hnn.LorentzMultiheadAttention(manifold, dim_per_head, dim_per_head, n_head, attention_type='full', trans_heads_concat=True)
        self.ln_1 = hnn.LorentzLayerNorm(manifold, out_dim)
        self.mlp = nn.Sequential(
            hnn.LorentzLinear(manifold, d_model, mlp_dim),
            hnn.LorentzActivation(manifold, activation=nn.GELU()),
            hnn.LorentzLinear(manifold, mlp_dim+1, out_dim),
        )
        self.ln_2 = hnn.LorentzLayerNorm(manifold, out_dim)
        self.res1 = hnn.LResNet(manifold, use_scale=True)
        self.res2 = hnn.LResNet(manifold, use_scale=True)

    def forward(self, x, attn_mask=None):
        # x: embedded features matrix of shape [batch, seq_len, dim_per_head * num_heads]
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, mask=attn_mask)
        x = self.res1(x, ax)
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x
```

For more examples of how to employ hyperbolic foundation model components in downstream tasks, please see [example usages](example_usage)

### Training a Hyperbolic Vision Transformer
Let's take a quick look at how to easily train a pre-built hyperbolic foundation model with HyperCore, by looking at an example of training a Lorentzian vision Transformer (LViT, see [our paper](https://arxiv.org/abs/2504.08912) for more details) on classifying images in the CIFAR10 dataset. 

In particular, since hyperbolic parameters require special update rules (see [here](https://arxiv.org/abs/1810.00760) for more details), HyperCore automatically sets up the optimizers to update Euclidean and hyperbolic parameters. Additionally, we can use different optimization schemes for parameters on different manifolds. 

Functionalities like these make it seamless to transition from training Euclidean foundation models to hyperbolic ones.

![LViT Module Architecture](assets/HViT.png)

```python
from torchvision import datasets, transforms
import torch
from hypercore.manifolds import Lorentz
from hypercore.optimizers import Optimizer, LR_Scheduler
from hypercore.models import LViT
import numpy as np

# prepare the dataset as usual
train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

train_set = datasets.CIFAR10('hypercore/data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10('hypercore/data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=8, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=8, pin_memory=True, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the hyperbolic manifold with curvature -1.0
manifold = Lorentz(1.0)
model = LViT(manifold_in=manifold, manifold_hidden=manifold, manifold_out=manifold).to(device)

# Initialize the optimizers. Note that we have separate optimizers for Euclidean and hyperbolic parameters
optimizer = Optimizer(model, euc_optimizer_type='adamW', euc_lr=1e-4, euc_weight_decay=1e-2, hyp_optimizer_type='radam', hyp_lr=1e-4, 
    hyp_weight_decay=1e-4, stabilize=1)

# Initiate the learning rate scheduler for each optimizers
lr_scheduler = LR_Scheduler(optimizer_euc=optimizer.optimizer[0], euc_lr_reduce_freq=30, euc_gamma=0.1, hyp_gamma=0.1, hyp_lr_reduce_freq=30, optimizer_hyp=optimizer.optimizer[1])

criterion = torch.nn.CrossEntropyLoss()
for epoch in range(0, 300):
        model.train()
        losses = []
        acc1 = []
        acc5 = []

        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Function to calculate top-k classification accuracy 
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

model.eval()
acc1 = []
acc5 = []

for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    top1, top5 = accuracy(logits, y, topk=(1, 5))
    acc1.append(top1.item(), x.shape[0])
    acc5.append(top5.item(), x.shape[0])

acc1_test = np.mean(acc1)
acc5_test = np.mean(acc5)
print("Results: Acc@1={:.4f}, Acc@5={:.4f}".format(acc1_test, acc5_test))
```

## Framework Overview
HyperCore is a framework that supports constructing, developing, and evaluating hyperbolic foundation models from multiple levels, from fundamental training schemes to modules with hyperbolic layers to the models and downstream tasks themselves. See framework snapshot for visual organization. 

- **Hyperbolic manifold and optimizers:** These are essential building blocks of training any hyperbolic foundation model. HyperCore builds on top of the well-optimized manifolds and optimizers of [Geoopt](https://github.com/geoopt/geoopt). HyperCore extends the manifolds to incorporate more fundamental operations, e.g. [hyperbolic entailment cones](https://arxiv.org/abs/2304.09172). The optimizers also allow for seemless transition to hyperbolic training schemes (see the [above section](#training-a-hyperbolic-vision-transformer)).
- **Hyperbolic Modules and Layers:** HyperCore implemented an extensive list of modules and layers from current research to support building both existing and new hyperbolic foundation models. Additionally, novel hyperbolic modules were developed specifically for HyperCore for building hyperbolic foundation models, such as performing hyperbolic RoPE through pseudo Lorentzian rotations. 
- **Lower-level Hyperbolic Models:** HyperCore supports building-block hyperbolic neural networks (e.g. GNNs, CNNs, etc) and hyperbolic foundation models (e.g. ViT and Transformer), implemented in [models](hypercore/models) with examples in [example_usage](example_usage).
- **Higher-level Hyperbolic Models:** With the lower-level hyperbolic models, HyperCore also supports building higher-level hyperbolic foundation models, such as fully hyperbolic CLIP models, hyperbolic GraphRAG models, and hyperbolic fine-tuning. These are implemented in [models](hypercore/models) with examples in [example_usage](example_usage).
- **Downstream Tasks Support:** Many hyperbolic operations for downstream tasks are also implemented, such as hyperbolic loss functions.

![Framework Snapshot](assets/snapshot.png)

## Implemented Modules and Details
HyperCore implements the following list of hyperbolic modules and layers:

<details>
<summary>Manifolds and Optimizers (click to expand)</summary>

- Lorentz hyperboloid ([source code](hypercore/manifolds/lorentzian.py)) and Poincare Ball model ([source code](hypercore/manifolds/poincare.py)), based on Kochurov et al., [Geoopt: Riemannian Optimization in PyTorch](https://arxiv.org/abs/2005.02819)
- Riemannian Adam ([source code](hypercore/optimizers/radam.py)) and SGD ([source code](hypercore/optimizers/rsgd.py)) optimizers from Bécigneul et al., [Riemannian Adaptive Optimization Methods](https://arxiv.org/abs/1810.00760), based on Kochurov et al., [Geoopt: Riemannian Optimization in PyTorch](https://arxiv.org/abs/2005.02819)
</details>

<details>
<summary>Hyperbolic Linear Layers(click to expand)</summary>

- HyboNet linear layer [source code](hypercore/nn/linear/hybonet_linear.py) from Chen et al., [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686)
- Hypformer linear layer [source code](hypercore/nn/linear/lorentz_linear.py) from Yang et al., [Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290)
- Tangent-space-based linear layer [source code](hypercore/nn/linear/hnn_layers.py) from Ganea et al., [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- Poincare linear layer [source code](hypercore/nn/linear/poincare_linear.py) from Shimizu et al., [Hyperbolic Neural Networks++](https://arxiv.org/abs/2006.08210)
</details>

<details>
<summary>Hyperbolic Activation Layers(click to expand)</summary>

- Fully hyperbolic (Lorentz) activation layer [source code](hypercore/nn/conv/conv_util_layers.py) from Yang et al., [Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290)
- Tangent-space-based activation layer [source code](hypercore/nn/linear/hnn_layers.py) from Ganea et al., [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
</details>

<details>
<summary>Hyperbolic Classification Layers(click to expand)</summary>

- Fully hyperbolic Lorentzian MLR layer [source code](hypercore/nn/conv/lorentz_MLR.py) from Bdeir et al., [Fully Hyperbolic Convolutional Neural Networks for Computer Vision](https://arxiv.org/abs/2303.15919)
- Poincare MLR layer [source code](hypercore/nn/conv/poincare_MLR.py) from van Spengler et al., [Poincare ResNet](https://arxiv.org/abs/2303.14027)
</details>

<details>
<summary>Hyperbolic Convolutional & Residual Layers(click to expand)</summary>

- Fully hyperbolic Lorentzian convolution layer [source code](hypercore/nn/conv/lorentz_convolution.py) from Bdeir et al., [Fully Hyperbolic Convolutional Neural Networks for Computer Vision](https://arxiv.org/abs/2303.15919), modified based on linear layer from Hypformer (Yang et al.,)
- Poincare convolution layer [source code](hypercore/nn/conv/poincare_convolution.py) from van Spengler et al., [Poincare ResNet](https://arxiv.org/abs/2303.14027)
- Fully hyperbolic Lorentzian residual layer (LResNet) [source code](hypercore/nn/conv/conv_util_layers.py) from He et al., [Lorentzian Residual Neural Networks](https://arxiv.org/abs/2412.14695)
- Parallel-transport-space-based residual layer [source code](hypercore/nn/linear/hnn_layers.py) from van Spengler et al., [Poincare ResNet](https://arxiv.org/abs/2303.14027)
</details>

<details>
<summary>Hyperbolic Normalization & Pooling Layers(click to expand)</summary>

- Lorentzian batch normalization from [source code](hypercore/nn/conv/poincare_batch_norm.py) from Bdeir et al., [Fully Hyperbolic Convolutional Neural Networks for Computer Vision](https://arxiv.org/abs/2303.15919)
- Poincare batch normalization from [source code](hypercore/nn/conv/poincare_batch_norm.py) from van Spengler et al., [Poincare ResNet](https://arxiv.org/abs/2303.14027)
- Fully hyperbolic Lorentzian layer normalization [source code](hypercore/nn/conv/conv_util_layers.py) from Yang et al., [Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290)
- Fully hyperbolic Lorentzian batch normalization [source code](hypercore/nn/conv/lorentz_batch_norm.py) from He et al., [Lorentzian Residual Neural Networks](https://arxiv.org/abs/2412.14695)
- Fully hyperbolic Lorentzian global pooling layer [source code](hypercore/nn/conv/lorentz_pooling.py) from Bdeir et al., [Fully Hyperbolic Convolutional Neural Networks for Computer Vision](https://arxiv.org/abs/2303.15919)
</details>

<details>
<summary>Hyperbolic Attention Mechanism and Transformer-related Modules(click to expand)</summary>

- Lorentzian self-attention layer [source code](hypercore/nn/attention/lorentz_former_conv.py) from Chen et al., [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686), modified based on linear layer from Hypformer (Yang et al.,)
- Lorentzian linear attention layer [source code](hypercore/nn/attention/lorentz_former_conv.py) from Yang et al., [Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290)
- Poincare self-attention layer [source code](hypercore/nn/attention/poincare_attention.py) from van Spengler et al., [Poincare ResNet](https://arxiv.org/abs/2303.14027)
- Lorentzian word embedding [source code](hypercore/nn/attention/lorentz_word_emb.py). Developed for HyperCore, optionally allows for positional encoding enabled by LResNet (He et al.,)
- Lorentzian patch embedding [source code](hypercore/nn/attention/patch_embedding.py). Developed for HyperCore, based on Lorentzian convolutional layer from HCNN (Bdeir et al.,)
- Lorentzian RoPE mechanism [source code](hypercore/nn/attention/lorentz_former_conv.py). Developed for HyperCore, based on pseudo Lorentzian rotation, incorporated into Lorentzian self-attention layers
- Lorentzian relative positional encoding [source code](hypercore/nn/attention/positional_encoding.py) from Yang et al., [Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290)
</details>

<details>
<summary>Hyperbolic Graph and Neighborhood Aggregation(click to expand)</summary>

- HGCN GCN layer [source code](hypercore/nn/graph_conv/hgcn_conv.py) from Chami et al., [Hyperbolic Graph Convolutional Neural Networks](https://arxiv.org/abs/1910.12933)
- HGNN GNN layer [source code](hypercore/nn/graph_conv/hgnn_conv.py) from Liu et al., [Hyperbolic Graph Neural Networks](https://arxiv.org/abs/1910.12892)
- HyboNet GCN layer [source code](hypercore/nn/graph_conv/hyobnet_conv.py) from Chen et al., [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686)
- HGAT GAT layer [source code](hypercore/nn/graph_conv/hgat_conv.py) from Zhang et al., [Hyperbolic Graph Attention Network](https://arxiv.org/abs/1912.03046)
- LGCN GCN layer [source code](hypercore/nn/graph_conv/lgcn_conv.py) from Zhang et al., [Lorentzian Graph Convolutional Networks](https://arxiv.org/abs/2104.07477)
- H2HGCN GCN layer [source code](hypercore/nn/graph_conv/h2h_conv.py) from Dai et al., [A Hyperbolic-to-Hyperbolic Graph Convolutional Network](https://arxiv.org/abs/2104.06942)
- GIL GAT layer [source code](hypercore/nn/graph_conv/gil_conv.py) from Zhu et al., [Graph Geometry Interaction Learning](https://arxiv.org/abs/2010.12135)
</details>