import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image as PILImage

from utils.quantize.lookupFree import LFQ
from utils.hypformer_backup import HypFormer
from utils.improved_model import Encoder, Decoder
from utils.manifolds.hyp_layer import Optimizer
import shutil

# ===================== Argument Parser ======================
parser = argparse.ArgumentParser(description="Train HypFormer on CIFAR-10")

parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--image_size', type=int, default=64, help='Resize images to this size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training set')
parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for HypFormer')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--vocab_size', type=int, default=2**18, help='Vocabulary size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hyp_lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Euclidean optimizer type')
parser.add_argument('--hyp_optimizer_type', type=str, default='radam', help='Hyperbolic optimizer type')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Euclidean weight decay')
parser.add_argument('--hyp_weight_decay', type=float, default=0.0, help='Hyperbolic weight decay')
parser.add_argument('--data_dir', type=str, default='/Users/nick/Documents/yale/科研/Dataset/tiny-imagenet-200', help='Path to Tiny-ImageNet directory')

args = parser.parse_args()

# ============ Set Random Seed for Reproducibility ===========
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# ========= Define Encoder as LFQ Input ======================

ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 18, ##TODO
    "ch_mult": (1, 2, 2, 4),
    "resolution": args.image_size,
    "double_z": False,
}

cnn_encoder = Encoder(**ddconfig).to(device).eval()

# ========= Load CIFAR-10 Dataset (torchvision) ==============
def organize_val_set(val_dir):
    images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    new_val_dir = os.path.join(val_dir, "organized")
    if not os.path.exists(new_val_dir):
        os.makedirs(new_val_dir)
    # 读取 txt 文件
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name = parts[0]
            wnid = parts[1]
            wnid_folder = os.path.join(new_val_dir, wnid)
            if not os.path.exists(wnid_folder):
                os.makedirs(wnid_folder)
            src = os.path.join(images_dir, img_name)
            dst = os.path.join(wnid_folder, img_name)
            # 这里选择复制，也可以用 move（shutil.move）
            shutil.copy(src, dst)
    return new_val_dir

val_dir = os.path.join(args.data_dir, "val")
organized_val_dir = organize_val_set(val_dir)
print(f"Organized validation set saved in: {organized_val_dir}")
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集使用 ImageFolder，目录结构应为 tiny-imagenet-200/train/<wnid>/xxx.jpg
train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "train"), transform=transform)
# 验证集使用刚刚整理好的 organized 文件夹
val_dataset = datasets.ImageFolder(root=organized_val_dir, transform=transform)
# 测试集：由于官方测试集没有标签，你可以创建一个自定义数据集，或者暂时忽略测试集
# 这里假设测试集图像在 tiny-imagenet-200/test/images 下，我们简单用 ImageFolder 加载，但注意需要手动将所有图片放到同一文件夹下并赋予一个虚拟标签
test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images, Test set: {len(test_dataset)} images")
# ========= Initialize LFQ and Embedding Layer ===============
### dim must be the same as z_channel; and the LFQ codebook is directly figured out, no need to train
lfq = LFQ(codebook_size=args.vocab_size, dim=16).to(device)
ckpt = torch.load("/Users/nick/Documents/yale/科研/代码/hpyllama/pretrain256_262144.ckpt", map_location=device)
# 如果 ckpt 中包含 'state_dict' 键，则加载 state_dict；否则直接检查是否有 codebook
if "state_dict" in ckpt:
    lfq.load_state_dict(ckpt["state_dict"], strict=False)
elif "codebook" in ckpt: 
    lfq.register_buffer("codebook", ckpt["codebook"])
lfq.eval()
##TODO 1: eval()?
##TODO 2: No need to set vocab_size if dim, as it is a default setting


# ========= Define HypFormer =================================
class Args:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = "hyp" ###TODO
        self.device = device
        self.add_positional_encoding = True
        self.attention_type = "full"
        self.power_k = 2
        self.trans_heads_concat = False

hyp_args = Args()

##TODO Add casual mask
model = HypFormer(
    in_channels=18,
    hidden_channels=args.hidden_channels,
    out_channels=args.vocab_size,
    trans_num_layers=args.num_layers,
    trans_num_heads=args.num_heads,
    trans_dropout=args.dropout,
    trans_use_bn=True,
    trans_use_residual=True,
    trans_use_weight=True,
    trans_use_act=True,
    args=hyp_args
).to(device)

### Need to add another dimension for time axis
embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=18).to(device)  ##TODO embedding_dim has been modified

# =========== Training Loop ==================================
### TODO: Refer to manifolds/layer.py and geoopt/tensor.py
### optimizer = layer.Optimizer(model, args) or likewise
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) ## TODO
optimizer = Optimizer(model, args)

##scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    total_tokens = 0

    ### batch: (images. label)
    for batch in train_loader: 
        images, _ = batch ## temporarily omit label
        images = images.to(device)

        with torch.no_grad():
            ## images: (batch_size, channels, height, width) = (32, 3, 64, 64)
            ### features: (batch_size, z_channels, final_height, final_width) = (32, 18, 8, 8)
            features = cnn_encoder(images) 
            quantized, entropy_aux_loss, tokenized_x_raw = lfq(features) ### TODO tokenized_x HAS BEEN MODIFIED
            #print(quantized.shape)          ### (32, 18, 8, 8)
            #print(tokenized_x_raw.shape)    ### batch_size 32 * final_height 8 * final_width 8 = (2048)
            #print(entropy_aux_loss)         ### scalar
            
            ### tokenized_x: (batch_size, fh * fw) = (32, 64)
            tokenized_x = tokenized_x_raw.view(images.size(0), -1).long() 
            
            ####################################TODO################################################
            ### PREVIOUS CODES, which is where the problem lies:
            ### tokenized_x: (batch_size, fh * fw) = (batch_size, token) = (32, 64)
            ### embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=18).to(device)
            tokenized_x_embed = embedding_layer(tokenized_x) 
            ### Current tokenized_x_embed: EUC (batch_size, token, embedding_dim) 
            ########################################################################################

        ### Refer to hypformer codes! 
        ### Inside Hypformer: decoder_type=`hyp` (lpogmap0 in `euc`)
        ###  -> HypLinear   x: HYP (batch_size, token, embedding_dim + 1) => manifolds_out? TODO
        ###  -> HypCLS      dist: HYP (batch_size, token, out_channels) TODO
        ###    --> 'prob'   output = softmax-like probability: (batch_size, token, out_channels)
        ### This is softmax-like, not real softmax cuz the sum might NOT be 1!!!
        ### TODO: We need to add arg 'prob' inside HypCLS!!!
        
        # Other ref: https://github.com/kschwethelm/HyperbolicCV/blob/main/code/lib/lorentz/layers/LMLR.py ?
        
        # output: (batch_size, token, out_channels)
        output = model(tokenized_x_embed)
        b, seq_len, vocab_size = output.shape
        output = output.view(b * seq_len, vocab_size) # new_output in the notation
        ### new output: (batch_size * token, out_channels)
        ### === (batch_size * seq_len, out_channels) = (32 * 8 * 8, 2**18)
        ### TODO: Set out_channels as vocab_size
        
        ###############################EG######################################
        ### output: (batch_size, seq_len, out_channels) = (2, 3, 2)
        ### vocab_size = 2**2
        ### output = [
        ###  [ ### batch 1
        ###    [0.1, 0.3, 0.2, 0.2],      ### seq0
        ###    [0.2, 0.2, 0.5, 0.1],      ### seq1
        ###    [0.5, 0.1, 0.1, 0.3]       ### seq2
        ###  ],
        ###  [ ### batch 2
        ###    [0.0, 0.2, 0.5, 0.1],
        ###    [0.3, 0.3, 0.1, 0.3],
        ###    [0.1, 0.2, 0.2, 0.5]
        ###  ]
        ###]
        ### new output: (batch_size * seq_len, out_channels) = (6, 2)
        ###  [ [0.1, 0.3, 0.2, 0.2],   -- pred --> 1
        ###    [0.2, 0.2, 0.5, 0.1],   -- pred --> 2
        ###    [0.5, 0.1, 0.1, 0.3],   -- pred --> 0
        ###    [0.0, 0.2, 0.5, 0.1],
        ###    [0.3, 0.3, 0.1, 0.3],
        ###    [0.1, 0.2, 0.2, 0.5]
        ###  ]
        ###
        ###
        ###  tokenized_x_raw: (batch_size * seq_len,)
        ###  [0, 2, 3, 1, 2, 2] 
        ###############################EG END###################################
        
        ### RECALL tokenized_x: (batch_size, fh * fw) = (batch_size, seq_len) = (32, 64)
        target = tokenized_x[:, 1:1+seq_len].reshape(b * seq_len) ### TODO: THIS IS WRONG
        
        ### Use ready-made cross entropy 

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
        
        

    avg_loss = total_loss / total_tokens
    train_perplexity = math.exp(avg_loss)

    # ============== Validation after each epoch =================
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)

            features = cnn_encoder(images)
            _, _, tokenized_x = lfq(features)
            tokenized_x = tokenized_x.view(images.size(0), -1).long()
            tokenized_x_embed = embedding_layer(tokenized_x)
            output = model(tokenized_x_embed)

            b, seq_len, vocab_size = output.shape
            output = output.view(b * seq_len, vocab_size)
            target = tokenized_x[:, 1:1+seq_len].reshape(b * seq_len)

            loss = loss_fn(output, target)
            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

    val_avg_loss = total_loss / total_tokens
    val_perplexity = math.exp(val_avg_loss)
    
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.4f}, Vad PPL: {val_perplexity:.4f}")

# =========== Test Loop (Next Token Prediction) ==========
model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch in test_loader:
        images, _ = batch
        images = images.to(device)

        features = cnn_encoder(images)
        _, _, tokenized_x = lfq(features)
        tokenized_x = tokenized_x.view(images.size(0), -1).long()
        tokenized_x_embed = embedding_layer(tokenized_x)
        output = model(tokenized_x_embed)
        b, seq_len_wrong, vocab_size = output.shape
        output = output.view(b * seq_len_wrong, vocab_size)

        target = tokenized_x[:, 1:1+seq_len_wrong].reshape(b * seq_len_wrong)
        loss = loss_fn(output, target)

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

test_avg_loss = total_loss / total_tokens
test_perplexity = math.exp(test_avg_loss)
print(f"Test Perplexity: {test_perplexity:.4f}")