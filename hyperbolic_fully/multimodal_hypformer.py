import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image as PILImage
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from utils.quantize.lookupFree import LFQ
from utils.hypformer_multimodality import HypFormer
from utils.improved_model import Encoder
from utils.manifolds.hyp_layer import Optimizer
import traceback
from huggingface_hub import hf_hub_download

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步CUDA执行，更容易定位错误


# ===================== 特殊令牌常量 ======================
IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"

# ===================== 参数解析 ======================
parser = argparse.ArgumentParser(description="Train MultiModal HypFormer")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--image_size', type=int, default=224, help='Resize images to this size')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--vocab_size', type=int, default=2**14, help='Text vocabulary size')
parser.add_argument('--image_vocab_size', type=int, default=2**16, help='Image vocabulary size')
parser.add_argument('--hidden_channels', type=int, default=768, help='Hidden channels for HypFormer')
parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
parser.add_argument('--num_layers', type=int, default=10, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--hyp_lr', type=float, default=2e-5, help='Hyperbolic learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Euclidean optimizer type')
parser.add_argument('--hyp_optimizer_type', type=str, default='radam', help='Hyperbolic optimizer type')
parser.add_argument('--hyp_weight_decay', type=float, default=1e-4, help='Hyperbolic weight decay')
parser.add_argument('--img_loss_weight', type=float, default=1.0, help='imgae loss weight')
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(args.device)

# ===================== 数据处理工具 ======================
# 图像预处理
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 文本处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



# ===================== 图像编码器和量化器 ======================
# CNN编码器配置
ddconfig = {
    "ch": 64,
    "out_ch": 3,
    "in_channels": 3,
    "num_res_blocks": 2,
    "z_channels": 16,
    "ch_mult": (1, 2, 2, 4),
    "resolution": args.image_size,
    "double_z": False,
}

cnn_encoder = Encoder(**ddconfig).to(device).eval()

# LFQ量化器
lfq = LFQ(codebook_size=args.image_vocab_size, dim=16).to(device)
ckpt_path = hf_hub_download(repo_id="TencentARC/Open-MAGVIT2-Tokenizer-262144-Video", filename="video_128_262144.ckpt")
ckpt = torch.load(ckpt_path, map_location=device)
if "state_dict" in ckpt:
    lfq.load_state_dict(ckpt["state_dict"], strict=False)
elif "codebook" in ckpt:
    lfq.register_buffer("codebook", ckpt["codebook"])
lfq.eval()

# ...existing code...

# ...existing code...

# ===================== 数据集定义 ======================
print("使用COCO API加载COCO数据集...")

try:
    import os
    from pycocotools.coco import COCO
    import numpy as np
    from PIL import Image
    import random
    
    # 设置COCO数据集路径 - 根据您的实际下载位置调整
    data_dir = 'coco_dataset'
    train_annotations = os.path.join(data_dir, 'annotations/captions_train2017.json')
    val_annotations = os.path.join(data_dir, 'annotations/captions_val2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')
    val_image_dir = os.path.join(data_dir, 'val2017')
    
    # 检查数据集文件是否存在
    if not os.path.exists(train_annotations) or not os.path.exists(train_image_dir):
        raise FileNotFoundError(f"COCO数据集文件未找到。请确保已下载数据集到 {data_dir} 目录")
    
    # 加载COCO API
    print("加载COCO训练集注释...")
    train_coco = COCO(train_annotations)
    print("加载COCO验证集注释...")
    val_coco = COCO(val_annotations)
    
    # 获取所有图像ID
    train_ids = list(train_coco.imgs.keys())
    val_ids = list(val_coco.imgs.keys())
    
    # 为了测试集，我们从验证集中分割
    random.seed(args.seed)
    random.shuffle(val_ids)
    val_split = int(len(val_ids) * 0.5)
    new_val_ids = val_ids[:val_split]
    test_ids = val_ids[val_split:]
    
    print(f"COCO数据集加载成功！训练集：{len(train_ids)}张图像，验证集：{len(new_val_ids)}张图像，测试集：{len(test_ids)}张图像")
    
    # 创建COCO数据集类
    class COCODataset(Dataset):
        def __init__(self, coco, img_ids, img_dir, transform=None, max_length=128):
            self.coco = coco
            self.img_ids = img_ids
            self.img_dir = img_dir
            self.transform = transform
            self.max_length = max_length
            
        def __len__(self):
            return len(self.img_ids)
        
        def __getitem__(self, idx):
            # 获取图像ID和图像路径
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # 加载图像
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                # 创建空白图像作为替代
                image = Image.new('RGB', (args.image_size, args.image_size), color='black')
            
            # 获取图像描述
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 随机选择一条描述
            if anns and 'caption' in anns[0]:
                caption = random.choice([ann['caption'] for ann in anns])
            else:
                caption = "无描述"
            
            # 应用图像转换
            if self.transform:
                image = self.transform(image)
            
            # 文本编码
            encoding = tokenizer(
                caption,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                "text_ids": encoding['input_ids'].squeeze(0),
                "image": image,
                "caption": caption,
                "image_id": img_id
            }
    
    # 创建数据集
    train_dataset = COCODataset(train_coco, train_ids, train_image_dir, transform=transform)
    val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, transform=transform)
    test_dataset = COCODataset(val_coco, test_ids, val_image_dir, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,  # 可以增加如果您的系统支持
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    print(f"数据加载器创建成功！批次大小: {args.batch_size}")

except Exception as e:
    print(f"COCO数据集加载失败: {e}")
    traceback.print_exc()  # 打印详细错误信息
    
    # 保留合成数据集作为后备方案
    print("无法加载COCO数据集，创建合成数据集进行测试...")
    
    # ... 保留您原来的合成数据集代码 ...
# ===================== 模型定义 ======================
class Args:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = "hyp"
        self.device = device
        self.add_positional_encoding = True
        self.attention_type = "full"
        self.power_k = 2
        self.trans_heads_concat = False

hyp_args = Args()


class MultimodalHypFormer(nn.Module):
    def __init__(self, text_vocab_size, image_vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        
        # 添加特殊token并扩展词表
        special_tokens = {"additional_special_tokens": [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"添加了{num_added}个特殊token")


        
        # 初始化文本和图像嵌入
        self.text_embedding = nn.Embedding(len(tokenizer), embed_dim)  # 使用扩展后的词表
        self.image_embedding = nn.Embedding(image_vocab_size, embed_dim)
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0=文本，1=图像

        if num_added > 0:
             with torch.no_grad():
                input_embeddings = self.text_embedding.weight.data
                input_embeddings_avg = input_embeddings[:-num_added].mean(dim=0, keepdim=True)
                input_embeddings[-num_added:] = input_embeddings_avg


        
        # 特殊token ID
        self.img_start_token_id = len(tokenizer) - 2  # 图像开始标记在词表倒数第二位
        self.img_end_token_id = len(tokenizer) - 1    # 图像结束标记在词表最后一位
        
        
        # 核心HypFormer模型
        self.hypformer = HypFormer(
            in_channels=embed_dim,
            hidden_channels=hidden_dim,
            trans_num_layers=num_layers,
            trans_num_heads=num_heads,
            trans_dropout=dropout,
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=True,
            out_channels=None,  # 多模态模式下不使用
            multimodal=True,  # 启用多模态模式
            text_vocab_size=len(tokenizer),
            image_vocab_size=image_vocab_size,
            args=hyp_args
        )
        
        # 图像token预测器 (用于图像生成任务)
        #self.img_predictor = nn.Linear(embed_dim, image_vocab_size)
        #self.text_output_head = nn.Linear(hidden_dim, len(tokenizer))
        
        # 模态类型记录
        self.register_buffer("text_token_type", torch.zeros(1, dtype=torch.long))
        self.register_buffer("image_token_type", torch.ones(1, dtype=torch.long))
        print(f"特殊token ID: 图像开始={self.img_start_token_id}, 图像结束={self.img_end_token_id}")
        print(f"tokenizer词表大小: {len(tokenizer)}")
         

    def generate_image_from_text(self, text):
        """从文本生成图像"""
        # 对文本进行编码
        encoding = tokenizer(
                text, 
                return_tensors="pt"
            ).to(next(self.parameters()).device)
            
        # 初始图像token (可以是随机的或特殊开始token)
        batch_size = encoding["input_ids"].shape[0]
        image_shape = (batch_size, 256)  # 根据您的需求调整大小
        image_tokens = torch.zeros(image_shape, dtype=torch.long, device=next(self.parameters()).device)
            
        # 使用模型生成图像tokens
        with torch.no_grad():
            text_logits, img_logits, _, _ = self(encoding["input_ids"], image_tokens)
            if img_logits is not None:
                # 从logits获取最可能的token
                pred_tokens = torch.argmax(img_logits, dim=-1)
                    
                # 使用LFQ和CNN解码器将image tokens转换回图像
                # 这部分需要根据您的具体实现调整
                image_tokens = pred_tokens.reshape(batch_size, -1)
                # 这里需要实现token转图像的过程
                    
                return image_tokens, img_logits
        
        return None, None

    def process_batch(self, text_ids, image_tokens):
        """VILA风格处理：先嵌入再拼接"""
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        # 存储所有预处理的嵌入和对应的掩码
        all_embeddings = []
        all_masks = []
        all_types = []  # 用于类型嵌入
        #all_positions = []  # 添加：为每个样本创建位置ID

        for i in range(batch_size):
            curr_text = text_ids[i]
            curr_img_tokens = image_tokens[i]
            
            # 1. 直接嵌入文本部分(安全)
            text_embeddings = self.text_embedding(curr_text)
            text_mask = torch.ones(text_embeddings.size(0), dtype=torch.bool, device=device)
            text_types = torch.zeros(text_embeddings.size(0), dtype=torch.long, device=device)
            
            # 2. 嵌入图像开始标记
            img_start_embed = self.text_embedding(torch.tensor([self.img_start_token_id], device=device))
            img_start_mask = torch.ones(1, dtype=torch.bool, device=device)
            img_start_type = torch.zeros(1, dtype=torch.long, device=device)
            
            # 3. 嵌入图像tokens (使用图像嵌入层)
            img_embeddings = self.image_embedding(curr_img_tokens)
            img_mask = torch.ones(img_embeddings.size(0), dtype=torch.bool, device=device)
            img_types = torch.ones(img_embeddings.size(0), dtype=torch.long, device=device)
            
            # 4. 嵌入图像结束标记
            img_end_embed = self.text_embedding(torch.tensor([self.img_end_token_id], device=device))
            img_end_mask = torch.ones(1, dtype=torch.bool, device=device)
            img_end_type = torch.zeros(1, dtype=torch.long, device=device)
            
            # 拼接所有部分
            sample_embeds = torch.cat([text_embeddings, img_start_embed, img_embeddings, img_end_embed], dim=0)
            sample_mask = torch.cat([text_mask, img_start_mask, img_mask, img_end_mask], dim=0)
            sample_types = torch.cat([text_types, img_start_type, img_types, img_end_type], dim=0)

            text_len = text_embeddings.size(0)
            img_len = img_embeddings.size(0)
            #positions = torch.arange(text_len + img_len + 2, device=device)  # +2是因为<img>和</img>标记
            
            all_embeddings.append(sample_embeds)
            all_masks.append(sample_mask)
            all_types.append(sample_types)
            #all_positions.append(positions)
        
        # 填充到同一长度
        max_len = max([emb.size(0) for emb in all_embeddings])
        padded_embeddings = []
        padded_masks = []
        padded_types = []
        #padded_positions = []
        
        for emb, mask, types in zip(all_embeddings, all_masks, all_types):
            # 计算需要填充的长度
            pad_len = max_len - emb.size(0)
            
            if pad_len > 0:
                # 创建填充嵌入
                padding_emb = torch.zeros((pad_len, emb.size(1)), device=device)
                padded_emb = torch.cat([emb, padding_emb], dim=0)
                
                # 填充掩码和类型
                padding_mask = torch.zeros(pad_len, dtype=torch.bool, device=device)
                padded_mask = torch.cat([mask, padding_mask], dim=0)
                
                padding_type = torch.zeros(pad_len, dtype=torch.long, device=device)
                padded_type = torch.cat([types, padding_type], dim=0)

                #padding_pos = torch.zeros(pad_len, dtype=torch.long, device=device)
                #padded_pos = torch.cat([pos, padding_pos], dim=0)
            else:
                padded_emb = emb
                padded_mask = mask
                padded_type = types
                #padded_pos = pos
            
            padded_embeddings.append(padded_emb)
            padded_masks.append(padded_mask)
            padded_types.append(padded_type)
            #padded_positions.append(padded_pos)
        
        # 堆叠成批次
        return torch.stack(padded_embeddings), torch.stack(padded_types), torch.stack(padded_masks)#torch.stack(padded_positions)
    
    def forward(self, text_ids, image_tokens, training=True):
        """VILA风格前向传播"""
        # 使用特殊token ID定位图像区域
        embeddings, token_types, attention_mask = self.process_batch(text_ids, image_tokens)
        
        # 添加类型嵌入
        type_embeds = self.type_embedding(token_types)
        embeddings = embeddings + type_embeds

        if training and embeddings.size(0) > 1:  # 仅批量大于1且训练模式
            seq_lengths = attention_mask.sum(dim=1)
            if seq_lengths.min() != seq_lengths.max():  # 序列长度不同
                # 按序列长度排序
                sorted_lengths, sorted_indices = torch.sort(seq_lengths, descending=True)
                embeddings = embeddings[sorted_indices]
                attention_mask = attention_mask[sorted_indices]
                token_types = token_types[sorted_indices]
                #position_ids = position_ids[sorted_indices]
                
                # 记录原始顺序
                inverse_indices = torch.argsort(sorted_indices)
                sorted = True
            else:
                sorted = False
        else:
            sorted = False
        
        # 通过HypFormer模型
        text_logits, img_logits, text_mask, img_mask = self.hypformer(
            embeddings, 
            attention_mask=attention_mask,
            token_types=token_types  # 传递模态类型
        )
        
        if True:
        #if batch_idx == 0 and epoch == 0:  # 只打印第一个 batch，第一轮
            print("\n[DEBUG] —— 第一批次诊断日志 ——")
            print(f"Text logits shape: {text_logits.shape}")
            print(f"Text target shape: {target_text.shape}")
            print(f"Text mask shape: {text_mask.shape}")

            # ✅ 文本预测诊断
            if text_mask.any():
                text_indices = torch.where(text_mask)
                valid_row = text_indices[0] < target_text.shape[0]
                valid_col = text_indices[1] < target_text.shape[1]
                valid_indices = valid_row & valid_col
                valid_row_indices = text_indices[0][valid_indices]
                valid_col_indices = text_indices[1][valid_indices]

                if valid_row_indices.numel() > 0:
                    print(f"[DEBUG] 有效 text 预测个数: {valid_row_indices.numel()}")
                    sample_preds = torch.argmax(text_logits[valid_row_indices, valid_col_indices], dim=-1)
                    sample_targets = target_text[valid_row_indices, valid_col_indices]
                    print(f"[DEBUG] Text 预测前10个 token: {sample_preds[:10]}")
                    print(f"[DEBUG] Text 实际目标前10个 token: {sample_targets[:10]}")
                    unique_ids, counts = torch.unique(sample_targets, return_counts=True)
                    print(f"[DEBUG] Text 目标 token 分布: {dict(zip(unique_ids.tolist(), counts.tolist()))}")  
                else:
                    print("[WARNING] ⚠️ 没有有效的 text 预测 token（可能是 mask 问题）")

            # ✅ 图像预测诊断
            if img_logits is not None:
                print(f"Image logits shape: {img_logits.shape}")
                print(f"Target image shape: {target_img.reshape(-1).shape}")
                if img_logits.size(0) > 0:
                    image_preds = torch.argmax(img_logits, dim=-1)
                    image_targets = target_img.reshape(-1)[:img_logits.size(0)]

                    print(f"[DEBUG] Image logits 范围: mean={img_logits.mean().item():.4f}, std={img_logits.std().item():.4f}")
                    print(f"[DEBUG] Image 预测 token（前10）: {image_preds[:10]}")
                    print(f"[DEBUG] Image 实际目标 token（前10）: {image_targets[:10]}")
                    unique_ids, counts = torch.unique(image_targets, return_counts=True)
                    image_token_dist = dict(zip(unique_ids.tolist(), counts.tolist()))
                    print(f"[DEBUG] Image 目标 token 分布: {image_token_dist}")
        
        
        # 如果进行了排序，恢复原始顺序
        if sorted:
            text_logits = text_logits[inverse_indices]
            text_mask = text_mask[inverse_indices]
            # 注意：img_logits可能是None，或者需要特殊处理
            # 如果img_logits不是None，可能需要调整其顺序
        
        # 直接返回由HypFormer处理后的logits和掩码
        return text_logits, img_logits, text_mask, img_mask
            
# 创建模型
model = MultimodalHypFormer(
    text_vocab_size=tokenizer.vocab_size,
    image_vocab_size=args.image_vocab_size,
    embed_dim=args.embed_dim,
    hidden_dim=args.hidden_channels,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout
).to(device)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    model = nn.DataParallel(model)
    # 当使用DataParallel时，需要确保device是主GPU
    device = torch.device('cuda:0')
    
# 定义优化器
optimizer = Optimizer(model, args)
'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer,
    T_0=3,  # 首次重启的epoch数
    T_mult=2,  # 每次重启后周期乘数
    eta_min=1e-6  # 最小学习率
)
'''
text_loss_fn = nn.CrossEntropyLoss()
image_loss_fn = nn.CrossEntropyLoss()


# ===================== 训练过程 ======================
for epoch in range(args.epochs):
    start_time = time.time()
    model.train()
    
    total_loss = 0
    text_tokens_count = 0
    img_tokens_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        text_ids = batch["text_ids"].to(device)
        images = batch["image"].to(device)
        
        # 将图像转为tokens
        with torch.no_grad():
            features = cnn_encoder(images)
            _, _, image_tokens = lfq(features)
            image_tokens = image_tokens.view(images.size(0), -1).long()
        
        # 为下一token预测调整序列
        input_text = text_ids[:, :-1]  # 去掉最后一个token
        target_text = text_ids[:, 1:]  # 目标是序列向后移一位
        
        # 图像token同理
        input_img = image_tokens[:, :-1]
        target_img = image_tokens[:, 1:]
        
        # 前向传播
        text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
        
        # 计算损失
        text_loss = torch.tensor(0.0, device=device, requires_grad=True)
        img_loss = torch.tensor(0.0, device=device, requires_grad=True)
        #print("\n==== 形状信息 ====")
        #print(f"input_text: {input_text.shape}, target_text: {target_text.shape}")
        #print(f"input_img: {input_img.shape}, target_img: {target_img.shape}")
        #print(f"text_logits: {text_logits.shape}, text_mask: {text_mask.shape}")
        #print(f"img_logits shape: {img_logits.shape if img_logits is not None else 'None'}")
        #print(f"img_mask: {img_mask.shape}")
        
        # 文本损失
        if text_mask.any():
          text_indices = torch.where(text_mask)
          
          # 过滤掉超出target_text范围的索引
          valid_row = text_indices[0] < target_text.shape[0]  # 行索引有效
          valid_col = text_indices[1] < target_text.shape[1]  # 列索引有效
          valid_indices = valid_row & valid_col
          
          # 使用有效索引获取预测和目标
          valid_row_indices = text_indices[0][valid_indices]
          valid_col_indices = text_indices[1][valid_indices]
          
          text_preds = text_logits[valid_row_indices, valid_col_indices]
          text_targets = target_text[valid_row_indices, valid_col_indices]
          
          text_loss = text_loss_fn(text_preds, text_targets)
          text_tokens_count += text_targets.numel()
        
        # 图像损失
        if img_mask.any() and img_logits is not None:
            img_targets = target_img.reshape(-1)[:img_logits.size(0)]  # 确保尺寸匹配
            img_loss = image_loss_fn(img_logits, img_targets)
            img_tokens_count += img_targets.numel()
        
        base_alpha = args.img_loss_weight  # 基础权重2.0
        max_alpha = 4.0  # 最大权重
        alpha = base_alpha + (max_alpha - base_alpha) * min(1.0, epoch / 8)  # 在前8个epoch逐渐增加

        loss = text_loss + img_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #scheduler.step()
        

        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            text_loss=f"{text_loss.item():.4f}" if text_tokens_count > 0 else "N/A",
            img_loss=f"{img_loss.item():.4f}" if img_tokens_count > 0 else "N/A"
        )
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    
    # 验证
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text_ids = batch["text_ids"].to(device)
            images = batch["image"].to(device)
            
            # 将图像转为tokens
            features = cnn_encoder(images)
            _, _, image_tokens = lfq(features)
            image_tokens = image_tokens.view(images.size(0), -1).long()
            
            # 为下一token预测调整序列
            input_text = text_ids[:, :-1]
            target_text = text_ids[:, 1:]
            
            input_img = image_tokens[:, :-1]
            target_img = image_tokens[:, 1:]
            
            # 前向传播
            text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
            
            # 计算损失
            v_text_loss = 0
            v_img_loss = 0
            
              # 文本损失
            if text_mask.any():
                text_indices = torch.where(text_mask)
                
                # 过滤掉超出target_text范围的索引
                valid_row = text_indices[0] < target_text.shape[0]  # 行索引有效
                valid_col = text_indices[1] < target_text.shape[1]  # 列索引有效
                valid_indices = valid_row & valid_col
                
                # 使用有效索引获取预测和目标
                valid_row_indices = text_indices[0][valid_indices]
                valid_col_indices = text_indices[1][valid_indices]
                
                text_preds = text_logits[valid_row_indices, valid_col_indices]
                text_targets = target_text[valid_row_indices, valid_col_indices]
                
                v_text_loss = text_loss_fn(text_preds, text_targets)
                      
            # 图像损失
            if img_mask.any() and img_logits is not None:
                img_targets = target_img.reshape(-1)[:img_logits.size(0)]
                v_img_loss = image_loss_fn(img_logits, img_targets)
            
            # 总损失
            v_loss = v_text_loss + alpha * v_img_loss
            val_loss += v_loss.item()
    
    val_avg_loss = val_loss / len(val_loader)

    epoch_time = time.time() - start_time
    #scheduler.step(val_avg_loss)
    # 周期性生成和保存样例图像
    '''
    if (epoch + 1) % 5 == 0:
        os.makedirs("generated_samples", exist_ok=True)
        
        # 生成一些示例图像
        sample_prompts = [
            "a beautiful sunset over the ocean",
            "a dog playing in the park", 
            "a red car driving down the road"
        ]
        
        print("\nGenerating sample images...")
        for i, prompt in enumerate(sample_prompts):
            try:
                generated_image, _ = model.generate_image_from_text(prompt)
                if generated_image is not None:
                    # 转换为PIL图像并保存
                    from torchvision.utils import save_image
                    save_image(
                        generated_image, 
                        f"generated_samples/epoch_{epoch+1}_sample_{i}.png",
                        normalize=True
                    )
                    print(f"  Saved image for prompt: '{prompt}'")
            except Exception as e:
                print(f"  Failed to generate image for prompt '{prompt}': {e}")
      '''

    print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s | "
          f"Train Loss: {avg_loss:.4f}(Text:{text_loss:.4f},Image:{img_loss:.4f}), Val Loss: {val_avg_loss:.4f}(Text:{v_text_loss:.4f},Image:{v_img_loss:.4f})",flush=True)

# ===================== 测试 ======================
model.eval()
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        text_ids = batch["text_ids"].to(device)
        images = batch["image"].to(device)
        
        # 将图像转为tokens
        features = cnn_encoder(images)
        _, _, image_tokens = lfq(features)
        image_tokens = image_tokens.view(images.size(0), -1).long()
        
        # 为下一token预测调整序列
        input_text = text_ids[:, :-1]
        target_text = text_ids[:, 1:]
        
        input_img = image_tokens[:, :-1]
        target_img = image_tokens[:, 1:]
        
        # 前向传播
        text_logits, img_logits, text_mask, img_mask = model(input_text, input_img)
        
        # 计算损失
        t_text_loss = 0
        t_img_loss = 0
        
        if text_mask.any():
            text_indices = torch.where(text_mask)
                
                # 过滤掉超出target_text范围的索引
            valid_row = text_indices[0] < target_text.shape[0]  # 行索引有效
            valid_col = text_indices[1] < target_text.shape[1]  # 列索引有效
            valid_indices = valid_row & valid_col
                
                # 使用有效索引获取预测和目标
            valid_row_indices = text_indices[0][valid_indices]
            valid_col_indices = text_indices[1][valid_indices]
                
            text_preds = text_logits[valid_row_indices, valid_col_indices]
            text_targets = target_text[valid_row_indices, valid_col_indices]
            t_text_loss = text_loss_fn(text_preds, text_targets)
        
        if img_mask.any() and img_logits is not None:
            img_targets = target_img.reshape(-1)[:img_logits.size(0)]
            t_img_loss = image_loss_fn(img_logits, img_targets)
        
        t_loss = t_text_loss + alpha * t_img_loss
        test_loss += t_loss.item()

test_avg_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_avg_loss:.4f}",flush=True)

# ===================== 保存模型 ======================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'args': args,
}, 'multimodal_hypformer.pth')

print("Model saved successfully!")