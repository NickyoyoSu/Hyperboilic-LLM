import sys
import os
import time
import math
import argparse
import traceback
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, MllamaForConditionalGeneration, get_linear_schedule_with_warmup

from torchvision import transforms

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def get_parser():
    parser = argparse.ArgumentParser(description="Train a multimodal vision-language model with Hyperbolic mapping")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for the dataset to load")
    parser.add_argument("--image_size", type=int, default=224, help="Size of input images (224 for MLLama)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use (e.g., 'cuda', 'cuda:1', 'cpu')")
    parser.add_argument("--gpu_ids", type=str, default="0", 
                    help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--use_parallel", action="store_true",
                    help="Use DataParallel for multi-GPU training")
    parser.add_argument("--curvature_lr", type=float, default=1e-6, 
                    help="Learning rate for the curvature parameter")
    parser.add_argument("--save_interval", type=int, default=1, 
                    help="Save checkpoint every N epochs")
    parser.add_argument("--eval_steps", type=int, default=2000, 
                    help="Evaluate model every N steps")
    parser.add_argument("--resume", type=str, default="", 
                    help="从检查点恢复训练的路径")
    parser.add_argument("--start_epoch", type=int, default=1,
                    help="恢复训练的起始epoch")
    parser.add_argument("--mode", type=str, default="train", 
                      choices=["train", "eval", "demo", "research"],
                      help="运行模式: train=训练, eval=评估, demo=演示, research=研究")
    return parser
    

args = get_parser().parse_args()

# 设置随机种子以确保可重复性
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

USE_HYPERBOLIC = args.use_hyperbolic
USE_LORA = args.use_lora
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BASE_LR = args.learning_rate
MAX_LENGTH = args.max_length
DEVICE = args.device
MAX_SAMPLES = args.max_samples
IMAGE_SIZE = args.image_size
CURVATURE_LR = args.curvature_lr

# ===================== 加载视觉语言模型 =====================
print(f"{'='*40}\n加载模型: Deepseek-AI/Janus-Pro\n{'='*40}")

model_name = "deepseek-ai/Janus-Pro-7B"
device = torch.device(DEVICE)

# 加载处理器和模型
processor = VLChatProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

# 模型加载与并行处理
if args.use_parallel and torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU 进行训练")
    # 先加载模型（不移动到设备）
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # 应用LoRA (如果需要)
    if USE_LORA:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Janus 的注意力模块名称
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        # 冻结原始模型参数
        for param in model.parameters():
            param.requires_grad = False
    
    # 然后包装成DataParallel
    model = nn.DataParallel(model)
    model = model.to(device)
    print(f"DataParallel启用，使用GPU: {list(range(torch.cuda.device_count()))}")
else:
    print(f"使用单个设备进行训练: {DEVICE}")
    # 单GPU模式
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # 应用LoRA或冻结参数
    if USE_LORA:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        # 冻结原始模型参数
        for param in model.parameters():
            param.requires_grad = False

# ============== 自定义多模态映射头 =================
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        # 处理DataParallel包装的模型
        if isinstance(base_model, nn.DataParallel):
            config = base_model.module.config
        else:
            config = base_model.config
        
        # 获取vocab_size
        self.vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else len(tokenizer)
        
        # 获取hidden_size
        self.hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 4096

        
        print(f"映射头配置: 词表大小={self.vocab_size}, 隐藏层大小={self.hidden_size}")
        
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # 多模态特有的映射层，处理多模态信息融合
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 线性变换与归一化层
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # 第二层（可选）
        if num_layers >= 2:
            self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(self.hidden_size)
        
        # 第三层（可选）- 增加模型容量
        if num_layers >= 3:
            self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(self.hidden_size)

        # 超曲分类器：num_features 为 hidden_size+1（多出的1表示 time 分量）
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # 欧氏分类器
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        print(f"初始化{'超曲' if use_hyperbolic else '欧氏'}映射头，层数: {num_layers}")

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # 通过多模态适配器
        x = self.multimodal_adapter(last_hidden_states)
        
        # 第一层处理
        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        
        # 第二层处理（如果有）
        if self.num_layers >= 2:
            x = self.linear2(x)
            x = self.norm2(x)
            x = torch.relu(x)
        
        # 第三层处理（如果有）
        if self.num_layers >= 3:
            x = self.linear3(x)
            x = self.norm3(x)
            x = torch.relu(x)
        
        # 根据几何类型选择分类器
        if self.use_hyperbolic:
            # 添加 time 分量，再进行超曲映射
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# 图像转换定义 - 适合MLLama模型的标准化
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ====================== 加载 COCO 数据集 ======================
print(f"{'='*40}\n加载COCO数据集\n{'='*40}")

try:
    from pycocotools.coco import COCO
    
    # 设置COCO数据集路径
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
    
    # 为了测试集，从验证集中分割
    random.shuffle(val_ids)
    val_split = int(len(val_ids) * 0.5)
    new_val_ids = val_ids[:val_split]
    test_ids = val_ids[val_split:]
    
    if MAX_SAMPLES and MAX_SAMPLES > 0:
        train_ids = train_ids[:MAX_SAMPLES]
        new_val_ids = new_val_ids[:MAX_SAMPLES//5]
        test_ids = test_ids[:MAX_SAMPLES//5]
    
    print(f"数据集大小 - 训练集：{len(train_ids)}张图像，验证集：{len(new_val_ids)}张图像，测试集：{len(test_ids)}张图像")
    
    # 创建COCO数据集类
    class COCODataset(Dataset):
        def __init__(self, coco, img_ids, img_dir, max_length=128):
            self.coco = coco
            self.img_ids = img_ids
            self.img_dir = img_dir
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
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
            
            # 获取图像描述
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 随机选择一条描述
            caption = random.choice([ann['caption'] for ann in anns]) if anns else "无描述"
            
            # 正方形处理图像 (保持宽高比)
            w, h = image.size
            if w > h:
                new_w = IMAGE_SIZE
                new_h = int(h * IMAGE_SIZE / w)
            else:
                new_h = IMAGE_SIZE
                new_w = int(w * IMAGE_SIZE / h)
                    
            image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # 创建正方形图像
            square_img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
            paste_x = (IMAGE_SIZE - new_w) // 2
            paste_y = (IMAGE_SIZE - new_h) // 2
            square_img.paste(image, (paste_x, paste_y))
            
            return {
                "image": square_img,
                "caption": caption,
                "image_id": img_id
            }

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
            
        return {
            "image": images,
            "caption": captions,
            "image_id": image_ids
        }
    
    # 创建数据集
    train_dataset = COCODataset(train_coco, train_ids, train_image_dir, max_length=MAX_LENGTH)
    val_dataset = COCODataset(val_coco, new_val_ids, val_image_dir, max_length=MAX_LENGTH)
    test_dataset = COCODataset(val_coco, test_ids, val_image_dir, max_length=MAX_LENGTH)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    print(f"数据加载器创建成功！批次大小: {BATCH_SIZE}")
    
except Exception as e:
    print(f"COCO数据集加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==================== 辅助函数 ====================
def prepare_multimodal_inputs_batch(batch, processor):
    """处理整个批次的图像和文字"""
    batch_size = len(batch["image"])
    all_inputs = []
    
    # 确保临时目录存在
    os.makedirs("temp_images", exist_ok=True)
    
    for i in range(batch_size):
        image = batch["image"][i]
        caption = batch["caption"][i]
        
        # 保存图像到临时文件 - 使用索引确保文件名唯一
        temp_img_path = f"temp_images/temp_{time.time()}_{i}.jpg"
        image.save(temp_img_path)
        
        # 构建会话
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>\n描述这张图片",
                "images": [temp_img_path],
            },
            {"role": "<|Assistant|>", "content": caption},
        ]
        
        # 处理图像和会话
        pil_images = load_pil_images(conversation)
        input_data = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        )
        
        # 将处理后的输入添加到列表
        all_inputs.append(input_data)
        
        # 清理临时文件
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    
    # 批量合并处理后的数据
    if len(all_inputs) == 0:
        raise ValueError("批次中没有有效的输入")
    
    # 合并所有输入成一个批次
    return batch_combine_inputs(all_inputs)

def batch_combine_inputs(input_list):
    """合并处理后的输入列表为一个批次，处理不同长度序列"""
    if len(input_list) == 1:
        return input_list[0]
    
    # 获取第一个输入的键作为参考
    keys = input_list[0].keys()
    combined = {}
    
    for key in keys:
        if isinstance(input_list[0][key], torch.Tensor):
            # 处理可能有不同长度的输入
            if input_list[0][key].dim() > 1:
                # 找出需要处理的张量
                tensors = [inp[key] for inp in input_list]
                shapes = [t.shape for t in tensors]
                
                # 检查是否所有形状在第一维之外相同
                if not all(s[1:] == shapes[0][1:] for s in shapes):
                    # 需要填充 - 找出每个维度的最大值
                    max_dims = []
                    for dim in range(1, len(shapes[0])):
                        max_dims.append(max(s[dim] for s in shapes))
                    
                    # 创建填充后的张量列表
                    padded_tensors = []
                    for tensor in tensors:
                        # 计算每个维度需要填充的量
                        pad_sizes = []
                        for dim in range(1, len(tensor.shape)):
                            pad_sizes.extend([0, max_dims[dim-1] - tensor.shape[dim]])
                        
                        # 如果需要填充
                        if any(p > 0 for p in pad_sizes):
                            # 从右到左填充（pytorch要求的填充顺序）
                            padded = torch.nn.functional.pad(tensor, pad_sizes, 'constant', 0)
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    
                    combined[key] = torch.cat(padded_tensors, dim=0)
                else:
                    # 形状相同，直接合并
                    combined[key] = torch.cat(tensors, dim=0)
            else:
                # 对于1D张量，直接连接
                combined[key] = torch.cat([inp[key] for inp in input_list], dim=0)
        elif key in ['pixel_values', 'images_emb_mask', 'images_seq_mask']:
            # 特殊处理图像相关张量，也需要检查形状
            try:
                combined[key] = torch.cat([inp[key] for inp in input_list], dim=0)
            except RuntimeError as e:
                print(f"图像张量形状不一致: {e}，尝试填充...")
                # 获取张量并检查其形状
                tensors = [inp[key] for inp in input_list]
                shapes = [t.shape for t in tensors]
                print(f"不同样本的形状: {shapes}")
                
        else:
            # 非张量字段，保持列表形式
            combined[key] = [inp[key] for inp in input_list]
    
    return combined

def prepare_labels(batch):
    """准备训练标签"""
    input_ids = batch["input_ids"].clone()
    

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100 # 忽略最后一个位置
    
    # 将填充部分设置为-100
    padding_mask = (input_ids == tokenizer.pad_token_id)
    labels[padding_mask] = -100
    
    return labels

def compute_lm_loss(logits, labels):
    """计算语言模型损失"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

def get_gpu_memory_stats():
    """获取GPU内存使用情况"""
    if not torch.cuda.is_available():
        return "GPU不可用"
    
    stats = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        stats.append(f"GPU{i}: {allocated:.1f}GB/{reserved:.1f}GB")
    
    return " | ".join(stats)

# =========== 初始化模型组件 ===========
custom_lm_head = MultimodalMappingHead(model, use_hyperbolic=USE_HYPERBOLIC, num_layers=3).to(device)
learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))

# ========= 构建优化器和学习率调度器 ===========
optimizer = torch.optim.AdamW([
    {"params": list(custom_lm_head.parameters()), "lr": BASE_LR},
    {"params": [learnable_curvature], "lr": CURVATURE_LR}
] + ([{"params": [p for p in model.parameters() if p.requires_grad], "lr": BASE_LR}] if USE_LORA else []))

# 计算训练步数
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_loss = float("inf")

# 增强训练指标日志函数
def log_metrics(epoch, avg_loss, ppl, elapsed_time):
    """记录每个epoch详细训练指标"""
    current_lr = optimizer.param_groups[0]['lr']
    gpu_stats = get_gpu_memory_stats() if torch.cuda.is_available() else "N/A"
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch} 训练总结 (用时: {elapsed_time:.2f}秒)")
    print(f"{'='*50}")
    print(f"训练损失: {avg_loss:.4f}")
    print(f"困惑度(PPL): {ppl:.2f}")
    print(f"学习率: {current_lr:.2e}")
    print(f"超曲曲率: {learnable_curvature.item():.4f}")
    print(f"GPU内存使用: {gpu_stats}")
    print(f"每批次平均用时: {elapsed_time/len(train_loader):.3f}秒")
    print(f"预计下一轮用时: {elapsed_time/60:.1f}分钟")
    print(f"{'='*50}")

def save_model_for_hf(model, custom_lm_head, curvature, output_dir, processor=None):
    """将模型保存为HuggingFace兼容格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存基础模型
    if isinstance(model, nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
    
    # 保存基础模型和配置
    base_model.save_pretrained(output_dir)
    
    # 如果有处理器，也保存它
    if processor is not None:
        processor.save_pretrained(output_dir)
    
    # 2. 保存配置文件 - 添加双曲映射信息
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 添加双曲映射信息到配置
        config["hyperbolic_mapping"] = {
            "enabled": USE_HYPERBOLIC,
            "curvature": float(curvature.item()),
            "num_layers": custom_lm_head.num_layers,
            "model_type": "janus-hyperbolic" if USE_HYPERBOLIC else "janus-euclidean"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # 3. 保存自定义映射头和曲率参数
    custom_head_path = os.path.join(output_dir, "hyperbolic_mapping_head.bin")
    torch.save({
        "lm_head_state": custom_lm_head.state_dict(),
        "curvature": curvature.item(),
        "use_hyperbolic": USE_HYPERBOLIC,
        "hidden_size": custom_lm_head.hidden_size,
        "vocab_size": custom_lm_head.vocab_size,
        "num_layers": custom_lm_head.num_layers
    }, custom_head_path)
    
    print(f"HuggingFace格式模型已保存到: {output_dir}")


    return output_dir

def save_model(epoch, loss, ppl):
    """保存模型 - 每个epoch都保存，同时保存HuggingFace格式"""
    global best_loss
    model_type = "hyp" if USE_HYPERBOLIC else "euc"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./checkpoints/{model_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录最佳模型
    is_best = loss < best_loss
    if is_best:
        best_loss = loss
    
    # 始终保存当前epoch的模型 (原始格式)
    filename = f"{model_type}_vision_epoch_{epoch}_loss_{loss:.4f}_PPL_{ppl:.2f}{'_best' if is_best else ''}.pth"
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        "lm_head_state": custom_lm_head.state_dict(),
        "curvature": learnable_curvature.item(),
        "epoch": epoch,
        "loss": loss,
        "ppl": ppl,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }, save_path)
    
    if USE_LORA:
        lora_save_path = os.path.join(save_dir, f"lora_{model_type}_epoch_{epoch}")
        if isinstance(model, nn.DataParallel):
            model.module.save_pretrained(lora_save_path)
        else:
            model.save_pretrained(lora_save_path)
    
    # 对最佳模型或每隔几个epoch保存HuggingFace格式
    if is_best or (epoch % 5 == 0):  # 每5个epoch保存一次或最佳模型
        hf_dir = f"./hf_models/{model_type}_epoch_{epoch}"
        save_model_for_hf(
            model=model,
            custom_lm_head=custom_lm_head,
            curvature=learnable_curvature,
            output_dir=hf_dir,
            processor=processor
        )
        print(f"同时已保存HuggingFace格式: {hf_dir}")
    
    print(f"模型检查点已保存: {save_path}")
    return save_path


def load_checkpoint(checkpoint_path):
    """加载检查点恢复训练"""
    print(f"正在从检查点恢复训练: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 恢复模型状态
    custom_lm_head.load_state_dict(checkpoint["lm_head_state"])
    
    # 恢复曲率参数
    with torch.no_grad():
        learnable_curvature.copy_(torch.tensor(checkpoint["curvature"], device=device))
    
    # 恢复优化器状态
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # 确保优化器张量在正确设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # 恢复调度器状态
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    # 恢复LoRA参数(如果存在)
    if USE_LORA and os.path.dirname(checkpoint_path):
        lora_dir = os.path.dirname(checkpoint_path)
        # 查找最相关的LoRA目录
        lora_dirs = [d for d in os.listdir(lora_dir) if d.startswith("lora_")]
        if lora_dirs:
            newest_lora = sorted(lora_dirs)[-1]
            lora_path = os.path.join(lora_dir, newest_lora)
            print(f"加载LoRA权重: {lora_path}")
            if isinstance(model, nn.DataParallel):
                model.module.load_adapter(lora_path)
            else:
                model.load_adapter(lora_path)
    
    # 返回恢复的训练信息
    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))


# ====================== 训练模型 ======================
gradient_accumulation_steps = 4
def train_model():
    global best_loss
    latest_checkpoint_path = None
    log_interval = 1000  # 每100个批次显示一次损失信息
    
    start_epoch = args.start_epoch
    if args.resume:
        loaded_epoch, loaded_loss = load_checkpoint(args.resume)
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1
            best_loss = min(best_loss, loaded_loss)
            print(f"成功恢复训练至第{loaded_epoch}轮后，将从第{start_epoch}轮继续")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0
        recent_losses = []  # 存储最近的损失值
        
        # 在循环外清零梯度
        optimizer.zero_grad()
        
        # 每个epoch前清理内存
        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(train_loader, desc=f"训练第{epoch}轮", 
                                      bar_format='{l_bar}{bar:30}{r_bar}')):

        # 处理多模态输入
            inputs = prepare_multimodal_inputs_batch(batch, processor)
            
            # 直接将输入移至设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            labels = prepare_labels(inputs)
            
# 在train_model和evaluate_model中，修改模型调用部分
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs_embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
                
                # 调用内部模型
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # 获取最后的隐藏状态
                hidden_states = outputs.last_hidden_state
                
                # 使用自定义映射头
                logits = custom_lm_head(hidden_states, learnable_curvature)
                raw_loss = compute_lm_loss(logits, labels)
                loss = raw_loss / gradient_accumulation_steps
            # 反向传播
            loss.backward()
            
            # 记录每批次损失
            batch_size = len(batch["image"])
            item_loss = raw_loss.item()
            recent_losses.append(item_loss)
            total_loss += item_loss * batch_size
            count += batch_size
            
            # 每log_interval批次显示一次平均损失
            if (i + 1) % log_interval == 0:
                avg_recent_loss = sum(recent_losses[-log_interval:]) / min(log_interval, len(recent_losses))
                lr = optimizer.param_groups[0]['lr']
                print(f"批次 {i+1}/{len(train_loader)}, 损失: {avg_recent_loss:.4f}, 学习率: {lr:.1e}, 曲率: {learnable_curvature.item():.4f}")
            
            # 累积指定步数后更新参数
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                # 梯度裁剪
                clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + 
                    list(custom_lm_head.parameters()) + 
                    [learnable_curvature], 
                    max_norm=1.0
                )
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 限制曲率范围
                with torch.no_grad():
                    learnable_curvature.clamp_(1e-3, 1e1)
            
            # 清理CUDA缓存以减少内存碎片
            if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
                
            # 定期评估
            if (i + 1) % args.eval_steps == 0:
                print("\n进行中间评估...")
                interim_loss = evaluate_model(val_loader, phase="中间验证", max_batches=50)
                print(f"中间验证损失: {interim_loss:.4f}")
                # 恢复训练模式
                model.train()
                custom_lm_head.train()

        # 计算并记录每个epoch的详细指标
        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        
        # 详细的epoch总结日志
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        
        # 验证
        print("\n开始完整验证...")
        val_loss = evaluate_model(val_loader, phase="验证")
        latest_checkpoint_path = save_model(epoch, val_loss, math.exp(val_loss) if val_loss < 20 else float("inf"))
        # 保存模型        
        print(f"已保存第{epoch}轮检查点: {latest_checkpoint_path}")

# ====================== 评估函数 ======================
def evaluate_model(data_loader, phase="验证", max_batches=None):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"评估{phase}", 
                                     bar_format='{l_bar}{bar:30}{r_bar}')):
            if max_batches and i >= max_batches:
                break
                
            # 处理多模态输入
            inputs = prepare_multimodal_inputs_batch(batch, processor)
            
            # 将输入移至设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}            
            # 准备标签
            labels = prepare_labels(inputs)
            
            # 前向传播 - 使用autocast确保类型一致性
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs_embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
                
                # 调用内部模型
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                
                # 获取最后的隐藏状态
                hidden_states = outputs.last_hidden_state
                
                # 使用自定义映射头
                logits = custom_lm_head(hidden_states, learnable_curvature)
                loss = compute_lm_loss(logits, labels) 
            
            batch_size = len(batch["image"])
            total_loss += loss.item() * batch_size
            count += batch_size
        
        # 评估完成后的简洁总结
        eval_time = time.time() - start_time
        avg_loss = total_loss / count if count > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        print(f"{phase}结果: 损失={avg_loss:.4f}, PPL={ppl:.2f}, 用时={eval_time:.2f}秒")
        return avg_loss


# ====================== 主执行流程 ======================
# ====================== 主执行流程 ======================
if __name__ == "__main__":
    print(f"{'='*50}")
    print(f"配置: Janus-Pro模型, 超曲={USE_HYPERBOLIC}, LoRA={USE_LORA}, 批量={BATCH_SIZE}")
    print(f"学习率={BASE_LR}, 曲率学习率={CURVATURE_LR}, 设备={DEVICE}")
    print(f"{'='*50}")
    
    # 使用args.mode，移除重复的参数解析
    if args.mode == "train":
        # 训练模式
        print("开始训练模型...")
        train_model()
    
    elif args.mode == "eval":
        # 评估模式
        print("开始评估模型...")
        evaluate_model(test_loader, phase="测试")
