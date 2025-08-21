import sys
import os
import time
import math
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, LlamaForCausalLM, AutoModelForVision2Seq, MllamaForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz


def get_parser():
    parser = argparse.ArgumentParser(description="Train a multimodal vision-language model with Hyperbolic mapping")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping (default: Euclidean)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum number of tokens per sample")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for the dataset to load")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--image_size", type=int, default=336, help="Size of input images")
    return parser

args = get_parser().parse_args()

USE_HYPERBOLIC = args.use_hyperbolic
USE_LORA = args.use_lora
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
BASE_LR = args.learning_rate
MAX_LENGTH = args.max_length
DEVICE = args.device
MAX_SAMPLES = args.max_samples
IMAGE_SIZE = args.image_size

# ===================== Load Llama Vision Model =====================
hf_token = "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"
model_name = "meta-llama/Llama-3.2-11B-Vision"  # 11B Vision model instead of 1B text-only

device = torch.device(DEVICE)
processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

# 使用正确的模型类加载Llama-3.2-Vision
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

# ===== Apply LoRA Adaptation or Freeze Original Model =====
if USE_LORA:
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
else:
    # Freeze original model parameters
    for param in model.parameters():
        param.requires_grad = False

# ====================== Load Stable Diffusion for Image Generation ======================
# 由于Llama Vision不能生成图像，我们需要单独的模型用于图像生成
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    scheduler=EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
    )
).to(device)

# ============== Custom Multimodal Mapping Head =================
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=2):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.vocab_size = base_model.config.vocab_size
        self.hidden_size = base_model.config.hidden_size
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # 多模态特有的映射层，用于处理多模态信息
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 第1层线性变换及归一化
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        # 第二层（可选）
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # 超曲分类器：num_features 为 hidden_size+1（多出的1表示 time 分量）
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        # 欧氏分类器
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # 首先通过多模态适配器
        x = self.multimodal_adapter(last_hidden_states)
        
        # 标准流程处理
        x = self.linear1(x)
        x = self.norm1(x)
        
        if self.num_layers == 2:
            x = torch.relu(x)
            x = self.linear2(x)
            x = self.norm2(x)
        
        if self.use_hyperbolic:
            # 添加 time 分量，再进行超曲映射
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

# ====================== Load COCO Dataset ======================
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

# ==================== 数据批处理 ====================
def get_batches(pairs, batch_size):
    """将数据划分为批次"""
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i+batch_size]

def compute_lm_loss(logits, labels):
    """计算语言模型损失"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)
    labels_2d = labels.view(-1)
    loss = loss_fct(logits_2d, labels_2d)
    return loss

# ====================== 训练模型 ======================
def train_model():
    global best_loss
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        custom_lm_head.train()
        epoch_start_time = time.time()
        total_loss, count = 0.0, 0

        # 训练循环
        for batch_pairs in tqdm(get_batches(train_pairs, BATCH_SIZE), total=len(train_pairs)//BATCH_SIZE, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            # 处理多模态输入
            inputs = prepare_multimodal_inputs(batch_pairs, processor, MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            
            # 前向传播
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                # 使用自定义映射头
                logits = custom_lm_head(outputs.hidden_states[-1], learnable_curvature)
                loss = compute_lm_loss(logits, labels)
            
            # 反向传播
            loss.backward()
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + 
                list(custom_lm_head.parameters()) + 
                [learnable_curvature], 
                max_norm=1.0
            )
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 确保曲率参数在有意义的范围内
            with torch.no_grad():
                learnable_curvature.clamp_(1e-3, 1e1)
                
            total_loss += loss.item() * len(batch_pairs)
            count += len(batch_pairs)

        # 计算并记录指标
        avg_loss = total_loss / count if count > 0 else 9999.0
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start_time
        
        log_metrics(epoch, avg_loss, ppl, epoch_time)
        val_loss = evaluate_model(val_pairs, phase="Validation")
        save_model(epoch, val_loss, math.exp(val_loss) if val_loss < 20 else float("inf"))

# ====================== 评估函数 ======================
def evaluate_model(pairs, phase="Val"):
    model.eval()
    custom_lm_head.eval()
    total_loss, count = 0.0, 0
    
    with torch.no_grad():
        for batch_pairs in tqdm(get_batches(pairs, BATCH_SIZE), total=len(pairs)//BATCH_SIZE, desc=f"Evaluating {phase}"):
            # 处理输入
            inputs = prepare_multimodal_inputs(batch_pairs, processor, MAX_LENGTH).to(device)
            labels = prepare_labels(inputs).to(device)
            
            # 前向传播
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            logits = custom_lm_head(hidden_states, learnable_curvature)
            
            # 计算损失
            loss_val = compute_lm_loss(logits, labels)
            total_loss += loss_val.item() * len(batch_pairs)
            count += len(batch_pairs)
            
    avg_loss = total_loss / count if count > 0 else 9999.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print(f"{phase} - loss={avg_loss:.4f}, PPL={ppl:.2f}")
    return avg_loss

# ====================== 基于图像生成文本 ======================
def generate_text_from_image(image_path, max_len=100):
    """输入图像，输出生成的文本"""
    model.eval()
    custom_lm_head.eval()
    
    # 加载图像
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # 处理图像输入
    inputs = processor(images=image, text="Describe this image:", return_tensors="pt").to(device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 解码输出
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return generated_text

# ====================== 基于文本生成图像 ======================
def generate_image_from_text(text_prompt, output_path=None):
    """输入文本提示，生成图像"""
    # 使用Stable Diffusion生成图像
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image = image_gen_model(text_prompt, guidance_scale=7.5).images[0]
    
    # 保存图像
    if output_path:
        image.save(output_path)
    
    return image

# ====================== 主执行流程 ======================
if __name__ == "__main__":
    print(f"USE_HYPERBOLIC = {USE_HYPERBOLIC}")
    print(f"USE_LORA = {USE_LORA}")
    
    # 训练模型
    train_model()
    
    # 测试集评估
    print("\n=== Testing ===")
    evaluate_model(test_pairs, phase="Test")
    
    # 演示多模态能力：输入图像，输出文本描述
    print("\n=== Image-to-Text Generation ===")
    sample_image = test_pairs[0]["image"]
    sample_image_path = "sample_image.jpg"
    sample_image.save(sample_image_path)
    generated_description = generate_text_from_image(sample_image_path)
    print(f"Generated description: {generated_description}")
    
    # 演示文本到图像生成
    print("\n=== Text-to-Image Generation ===")
    sample_text_prompt = "A beautiful landscape with mountains and a lake at sunset"
    generated_image = generate_image_from_text(sample_text_prompt, "generated_image.jpg")
    print(f"Image generated and saved as generated_image.jpg")