import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, PreTrainedModel
from janus.models import VLChatProcessor, MultiModalityCausalLM # 确保导入 MultiModalityCausalLM
from janus.models.vq_model import VQModel, ModelArgs # 从 Janus 导入 VQModel
from janus.models.image_processing_vlm import VLMImageProcessor, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD # 从 Janus 导入
from PIL import Image
from tqdm import tqdm
import math
import random
import numpy as np
from peft import LoraConfig, get_peft_model
import json
import traceback
from pycocotools.coco import COCO # 用于加载 COCO 数据集

# 导入 HyperLib 组件
from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz

# --- 1. 参数解析 ---
def get_parser():
    parser = argparse.ArgumentParser(description="训练文本到图像视觉 Token 的超曲映射头")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Janus model")
    parser.add_argument("--coco_data_dir", type=str, default="./coco_dataset", help="Path to COCO dataset directory (containing annotations/ and train2017/)")
    parser.add_argument("--output_dir", type=str, default="./t2i_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_hyperbolic", action="store_true", help="Use hyperbolic mapping head")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation for the language model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the head/LoRA")
    parser.add_argument("--curvature_lr", type=float, default=1e-6, help="Learning rate for curvature")
    parser.add_argument("--max_text_length", type=int, default=77, help="Max length for text prompts")
    parser.add_argument("--image_size", type=int, default=384, help="Target image size for VQModel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Use pin memory for DataLoader")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Computation data type")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--head_num_layers", type=int, default=3, help="Number of layers in the mapping head")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点文件恢复训练")
    # 添加 LoRA 目标模块参数
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Target modules for LoRA")

    return parser

# --- 2. 设置随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 3. 自定义生成头 (HyperbolicGenHead) ---
class HyperbolicGenHead(nn.Module):
    def __init__(self, hidden_size, codebook_size, use_hyperbolic=True, num_layers=3):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size # 视觉码本大小 (e.g., 16384)
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # 与图生文头类似，但最终输出维度是 codebook_size
        # 注意：这里没有 multimodal_adapter，直接从 hidden_states 开始
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        if num_layers >= 2:
            self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(hidden_size)
        if num_layers >= 3:
            self.linear3 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(hidden_size)
        # 可以根据需要添加更多层

        # 分类器
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=hidden_size + 1, # Lorentz 需要 time 分量
            num_classes=self.codebook_size
        )
        self.euc_cls = nn.Linear(hidden_size, self.codebook_size, bias=False)

        print(f"初始化 {'超曲' if use_hyperbolic else '欧氏'} 生成头 (GenHead)，输出维度: {self.codebook_size}, 层数: {self.num_layers}")

    def lorentz_map(self, x, c_param):
        k_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
        if k_tensor.dim() == 0:
             k_tensor = k_tensor.unsqueeze(0)
        return expmap0(x, k=k_tensor, dim=-1)

    def forward(self, hidden_states, c_param=None):
        target_dtype = self.linear1.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        x = hidden_states # 直接使用 hidden_states

        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        if self.num_layers >= 2:
            x = self.linear2(x)
            x = self.norm2(x)
            x = torch.relu(x)
        if self.num_layers >= 3:
            x = self.linear3(x)
            x = self.norm3(x)
            x = torch.relu(x)
        # 可以根据需要添加更多层

        if self.use_hyperbolic:
            if c_param is None:
                raise ValueError("Curvature parameter 'c_param' is required for hyperbolic head.")
            x = self.manifold.add_time(x)
            c_param_tensor = torch.as_tensor(c_param, dtype=x.dtype, device=x.device)
            hyper_embs = self.lorentz_map(x, c_param_tensor)
            if hasattr(self.hyp_cls, 'z'):
                if hyper_embs.dtype != self.hyp_cls.z.dtype:
                     hyper_embs = hyper_embs.to(self.hyp_cls.z.dtype)
                 # 如果 'z' 不存在，可以尝试 'a' 或抛出错误
                 # print("Warning: LorentzMLR has no 'z' attribute, checking 'a'")
                 # if hasattr(self.hyp_cls, 'a') and hyper_embs.dtype != self.hyp_cls.a.dtype:
                 #      hyper_embs = hyper_embs.to(self.hyp_cls.a.dtype)
            else:
                 raise AttributeError("LorentzMLR object has no 'z' or 'a' attribute to check dtype against.")
            # --- 修改结束 ---
                 
            logits = self.hyp_cls(hyper_embs)
        else:
            if x.dtype != self.euc_cls.weight.dtype:
                 x = x.to(self.euc_cls.weight.dtype)
            logits = self.euc_cls(x)

        return logits # 输出视觉 Token 的 logits


def evaluate(model, val_loader, loss_fn, device, compute_dtype, args, learnable_curvature, codebook_size):
    model.eval() # 设置评估模式
    total_val_loss = 0.0
    with torch.no_grad(): # 禁用梯度计算
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            if batch is None: continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_visual_tokens = batch["target_visual_tokens"].to(device)

            # --- 与训练循环类似的前向传播 ---
            with torch.autocast(device_type=args.device, dtype=compute_dtype, enabled=(compute_dtype != torch.float32)):
                text_embeds = model.language_model.get_input_embeddings()(input_ids)
                visual_bos_id = 0
                visual_bos_ids = torch.full((target_visual_tokens.size(0), 1), visual_bos_id, dtype=torch.long, device=device)
                input_visual_ids = torch.cat([visual_bos_ids, target_visual_tokens[:, :-1]], dim=1)
                visual_embeds = model.prepare_gen_img_embeds(input_visual_ids)
                combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)
                visual_attention_mask = torch.ones_like(input_visual_ids)
                combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

                outputs = model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                visual_hidden_states = hidden_states[:, input_ids.shape[1]:, :]
                current_curvature = learnable_curvature.to(visual_hidden_states.dtype) if learnable_curvature is not None else None
                logits = model.gen_head(visual_hidden_states, current_curvature) # 使用 model.gen_head

                loss = loss_fn(logits.view(-1, codebook_size), target_visual_tokens.view(-1))
                total_val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# --- 4. COCO 数据集定义 ---
class COCODataset(Dataset):
    def __init__(self, data_dir, processor, vq_model, image_processor, max_text_length, split='train'):
        self.data_dir = data_dir
        self.processor = processor
        self.vq_model = vq_model # 传入 VQModel 实例 (应处于 eval 模式并冻结)
        self.image_processor = image_processor # 传入 VLMImageProcessor 实例
        self.max_text_length = max_text_length
        self.split = split

        # 设置 COCO 路径
        self.img_dir = os.path.join(data_dir, f'{split}2017')
        self.ann_file = os.path.join(data_dir, 'annotations', f'captions_{split}2017.json')

        if not os.path.exists(self.img_dir) or not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"COCO 数据集文件未在 {data_dir} 中找到。请确保 annotations/ 和 {split}2017/ 存在。")

        print(f"加载 COCO {split} 集注释: {self.ann_file}")
        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(f"找到 {len(self.ids)} 张图像。")

        # 确保 VQModel 在评估模式
        self.vq_model.eval()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 随机选择一个文本描述作为输入
        caption = random.choice(anns)['caption']

        # 1. 处理文本
        # 使用 DeepSeek 的模板格式化文本输入
        # 注意：这里假设我们只用一个 User turn 作为输入
        conv = self.processor.new_chat_template() # 获取对话模板
        conv.append_message(conv.roles[0], caption) # 添加用户消息
        conv.append_message(conv.roles[1], None) # 添加空的助手消息以触发生成格式
        text_prompt = conv.get_prompt() # 获取格式化后的文本

        text_inputs = self.processor.tokenizer(
            text_prompt,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.squeeze(0) # (seq_len)
        attention_mask = text_inputs.attention_mask.squeeze(0) # (seq_len)

        # 2. 处理图像并获取目标视觉 Tokens
        try:
            image = Image.open(img_path).convert('RGB')
            # 使用 VLMImageProcessor 处理图像
            image_inputs = self.image_processor.preprocess([image], return_tensors="pt")
            pixel_values = image_inputs['pixel_values'] # (1, 3, H, W)

            # 使用 VQModel 获取目标视觉 Tokens (在 no_grad 上下文中)
            with torch.no_grad():
                # 将 pixel_values 移动到 vq_model 所在的设备和类型
                vq_device = next(self.vq_model.parameters()).device
                vq_dtype = next(self.vq_model.parameters()).dtype
                pixel_values = pixel_values.to(vq_device, vq_dtype)

                # 调用 encode 获取视觉 tokens
                _, _, info = self.vq_model.encode(pixel_values)
                # info = (perplexity, min_encodings, min_encoding_indices)
                target_visual_tokens = info[2].squeeze(0) # (latent_h * latent_w) or (num_tokens)

        except Exception as e:
            print(f"处理图像 {img_path} 或获取视觉 Tokens 时出错: {e}")
            traceback.print_exc()
            # 返回 None 或引发错误，让 collate_fn 处理
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_visual_tokens": target_visual_tokens, # (num_visual_tokens)
        }

# --- 5. Collate 函数 ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None] # 过滤掉加载失败的样本
    if not batch:
        return None

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # 处理 target_visual_tokens (它们长度应该是一致的)
    target_visual_tokens = torch.stack([item["target_visual_tokens"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_visual_tokens": target_visual_tokens,
    }

# ... (其他导入) ...
import os
import json
import torch
import torch.nn as nn
from peft import PeftModel # 确保导入 PeftModel

# ... (其他代码, 包括 HyperbolicGenHead, COCODataset, main 等) ...

def save_model_for_hf(model, hyperbolic_gen_head, learnable_curvature, output_dir, processor, args, hidden_size, codebook_size):
    """
    将模型（包括基础模型配置、LoRA适配器和自定义生成头）保存为HuggingFace兼容格式。

    Args:
        model: 训练后的模型 (可能是 PeftModel 实例)。
        hyperbolic_gen_head: 训练后的自定义生成头 (HyperbolicGenHead)。
        learnable_curvature: 训练后的曲率参数 (torch.Tensor)。
        output_dir: 保存模型的目录。
        processor: 使用的 VLChatProcessor 实例。
        args: 包含配置的 argparse 参数对象 (例如 args.use_hyperbolic, args.head_num_layers, args.use_lora)。
        hidden_size: 模型的隐藏层大小。
        codebook_size: 视觉码本的大小 (VQ模型的词汇表大小)。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始保存 HuggingFace 格式模型到: {output_dir}")

    # 1. 保存 Processor
    if processor is not None:
        try:
            processor.save_pretrained(output_dir)
            print("  - Processor 已保存。")
        except Exception as e:
            print(f"  - 错误: 保存 Processor 失败: {e}")
    else:
        print("  - 警告: 未提供 Processor，无法保存。")

    # 2. 保存 LoRA 适配器 (如果使用了 LoRA)
    if args.use_lora:
        if isinstance(model, PeftModel):
            try:
                model.save_pretrained(output_dir) # 保存 adapter_config.json 和 adapter_model.bin
                print("  - LoRA 适配器已保存。")
            except Exception as e:
                print(f"  - 错误: 保存 LoRA 适配器失败: {e}")
        else:
            print("  - 警告: 使用了 LoRA 但模型不是 PeftModel 实例，无法单独保存适配器。")

    # 3. 保存基础模型配置 (总是需要)
    # 获取基础模型实例 (处理 PeftModel 和 DataParallel)
    if isinstance(model, PeftModel):
        base_model_instance = model.base_model.model # 获取底层的 MultiModalityCausalLM
    elif isinstance(model, nn.DataParallel):
         base_model_instance = model.module
    else:
        base_model_instance = model

    try:
        # 保存基础模型的 config.json
        base_model_instance.config.save_pretrained(output_dir)
        print(f"  - 基础模型配置已保存。")
    except Exception as e:
        print(f"  - 错误: 保存基础模型配置失败: {e}")

    # 4. 如果没有使用 LoRA，则保存完整的基础模型权重
    if not args.use_lora:
        try:
            base_model_instance.save_pretrained(output_dir)
            print(f"  - 完整基础模型权重已保存 (未使用 LoRA)。")
        except Exception as e:
            print(f"  - 错误: 保存完整基础模型权重失败: {e}")


    # 5. 修改配置文件以包含自定义生成头信息
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # 添加自定义生成头的信息
            config_data["custom_gen_head_config"] = {
                "is_hyperbolic": args.use_hyperbolic,
                "curvature": float(learnable_curvature.item()) if args.use_hyperbolic else None,
                "num_layers": args.head_num_layers,
                "hidden_size": hidden_size,
                "codebook_size": codebook_size,
                "head_type": "HyperbolicGenHead", # 记录头的类型
                "weights_file": "mapping_gen_head.bin" # 指向权重文件
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print("  - 主配置文件已更新，包含自定义生成头信息。")

        except Exception as e:
            print(f"  - 错误: 修改主配置文件失败: {e}")
    else:
        print(f"  - 警告: 未找到主配置文件 {config_path}，无法添加自定义头信息。")

    # 6. 保存自定义生成头的权重
    head_state_dict_path = os.path.join(output_dir, "mapping_gen_head.bin") # 使用不同的文件名
    try:
        torch.save(hyperbolic_gen_head.state_dict(), head_state_dict_path)
        print(f"  - 自定义生成头权重已保存到: {os.path.basename(head_state_dict_path)}")
    except Exception as e:
        print(f"  - 错误: 保存自定义生成头权重失败: {e}")

    print(f"HuggingFace 格式模型保存完成: {output_dir}")
    return output_dir


# --- 6. 主训练函数 ---
def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 设置数据类型 ---
    if args.dtype == "float16":
        compute_dtype = torch.float16
    elif args.dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # --- 加载模型和 Processor ---
    print("加载模型和 Processor...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    # 加载 Janus 模型，确保 trust_remote_code=True
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype, # 使用选择的数据类型
        trust_remote_code=True,
        attn_implementation="eager" # 保持一致
    ).to(device)
    print("检查 gen_embed 和 codebook 大小...")
    try:
        # 尝试从 config 获取 image_token_size (更可靠)
        gen_embed_size = model.config.gen_head_config.params.get('image_token_size', model.gen_embed.num_embeddings) # 从 gen_head 配置获取，或回退到 embedding 层
        codebook_size = None
        if hasattr(model, 'gen_vision_model') and hasattr(model.gen_vision_model, 'config') and hasattr(model.gen_vision_model.config, 'codebook_size'):
            codebook_size = model.gen_vision_model.config.codebook_size
            print("  - 从 model.gen_vision_model.config 获取 codebook_size")
        elif hasattr(model, 'config') and hasattr(model.config, 'gen_vision_config'):
            # 尝试原始路径，以防万一
            if hasattr(model.config.gen_vision_config, 'params') and hasattr(model.config.gen_vision_config.params, 'codebook_size'):
                 codebook_size = model.config.gen_vision_config.params.codebook_size
                 print("  - 从 model.config.gen_vision_config.params 获取 codebook_size")
            elif hasattr(model.config.gen_vision_config, 'codebook_size'): # 检查是否直接在 gen_vision_config 下
                 codebook_size = model.config.gen_vision_config.codebook_size
                 print("  - 从 model.config.gen_vision_config 获取 codebook_size")
        if codebook_size is None:
             # 如果仍然找不到，添加调试信息
             print("  - 警告: 无法自动确定 codebook_size。请检查模型配置。")
             print("    - model.config:", model.config)
             if hasattr(model, 'gen_vision_model'):
                 print("    - model.gen_vision_model.config:", getattr(model.gen_vision_model, 'config', 'N/A'))
             raise AttributeError("无法找到 codebook_size")
        print(f"  - model.gen_embed.num_embeddings (或配置中的 image_token_size): {gen_embed_size}")
        print(f"  - codebook_size: {codebook_size}")

    except AttributeError as e:
        print(f"  - 无法检查大小: {e}")
    # --- 验证必要的组件是否存在 ---
    if not hasattr(model, 'gen_vision_model') or not isinstance(model.gen_vision_model, VQModel):
        raise AttributeError("模型实例中未找到 'gen_vision_model' 或其类型不是预期的 VQModel。请检查模型加载和代码。")
    if not hasattr(model, 'gen_head'):
        raise AttributeError("模型实例中未找到 'gen_head'。")
    if not hasattr(model, 'prepare_gen_img_embeds'):
        raise AttributeError("模型实例中未找到 'prepare_gen_img_embeds' 方法。")
    # --- 验证 codebook_size 是否已成功获取 ---
    if codebook_size is None: # 检查上面逻辑是否成功获取
         raise AttributeError("在验证阶段未能获取 codebook_size。")

    # --- 准备 VLMImageProcessor ---
    # 使用 Janus 默认的或推理脚本中的值
    image_processor = VLMImageProcessor(
        image_size=args.image_size,
        image_mean=IMAGENET_INCEPTION_MEAN, # 假设使用 Inception 归一化
        image_std=IMAGENET_INCEPTION_STD,
    )

    # --- 准备自定义头并替换 ---
    print("准备并替换生成头...")
    # 获取 hidden_size 和 codebook_size
    hidden_size = model.language_model.config.hidden_size
    # --- End FIX 1 ---
    hyperbolic_gen_head = HyperbolicGenHead(
        hidden_size=hidden_size,
        codebook_size=codebook_size,
        use_hyperbolic=args.use_hyperbolic,
        num_layers=args.head_num_layers
    ).to(device).to(compute_dtype) # 保持类型一致

    # 替换原始 gen_head
    model.gen_head = hyperbolic_gen_head

    # --- 冻结参数 ---
    print("冻结模型原始参数...")
    for name, param in model.named_parameters():
        if "gen_head" not in name: # 不冻结新的 gen_head
            param.requires_grad = False

    # --- 应用 LoRA (可选) ---
    if args.use_lora:
        print(f"应用 LoRA 到模块: {args.lora_target_modules}")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules # 使用参数指定的目标模块
        )
        # 只对 language_model 应用 LoRA
        model.language_model = get_peft_model(model.language_model, config)
        model.language_model.print_trainable_parameters()

    # --- 准备曲率参数 ---
    learnable_curvature = None
    if args.use_hyperbolic:
        # 初始化为较小正值，使用 float32 以提高稳定性
        learnable_curvature = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
        learnable_curvature.requires_grad = True
        print(f"使用可学习曲率 (float32)，初始值: {learnable_curvature.item():.4f}")

    # --- 打印可训练参数 ---
    print("可训练参数:")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name} ({param.numel()}) - dtype: {param.dtype}")
            total_trainable_params += param.numel()
    if learnable_curvature is not None:
         print(f"- learnable_curvature ({learnable_curvature.numel()}) - dtype: {learnable_curvature.dtype}")
         total_trainable_params += learnable_curvature.numel()
    print(f"总可训练参数数量: {total_trainable_params / 1e6:.2f} M")


    # --- 准备数据集和 DataLoader ---
    print("加载 COCO 数据集...")
    # 确保 vq_model 在正确的设备上并冻结
    model.gen_vision_model.to(device).eval()
    train_dataset = COCODataset(
        args.coco_data_dir, processor, model.gen_vision_model, image_processor, args.max_text_length, split='train'
    )
    # 可以在这里添加验证集加载逻辑
    # val_dataset = COCODataset(args.coco_data_dir, ..., split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    # val_loader = DataLoader(val_dataset, ...)

    # --- 准备优化器和调度器 ---
    print("准备优化器和调度器...")
    # --- FIX 2: Optimize original parameters ---
    # params_to_optimize_head = [p.float() for p in model.gen_head.parameters() if p.requires_grad] # Original problematic line
    params_to_optimize_head = [p for p in model.gen_head.parameters() if p.requires_grad] # Pass original parameters
    # --- End FIX 2 ---
    params_to_optimize_lora = []
    if args.use_lora:
        # LoRA 参数通常已经是 float32
        params_to_optimize_lora = [p for p in model.language_model.parameters() if p.requires_grad]

    optimizer_grouped_parameters = []
    if params_to_optimize_head:
        # Optimizer will handle mixed precision based on the parameter's dtype
        optimizer_grouped_parameters.append({"params": params_to_optimize_head, "lr": args.learning_rate})
    if params_to_optimize_lora:
         optimizer_grouped_parameters.append({"params": params_to_optimize_lora, "lr": args.learning_rate}) # LoRA 使用相同学习率

    if learnable_curvature is not None:
        # 曲率使用单独的学习率 (keep as float32 if intended)
        optimizer_grouped_parameters.append({"params": [learnable_curvature], "lr": args.curvature_lr})

    if not optimizer_grouped_parameters:
        raise ValueError("没有找到可训练的参数！")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_training_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # --- 加载检查点 (如果指定) ---
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

            # 加载模型状态
            # 使用 strict=False 如果模型结构可能略有不同，或者只保存了部分权重
            # 如果保存了完整 state_dict，通常用 strict=True
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"警告: 加载模型状态时缺少键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 加载模型状态时存在意外键: {unexpected_keys}")

            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载调度器状态
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载 GradScaler 状态
            if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # 加载曲率
            if learnable_curvature is not None and 'learnable_curvature' in checkpoint and checkpoint['learnable_curvature'] is not None:
                 # 直接设置 Parameter 的 data
                 with torch.no_grad():
                      learnable_curvature.data.fill_(checkpoint['learnable_curvature'])
                 print(f"恢复曲率: {learnable_curvature.item():.4f}")

            # 恢复 epoch 和 step
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']

            print(f"成功从 epoch {start_epoch}, global_step {global_step} 恢复。")

            # 可选：验证参数是否匹配
            # loaded_args = checkpoint.get('args', {})
            # current_args = vars(args)
            # for key, value in loaded_args.items():
            #     if key != 'resume_from_checkpoint' and current_args.get(key) != value:
            #         print(f"警告: 检查点参数 '{key}' ({value}) 与当前参数 ({current_args.get(key)}) 不匹配。")

        except Exception as e:
            print(f"加载检查点失败: {e}. 从头开始训练。")
            start_epoch = 0
            global_step = 0
    else:
        print("未找到检查点或未指定，从头开始训练。")

    # --- 训练循环 ---
    print("开始训练...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype == torch.float16)) # 仅 float16 需要 scaler

    for epoch in range(args.num_epochs):
        model.train() # 确保模型在训练模式 (特别是 LoRA 和 Dropout)
        if args.use_lora:
            model.language_model.train()
        model.gen_head.train() # 确保自定义头在训练模式

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", initial=0)

        for step, batch in enumerate(progress_bar):
            if batch is None: continue # 跳过加载失败的批次

            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_visual_tokens = batch["target_visual_tokens"].to(device) # (B, num_visual_tokens)

            # --- 混合精度训练上下文 ---
            with torch.autocast(device_type=args.device, dtype=compute_dtype, enabled=(compute_dtype != torch.float32)):
                # 1. 获取文本嵌入
                text_embeds = model.language_model.get_input_embeddings()(input_ids) # (B, text_seq_len, D)

                # 2. 准备输入视觉 Token IDs (右移 + BOS)
                # TODO: 确认视觉序列的起始 Token ID visual_bos_id
                visual_bos_id = 0 # 假设为 0，需要验证！
                visual_bos_ids = torch.full((target_visual_tokens.size(0), 1), visual_bos_id,
                                            dtype=torch.long, device=device)
                # input_visual_ids: (B, visual_seq_len)
                input_visual_ids = torch.cat([visual_bos_ids, target_visual_tokens[:, :-1]], dim=1)

                # 3. 获取输入视觉 Token 的嵌入
                try:
                    # 使用模型提供的 prepare_gen_img_embeds 方法
                    # visual_embeds: (B, visual_seq_len, D)
                    visual_embeds = model.prepare_gen_img_embeds(input_visual_ids)
                    # 确保数据类型与 text_embeds 一致 (autocast 会处理)
                    # visual_embeds = visual_embeds.to(text_embeds.dtype)
                except Exception as e:
                    print(f"获取 visual_embeds 时出错: {e}")
                    traceback.print_exc()
                    continue # 跳过这个批次

                # 4. 拼接嵌入序列
                # combined_embeds: (B, text_len + visual_seq_len, D)
                combined_embeds = torch.cat([text_embeds, visual_embeds], dim=1)

                # 5. 构建注意力掩码
                visual_attention_mask = torch.ones_like(input_visual_ids) # 视觉部分全部可见
                # combined_attention_mask: (B, text_len + visual_seq_len)
                combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

                # --- 模型前向传播 ---
                outputs = model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True # 确保输出隐藏状态
                )
                # hidden_states: (B, text_len + visual_seq_len, D)
                hidden_states = outputs.hidden_states[-1]

                # 只取视觉部分的 hidden_states 来预测
                # 视觉部分从 text_len 开始
                # visual_hidden_states: (B, visual_seq_len, D)
                visual_hidden_states = hidden_states[:, input_ids.shape[1]:, :]

                # 使用你的自定义头进行预测
                # 将曲率转换为当前计算类型
                current_curvature = learnable_curvature.to(visual_hidden_states.dtype) if learnable_curvature is not None else None
                # logits: (B, visual_seq_len, codebook_size)
                logits = model.gen_head(visual_hidden_states, current_curvature)

                # --- 计算损失 ---
                # 确保 logits 是 float32 以计算损失
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.float().reshape(-1, codebook_size),
                                target_visual_tokens.reshape(-1))

            # --- 反向传播 ---
            loss_value = loss.item() # 获取损失值
            loss = loss / args.gradient_accumulation_steps

            # 使用 GradScaler 进行缩放
            scaler.scale(loss).backward()

            epoch_loss += loss_value # 累加未缩放的损失

            if (step + 1) % args.gradient_accumulation_steps == 0:
                 # 在 unscale 之前裁剪梯度 (可选，但推荐)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_norm=1.0 # 可以调整 max_norm
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True) # 使用 set_to_none 以提高效率
                global_step += 1

                # 更新进度条
                lr_value = scheduler.get_last_lr()[0]
                curve_value = learnable_curvature.item() if learnable_curvature is not None else float('nan')
                progress_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{lr_value:.2e}",
                    "curve": f"{curve_value:.3f}" if not math.isnan(curve_value) else "N/A",
                    "scale": f"{scaler.get_scale():.1f}" if compute_dtype == torch.float16 else "N/A"
                })

        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")

        # --- 验证 (可选) ---
        if (epoch + 1) % args.eval_interval == 0:
             avg_val_loss = evaluate(...)
             print(f"Epoch {epoch+1} 验证平均损失: {avg_val_loss:.4f}")
             # 可以根据 avg_val_loss 保存最佳模型

        # --- 保存模型 ---

        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}") # 使用 epoch 编号命名目录
        save_model_for_hf(
            model=model,
            hyperbolic_gen_head=hyperbolic_gen_head,
            learnable_curvature=learnable_curvature if args.use_hyperbolic else torch.tensor(0.0),
            output_dir=epoch_output_dir, # 使用特定 epoch 的目录
            processor=processor,
            args=args,
            hidden_size=hidden_size,
            codebook_size=codebook_size
        )

        checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pt") # 保存到固定文件名或带 epoch 的文件名
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True) # 确保目录存在

        # 准备要保存的状态
        # 注意：对于 LoRA，我们通常只保存适配器权重，但为了恢复优化器状态，
        # 保存整个 model.state_dict() 可能更简单，或者只保存可训练部分的状态。
        # 这里我们保存整个模型状态，加载时可能需要处理非严格匹配（如果模型结构变化）
        model_state_to_save = model.state_dict()
        # 如果只想保存可训练部分，可以这样做：
        # model_state_to_save = {
        #     name: param.cpu().clone() for name, param in model.named_parameters() if param.requires_grad
        # }

        checkpoint_data = {
            'epoch': epoch + 1, # 保存的是 *下一个* 要开始的 epoch
            'global_step': global_step,
            'model_state_dict': model_state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'learnable_curvature': learnable_curvature.item() if learnable_curvature is not None else None,
            'args': vars(args) # 保存命令行参数以供参考
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"检查点已保存到: {checkpoint_path}")
        except Exception as e:
            print(f"保存检查点失败: {e}")

        # --- 验证 (可选) ---
        # evaluate(...)

if __name__ == "__main__":
    # --- Add this before calling main ---
    try:
        # Set the start method *before* any CUDA calls or DataLoader creation
        import platform
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # Might happen if it was already set or in certain environments
        print(f"Note: Could not set start method to 'spawn': {e}")
    # --- End of addition ---
    main()