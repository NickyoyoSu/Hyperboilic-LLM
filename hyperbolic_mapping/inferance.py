import os
import sys
import torch
import argparse
from PIL import Image
import requests
from io import BytesIO
import torch.nn as nn
from transformers import AutoProcessor, MllamaForConditionalGeneration
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from HyperLib.geoopt.manifolds.lorentz.math import expmap0
from HyperLib.lorentz.layers.LMLR import LorentzMLR
from HyperLib.lorentz.manifold import CustomLorentz

# 参数解析
parser = argparse.ArgumentParser(description="使用训练好的检查点进行多模态推理")
parser.add_argument("--checkpoint", type=str, required=True, help="检查点文件路径 (.pth)")
parser.add_argument("--lora_dir", type=str, default="", help="LoRA目录路径（如果使用LoRA）")
parser.add_argument("--image", type=str, default="", help="输入图像路径或URL")
parser.add_argument("--text", type=str, default="", help="用于生成图像的提示文本")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--image_size", type=int, default=224, help="输入图像大小")
args = parser.parse_args()

# 设备设置
device = torch.device(args.device)

# 自定义多模态映射头
class MultimodalMappingHead(nn.Module):
    def __init__(self, base_model, use_hyperbolic=True, num_layers=3):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        
        # 获取config
        if isinstance(base_model, nn.DataParallel):
            config = base_model.module.config
        else:
            config = base_model.config
        
        # 获取vocab_size
        if hasattr(config, 'text_config'):
            self.vocab_size = config.text_config.vocab_size
        elif hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            self.vocab_size = 32000  # 默认值
        
        # 获取hidden_size
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        elif hasattr(config, 'text_config'):
            self.hidden_size = config.text_config.hidden_size
        else:
            self.hidden_size = 4096  # 默认值
        
        self.num_layers = num_layers
        self.manifold = CustomLorentz()

        # 多模态适配器
        self.multimodal_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 线性层和归一化
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        
        if num_layers >= 2:
            self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm2 = nn.LayerNorm(self.hidden_size)
        
        if num_layers >= 3:
            self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.norm3 = nn.LayerNorm(self.hidden_size)

        # 分类器
        self.hyp_cls = LorentzMLR(
            self.manifold,
            num_features=self.hidden_size + 1, 
            num_classes=self.vocab_size
        )
        self.euc_cls = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def lorentz_map(self, x, c_param):
        return expmap0(x, k=c_param, dim=-1)
    
    def forward(self, last_hidden_states, c_param):
        # 多模态适配
        x = self.multimodal_adapter(last_hidden_states)
        
        # 第一层
        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        
        # 第二层(如果有)
        if self.num_layers >= 2:
            x = self.linear2(x)
            x = self.norm2(x)
            x = torch.relu(x)
        
        # 第三层(如果有)
        if self.num_layers >= 3:
            x = self.linear3(x)
            x = self.norm3(x)
            x = torch.relu(x)
        
        # 基于几何选择分类器
        if self.use_hyperbolic:
            x = self.manifold.add_time(x)
            hyper_embs = self.lorentz_map(x, c_param)
            logits = self.hyp_cls(hyper_embs)
        else:
            logits = self.euc_cls(x)

        return logits

def load_model():
    """加载MLLama-3.2-Vision模型"""
    print("加载MLLama-3.2-Vision模型...")
    hf_token = "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"
    model_name = "meta-llama/Llama-3.2-11B-Vision"
    
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    
    try:
        # 加载MLLama模型
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name, 
            token=hf_token,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"错误：无法加载原始训练模型 {model_name}")
        print(f"详细错误: {e}")
        print("请确保您有足够的GPU内存并正确配置环境")
        sys.exit(1)
    
    # 加载StableDiffusion用于图像生成
    print("加载Stable Diffusion模型...")
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
        )
    ).to(device)
    
    return processor, model, image_gen_model

def load_checkpoint(checkpoint_path, model, lora_dir=""):
    """加载训练好的检查点"""
    print(f"加载检查点: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否是超曲模型
    is_hyperbolic = "hyp" in os.path.basename(checkpoint_path)
    
    # 创建并加载自定义头部
    custom_lm_head = MultimodalMappingHead(model, use_hyperbolic=is_hyperbolic, num_layers=3).to(device)
    custom_lm_head.load_state_dict(checkpoint["lm_head_state"])
    
    # 加载曲率参数
    curvature = torch.tensor(checkpoint["curvature"], device=device)
    
    # 加载LoRA权重(如果指定)
    if lora_dir and os.path.exists(lora_dir):
        print(f"加载LoRA权重: {lora_dir}")
        if isinstance(model, nn.DataParallel):
            model.module.load_adapter(lora_dir)
        else:
            model.load_adapter(lora_dir)
    elif "lora" in checkpoint_path.lower():
        # 尝试从检查点路径找到LoRA
        parent_dir = os.path.dirname(checkpoint_path)
        lora_dirs = [d for d in os.listdir(parent_dir) if d.startswith("lora_")]
        if lora_dirs:
            newest_lora = sorted(lora_dirs)[-1]
            lora_path = os.path.join(parent_dir, newest_lora)
            print(f"自动加载LoRA权重: {lora_path}")
            if isinstance(model, nn.DataParallel):
                model.module.load_adapter(lora_path)
            else:
                model.load_adapter(lora_path)
    
    return custom_lm_head, curvature, is_hyperbolic

def generate_text_from_image(image_path, model, processor, custom_lm_head, curvature, max_len=100):
    """从图像生成文本"""
    model.eval()
    custom_lm_head.eval()
    
    # 加载图像
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # 处理图像
    inputs = processor(images=image, text="描述这张图片:", return_tensors="pt").to(device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 解码输出
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text

def generate_image_from_text(text_prompt, image_gen_model, output_path=None):
    """从文本生成图像"""
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                      dtype=torch.float16):
        image = image_gen_model(text_prompt, guidance_scale=7.5).images[0]
    
    if output_path:
        image.save(output_path)
    
    return image

# 主函数
def main():
    # 加载模型
    processor, model, image_gen_model = load_model()
    
    # 加载检查点
    custom_lm_head, curvature, is_hyperbolic = load_checkpoint(
        args.checkpoint, model, args.lora_dir
    )
    
    print(f"{'='*50}")
    print(f"模型已加载 - 几何空间: {'超曲率' if is_hyperbolic else '欧几里得'}, 曲率: {curvature.item():.4f}")
    print(f"{'='*50}")
    
    # 图像到文本
    if args.image:
        print(f"\n=== 从图像生成文本 ===")
        generated_text = generate_text_from_image(
            args.image, model, processor, custom_lm_head, curvature
        )
        print(f"图像路径: {args.image}")
        print(f"生成的描述: {generated_text}")
    
    # 文本到图像
    if args.text:
        print(f"\n=== 从文本生成图像 ===")
        output_path = "generated_image.jpg"
        generate_image_from_text(args.text, image_gen_model, output_path)
        print(f"提示文本: {args.text}")
        print(f"生成的图像已保存到: {output_path}")

if __name__ == "__main__":
    main()