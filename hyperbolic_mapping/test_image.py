import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, MllamaForConditionalGeneration

# 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
hf_token = "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"

# 加载处理器和模型
print("加载处理器和模型...")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision", token=hf_token)
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

def process_image(image_path):
    try:
        # 加载和显示图像信息
        image = Image.open(image_path)
        print(f"\n处理图像: {image_path}")
        print(f"图像大小: {image.size}, 模式: {image.mode}")
        
        # 进行不同的预处理
        image_rgb = image.convert("RGB")
        image_resized = image_rgb.resize((224, 224))
        
        # 使用processor处理图像
        inputs = processor(
            images=image_resized, 
            text="描述这张图片:", 
            return_tensors="pt",
            do_rescale=True
        ).to(device)
        
        # 输出处理后的tensor形状
        print("处理后输入形状:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        # 尝试推理
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"生成的描述: {generated_text}")
            return True
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 测试真实图像和生成图像
test_images = ["sample_image.jpg", "generated_image.jpg"]
results = {}

for img_path in test_images:
    success = process_image(img_path)
    results[img_path] = "成功" if success else "失败"

print("\n测试结果总结:")
for img_path, result in results.items():
    print(f"{img_path}: {result}")