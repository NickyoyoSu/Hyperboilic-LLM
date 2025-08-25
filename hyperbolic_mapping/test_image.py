import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hf_token = "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"

# Load processor and model
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision", token=hf_token)
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

def process_image(image_path):
    try:
        # Load and display image info
        image = Image.open(image_path)
        print(f"\nProcessing image: {image_path}")
        print(f"Image size: {image.size}, Mode: {image.mode}")
        
        # Perform different preprocessing steps
        image_rgb = image.convert("RGB")
        image_resized = image_rgb.resize((224, 224))
        
        # Use processor to handle the image
        inputs = processor(
            images=image_resized, 
            text="Describe this image:", 
            return_tensors="pt",
            do_rescale=True
        ).to(device)
        
        # Output the processed tensor shapes
        print("Processed input shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        # Try inference
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"Generated description: {generated_text}")
            return True
            
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test with real and generated images
test_images = ["sample_image.jpg", "generated_image.jpg"]
results = {}

for img_path in test_images:
    success = process_image(img_path)
    results[img_path] = "Success" if success else "Failed"

print("\nTest results summary:")
for img_path, result in results.items():
    print(f"{img_path}: {result}")
