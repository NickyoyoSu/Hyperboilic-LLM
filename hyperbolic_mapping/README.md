### Hyperbolic Multimodal Mapping

This repository provides training and inference scripts for language and multimodal modeling with hyperbolic mappings (Lorentz geometry). It covers: (1) pure-text language modeling, (2) image–text alignment on COCO captions, and (3) text-to-image visual token generation. The code supports Euclidean vs. hyperbolic heads, learnable curvature, LoRA fine-tuning, and multi-GPU training.

### Key Features
- **Geometry switch**: Toggle Euclidean/Hyperbolic with `--use_hyperbolic`.
- **Learnable curvature**: Curvature is optimized jointly with the mapping head.
- **Parameter-efficient fine-tuning**: LoRA support via `--use_lora`.
- **Multi-GPU**: DataParallel with `--use_parallel`.
- **Multiple bases**: Llama (1B for text), MLLama 11B-Vision (image–text), and DeepSeek Janus-Pro-7B (multimodal + VQ visual tokens).
- **Text→image tokens**: Train a generation head to predict VQModel visual tokens for text-to-image workflows.

### Repository Structure
- `text_mapping.py`: Text LM head training on WikiText-103 with optional hyperbolic mapping (base: `meta-llama/Llama-3.2-1B`).
- `image_mapping.py`: COCO image–caption training with MLLama; includes a Stable Diffusion demo for text-to-image.
- `multimodal_mapping.py`: Janus-Pro-7B pipeline for image–text; supports multiple modes (`train/eval/demo/research`).
- `multimodel_mapping_image.py`: Train a hyperbolic/euclidean generation head to predict visual tokens; saves in HF-compatible format.
- `multimodel_mapping_1B.py`: A variant of the above with a slightly simplified training loop.
- `inferance.py`: Inference from `.pth` checkpoints (image→text and text→image).
- `final_test.py`: Run a single-image captioning demo from a saved HF directory (Janus).
- `test_image.py`: Quick connectivity test for MLLama processor/model.
- `1.py` / `2.py`: Experimental variants of `multimodal_mapping.py`.

### Setup
- Python 3.9/3.10; CUDA-capable GPU recommended.
- Install dependencies (example):
```bash
pip install transformers datasets peft diffusers accelerate pillow scikit-learn matplotlib tqdm pycocotools
pip install git+https://github.com/deepseek-ai/Janus.git
```
- Make sure `HyperLib` is importable (clone and export `PYTHONPATH`):
```bash
export PYTHONPATH="/path/to/HyperLib:${PYTHONPATH}"
```
- Log in to Hugging Face to avoid hardcoding tokens:
```bash
huggingface-cli login
# or
export HF_TOKEN=********
```
- Precision and devices: several scripts default to bfloat16/float16; switch to float32 and `--device cpu` if running on CPU/Apple Silicon (expect slower performance).

### Data (COCO 2017)
Expected layout under `coco_dataset/`:
```
coco_dataset/
  annotations/
    captions_train2017.json
    captions_val2017.json
  train2017/
  val2017/
```
If your paths differ, adjust `data_dir` in scripts or add a symlink.

### Quickstart
- Text LM (Llama-3.2-1B):
```bash
python text_mapping.py \
  --use_hyperbolic \
  --use_lora \
  --num_epochs 3 \
  --batch_size 16 \
  --device cuda:0
```

- Multimodal (Janus-Pro-7B):
```bash
python multimodal_mapping.py \
  --use_hyperbolic \
  --use_lora \
  --batch_size 2 \
  --device cuda \
  --mode train
```
Multi-GPU: add `--use_parallel` and set CUDA devices appropriately.

- Image–Text (MLLama 11B-Vision):
```bash
python image_mapping.py \
  --use_hyperbolic \
  --use_lora \
  --batch_size 1 \
  --image_size 224 \
  --device cuda
```

- Text→Image visual tokens (Janus):
```bash
python multimodel_mapping_image.py \
  --model_path /path/to/janus-pro-7b \
  --coco_data_dir ./coco_dataset \
  --use_hyperbolic \
  --head_num_layers 3 \
  --device cuda
```
(`multimodel_mapping_1B.py` has similar usage; swap the script name as needed.)

- Inference from checkpoint:
```bash
python inferance.py \
  --checkpoint /path/to/checkpoint.pth \
  --image /path/to/image.jpg \
  --text "a cat sitting on a bench" \
  --device cuda
```

- Demo from HF directory:
```bash
python final_test.py \
  --model_dir /path/to/saved_hf_dir \
  --image_path /path/to/image.jpg \
  --device cuda \
  --show_image
```

### Training Artifacts & Resuming
- Some scripts save checkpoints under timestamped directories. To change locations, search in the script for save paths and modify accordingly.
- Resuming/evaluation: use script flags such as `--resume`, `--start_epoch`, and `--mode eval` when supported.

### Tips & Troubleshooting
- COCO not found: ensure `annotations/` and `{train,val}2017/` exist under `coco_dataset/`.
- Out-of-memory: reduce `--batch_size`, lower `--image_size`, use fewer head layers (e.g., `--head_num_layers 2`), or disable LoRA.
- Precision: bfloat16/float16 require GPU support; on CPU/Apple Silicon, switch to float32 and `--device cpu`.
- `trust_remote_code` errors: ensure recent `transformers` and Janus are installed.
- HyperLib dtype/attribute issues: match your `HyperLib` version to the code (e.g., `LorentzMLR` internals like `z/a`).
- Security: avoid hardcoding HF tokens; prefer `huggingface-cli login` or environment variables.

### Acknowledgements
- DeepSeek-AI Janus (`deepseek-ai/Janus-Pro-7B`)
- Meta Llama (`meta-llama/Llama-3.2-*`)
- Stability AI Stable Diffusion
- MS COCO Dataset
- `HyperLib` (Lorentz geometry components)

### Current progress
For text_mapping.py I find after severel eooches hyperbolic version is slitly better than euclidean one
![image](https://github.com/NickyoyoSu/Hyperboilic-LLM/blob/main/hyperbolic_mapping/images/loss.pic.jpg)
![image](https://github.com/NickyoyoSu/Hyperboilic-LLM/blob/main/hyperbolic_mapping/images/perplexity.pic.jpg)
For image mapping, for some reason the experiment didn't finished.

Downstrean tasks should be finished in the future.

