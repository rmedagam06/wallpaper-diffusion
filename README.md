# Wallpaper Diffusion

A text-to-image latent diffusion model trained from scratch on scraped wallpaper images. Given a short text prompt, the model generates a 512×512 image.

---

## Overview

The model follows the latent diffusion approach: instead of running the diffusion process in pixel space, images are first compressed into a lower-dimensional latent space using a variational autoencoder (VAE). The denoising network operates on these 64×64×4 latents rather than 512×512×3 pixels, which significantly reduces memory and compute requirements during training.

Text conditioning is handled by a frozen CLIP encoder. At each denoising step, the U-Net attends to the text embeddings via cross-attention, which is what causes the output to reflect the prompt.

---

## Architecture

**U-Net** (`models/unet.py`)  
The denoising network is a U-Net with skip connections between encoder and decoder stages. Residual blocks handle spatial feature extraction, and sinusoidal timestep embeddings inform each block about how much noise is currently present in the input. Cross-attention layers in the bottleneck and decoder attend to the 77×768 CLIP text embeddings. Total parameters: ~25M.

**Noise schedule** (`diffusion/ddpm.py`)  
A cosine schedule over T=1000 timesteps. The cosine schedule keeps the noise level from increasing too quickly in the early timesteps, compared to the original linear schedule from the DDPM paper.

**Sampler** (`diffusion/ddim.py`)  
DDIM sampling with 50 steps, using classifier-free guidance (CFG scale 7.5). DDIM treats the denoising process as a deterministic ODE rather than a stochastic one, which allows it to produce good results in far fewer steps than the full T=1000 reverse process. The null context for CFG is a zero embedding.

**VAE and text encoder** (`models/vae.py`, `models/text_encoder.py`)  
Both are loaded from pretrained Hugging Face weights (`stabilityai/sd-vae-ft-mse` and `openai/clip-vit-large-patch14`) and kept frozen throughout training.

---

## Training

Training runs for 200,000 steps with AdamW (lr=1e-4, weight decay=0.01), batch size 8, and mixed-precision arithmetic (fp16 via PyTorch AMP). An exponential moving average of the U-Net weights (decay=0.9999) is maintained alongside the online weights; the EMA weights are what get saved for inference, as they tend to produce cleaner outputs.

Gradient checkpointing is enabled on CUDA, which recomputes activations during the backward pass rather than storing them. This roughly halves peak VRAM usage at the cost of about 20% extra compute — necessary to fit training on a GTX 1050 (4 GB VRAM) for development runs.

Classifier-free guidance dropout: 10% of captions are replaced with empty strings during training, which teaches the model the unconditional distribution and is required for CFG to work at inference time.

Checkpoints are saved every 5,000 steps to `checkpoints/`. Training progress is logged to Weights & Biases.

**Hardware:** Development and smoke tests on a GTX 1050 (4 GB VRAM). Full 200k-step training on a Google Colab T4 (~6–10 hours).

---

## Dataset

Images were scraped from wallpaper sites across ten query categories (nature, architecture, space, abstract art, etc.) and preprocessed to 512×512. Approximate dataset size: 7,500 images for development, with the pipeline designed to scale to 50k+.

The raw and processed images are not included in this repository. To reproduce the dataset:

```bash
python data/scrape.py       # scrapes images + writes data/metadata.jsonl
python data/preprocess.py   # resizes, deduplicates, writes data/processed/512x512/
```

---

## LoRA Fine-tuning

`training/lora_finetune.py` applies LoRA to the attention projection layers (`to_q`, `to_k`, `to_v`, `to_out`) of the U-Net with rank 16. This freezes the base weights and trains only a small set of low-rank adapter matrices, reducing the number of trainable parameters to roughly 1–2% of the full model. Fine-tuning runs for 2,000 steps.

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers peft accelerate requests imagehash Pillow tqdm wandb pytorch-fid
```

Verify setup:

```bash
python -c "import torch; import diffusers; import transformers; print('Torch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Training

```bash
python training/train.py
```

Checkpoints are written to `checkpoints/ckpt_{step}.pt` every 5,000 steps. The final EMA weights are saved to `checkpoints/final_ema_unet.pt`.

---

## Inference

```bash
python generate.py checkpoints/final_ema_unet.pt "purple neon cityscape, wallpaper"
```

Output is saved to `output.png` (512×512). The script accepts any checkpoint and any prompt.

---

## Evaluation

Generate 2,000 sample images:

```bash
python eval/fid.py
```

Compute FID against the processed dataset:

```bash
python -m pytorch_fid eval/generated data/processed/512x512 --device cuda
```

FID (Fréchet Inception Distance) measures the distance between the distribution of generated images and real images using features from an Inception network. Lower is better.

---

## Repository Structure

```
models/         U-Net, VAE wrapper, CLIP text encoder
diffusion/      DDPM noise scheduler, DDIM sampler
data/           Scraper, preprocessor, PyTorch dataset class
training/       Main training loop, LoRA fine-tuning script
eval/           FID evaluation (image generation + score computation)
generate.py     Single-image inference script
PLAN.md         Implementation notes and checkpoint sequence
```
