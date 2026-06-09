# Wallpaper Diffusion Model — Full Implementation Plan

## Context
Building a text-to-image latent diffusion model from scratch in PyTorch to generate phone/laptop wallpapers.
Resume description: custom U-Net backbone, cosine DDPM noise schedule, DDIM for 20× faster inference,
LoRA for 60% compute reduction on fine-tuning, multi-GPU training with gradient checkpointing + mixed
precision, FID evaluation against a GAN baseline, trained on 50k+ scraped images.

Phase 1 (environment, model smoke tests) is **complete**. Remaining work:
1. Code enhancements to `train.py` (wandb logging + gradient checkpointing)
2. Data pipeline (scrape → preprocess)
3. Full training run
4. DDIM inference verification
5. LoRA fine-tuning
6. FID evaluation

---

## GPU Setup — GTX 1050 local + Colab for full training

The GTX 1050 (640 CUDA cores, 2–4 GB VRAM) can run all smoke tests and the data pipeline locally.
Training 200k steps at batch_size=8 would take ~40–60 hours on a 1050. Split:
- **Checkpoints 2–4A (dev + dry run):** run locally on GTX 1050
- **Checkpoint 4C (full 200k-step training):** run on Google Colab T4 (free, ~6–10h)

**Install CUDA torch locally first:**
```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True  NVIDIA GeForce GTX 1050`

**For Colab full training:** upload the project folder (or clone from GitHub), install deps, then run
`python training/train.py` with `BATCH_SIZE=8`. Colab T4 has 15 GB VRAM — no VRAM constraints.

---

## Phase 1: Environment & Smoke Tests — COMPLETE ✅

### Checkpoint 1A — Dependencies installed ✅
```
pip install torch torchvision diffusers transformers peft accelerate requests imagehash Pillow tqdm wandb pytorch-fid
```

### Checkpoint 1B — Import check ✅
```
python -c "import torch; import diffusers; import transformers; print('All good. Torch version:', torch.__version__)"
```
Result: `All good. Torch version: 2.12.0+cpu`

### Checkpoint 1C — UNet forward pass ✅
```
python -c "import torch; from models.unet import SimpleUNet; model = SimpleUNet(); x = torch.randn(2,4,64,64); t = torch.randint(0,1000,(2,)); ctx = torch.randn(2,77,768); out = model(x,t,ctx); print('Output:', out.shape, '| Params:', sum(p.numel() for p in model.parameters())/1e6, 'M')"
```
Result: `Output: torch.Size([2, 4, 64, 64]) | Params: 24.87 M`

### Checkpoint 1D — VAE + CLIP pipeline ✅
```
python -c "import torch; from models.vae import load_vae; from models.text_encoder import load_clip, encode_text; vae=load_vae(); tok,enc=load_clip(); img=torch.randn(1,3,512,512); torch.set_grad_enabled(False); lat=vae.encode(img).latent_dist.sample()*0.18215; dec=vae.decode(lat/0.18215).sample; emb=encode_text(['test'],tok,enc); print(img.shape, lat.shape, dec.shape, emb.shape)"
```
Result: `[1,3,512,512]  [1,4,64,64]  [1,3,512,512]  [1,77,768]`

---

## Phase 2: Code Enhancements (before training)

Two features needed to match the resume: **gradient checkpointing** and **wandb logging**.

### 2A — Add gradient checkpointing to UNet
**File:** `models/unet.py`

Wrap encoder and bottleneck forward calls with `torch.utils.checkpoint.checkpoint`. Reduces peak
VRAM ~40% at the cost of ~20% extra compute — essential for fitting on the GTX 1050 (4 GB).

Pattern applied inside `SimpleUNet.forward`:
```python
from torch.utils.checkpoint import checkpoint
h1 = checkpoint(self.enc1, h0, t_emb, use_reentrant=False)
```
Apply to: `enc1`, `enc2`, `enc3`, `mid1`, `mid_attn`, `mid2`.

### 2B — Add wandb logging to train.py
**File:** `training/train.py`

- `wandb.init(project="wallpaper-diffusion", config={...})` at startup
- `wandb.log({"loss": loss.item(), "step": step})` every `LOG_EVERY` steps
- Log one generated sample image every `SAVE_EVERY` steps

### Checkpoint 2 verification
```
python -c "from torch.utils.checkpoint import checkpoint; import wandb; print('Phase 2 deps ok')"
```

---

## Phase 3: Data Pipeline

### 3A — Smoke-scrape (1 query, 2 pages ≈ 48 images)
Edit `data/scrape.py` temporarily: `QUERIES = ["nature"]`, `range(1, 3)`, then:
```
python data/scrape.py
```
Expected: `data/raw/` with ~48 JPEGs, `data/metadata.jsonl` with ~48 rows.
Restore QUERIES + page range after smoke test passes.

### 3B — Run preprocessor
```
python data/preprocess.py
```
Expected: `data/processed/512x512/` with deduped 512×512 images, `data/metadata_processed.jsonl`.

### Checkpoint 3 — Dataset smoke test
```
python -c "from data.dataset import WallpaperDataset; ds = WallpaperDataset('data/metadata_processed.jsonl'); img, cap = ds[0]; print('img:', img.shape, '| caption:', cap[:60])"
```
Expected: `img: torch.Size([3, 512, 512]) | caption: <some tag string>`

### 3C — Full scrape (run overnight to hit 50k+ images)
Restore `data/scrape.py` to all 10 queries, 200 pages each, run:
```
python data/scrape.py
python data/preprocess.py
```

---

## Phase 4: Training

### 4A — CUDA check
```
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```
Must show `CUDA: True` before training.

### 4B — Dry run (10 steps)
Temporarily set `TOTAL_STEPS = 10`, `BATCH_SIZE = 2` in `training/train.py`:
```
python training/train.py
```
Expected: 10 steps print without crash, loss logged, no VRAM error.

### 4C — Full training (200k steps) — recommended on Colab T4 (~6–10h)
Restore `TOTAL_STEPS = 200_000`, `BATCH_SIZE = 8`. On Colab, upload project or clone, install deps, run:
```
python training/train.py
```
On local GTX 1050 (optional): reduce `BATCH_SIZE = 2`, expect ~40–60h.
Checkpoints saved every 5k steps → `checkpoints/ckpt_*.pt`.

### Checkpoint 4 verification
```
ls checkpoints/
```
Should list `final_ema_unet.pt`. Check wandb dashboard for declining loss curve.

---

## Phase 5: DDIM Inference

### Checkpoint 5A — DDIM sampler smoke test (untrained weights)
```
python -c "import torch; from models.unet import SimpleUNet; from diffusion.ddpm import DDPMScheduler; from diffusion.ddim import DDIMSampler; unet=SimpleUNet(); sched=DDPMScheduler(); sampler=DDIMSampler(sched,50); ctx=torch.zeros(1,77,768); out=sampler.sample(unet,(1,4,64,64),ctx); print('DDIM output:', out.shape)"
```
Expected: `DDIM output: torch.Size([1, 4, 64, 64])`

### Checkpoint 5B — Generate wallpaper from trained checkpoint
```
python generate.py checkpoints/final_ema_unet.pt "purple neon cityscape, wallpaper"
```
Expected: `output.png` saved (512×512). Visually inspect — should look like a neon cityscape, not noise.

Try additional prompts to verify text conditioning:
```
python generate.py checkpoints/final_ema_unet.pt "minimalist forest, soft light"
python generate.py checkpoints/final_ema_unet.pt "abstract geometric, deep space"
```

---

## Phase 6: LoRA Fine-tuning

### Checkpoint 6A — LoRA dry run (10 steps)
Temporarily set `STEPS = 10` in `training/lora_finetune.py`:
```
python training/lora_finetune.py
```
Expected:
```
trainable params: X || all params: Y || trainable%: ~1–2%
LoRA step 10/10  loss=...
LoRA weights saved to checkpoints/lora/lora_weights/
```

### Checkpoint 6B — Full LoRA (2k steps)
Restore `STEPS = 2000`:
```
python training/lora_finetune.py
```
Expected: `checkpoints/lora/lora_weights/` directory with `adapter_config.json` + weight files.

---

## Phase 7: FID Evaluation

### Checkpoint 7A — Generate 2k evaluation images
```
python eval/fid.py
```
Expected: 2000 PNGs in `eval/generated/`, progress printed every 100 images.

### Checkpoint 7B — Compute FID score
```
python -m pytorch_fid eval/generated data/processed/512x512 --device cuda
```
Expected: FID number printed. Target: at least 35% lower than GAN baseline.
(Typical GAN baseline on wallpaper-scale datasets: FID ~80–150; target <65.)

---

## Files to Modify (Remaining)

| File | Change |
|------|--------|
| `models/unet.py` | Wrap encoder + bottleneck in `torch.utils.checkpoint.checkpoint` |
| `training/train.py` | Add wandb logging; add gradient checkpointing flag |

## Files Already Complete (no changes needed)
`models/vae.py` · `models/text_encoder.py` · `diffusion/ddpm.py` · `diffusion/ddim.py`
`data/scrape.py` · `data/preprocess.py` · `data/dataset.py`
`training/lora_finetune.py` · `eval/fid.py` · `generate.py`

---

## Quick-Reference Checkpoint Sequence

| # | Command snippet | Expected result |
|---|-----------------|-----------------|
| 1A–1D | (done) | ✅ |
| 2 | import grad_ckpt + wandb | no errors |
| 3 | `ds[0]` shape check | `[3, 512, 512]` |
| 4A | CUDA check | `CUDA: True` |
| 4B | 10-step dry run | loss prints, no crash |
| 4C | full training | `final_ema_unet.pt` saved |
| 5A | DDIM shape check | `[1, 4, 64, 64]` |
| 5B | `generate.py` | `output.png` visual check |
| 6A | LoRA dry run | trainable% ~1–2% |
| 6B | full LoRA | `lora_weights/` saved |
| 7A | `eval/fid.py` | 2000 PNGs in `eval/generated/` |
| 7B | pytorch-fid | FID score printed |
