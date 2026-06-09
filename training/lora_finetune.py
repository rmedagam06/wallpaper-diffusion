import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WallpaperDataset
from models.unet import SimpleUNet
from models.vae import load_vae
from models.text_encoder import load_clip, encode_text
from diffusion.ddpm import DDPMScheduler

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CKPT  = "checkpoints/final_ema_unet.pt"
STYLE_DATA = "data/metadata_processed.jsonl"
LORA_RANK  = 16
LR         = 1e-4
STEPS      = 2_000
CKPT_DIR   = Path("checkpoints/lora")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

unet = SimpleUNet().to(DEVICE)
unet.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE))

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    lora_dropout=0.1,
    bias="none",
)
unet = get_peft_model(unet, lora_cfg)
unet.print_trainable_parameters()

vae                = load_vae(DEVICE)
tokenizer, encoder = load_clip(DEVICE)
scheduler          = DDPMScheduler().to(DEVICE)
optimizer          = torch.optim.AdamW(unet.parameters(), lr=LR)

dataset = WallpaperDataset(STYLE_DATA)
loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)

print(f"LoRA fine-tuning for {STEPS} steps on {len(dataset)} images  |  device={DEVICE}")
step = 0
for imgs, captions in loader:
    if step >= STEPS:
        break
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        latents = vae.encode(imgs).latent_dist.sample() * 0.18215
    context = encode_text(list(captions), tokenizer, encoder, DEVICE)
    noise   = torch.randn_like(latents)
    t       = torch.randint(0, scheduler.T, (4,), device=DEVICE)
    noisy   = scheduler.add_noise(latents, noise, t)
    pred    = unet(noisy, t, context)
    loss    = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
    if step % 100 == 0:
        print(f"LoRA step {step}/{STEPS}  loss={loss.item():.4f}")

unet.save_pretrained(str(CKPT_DIR / "lora_weights"))
print(f"LoRA weights saved to checkpoints/lora/lora_weights/")
