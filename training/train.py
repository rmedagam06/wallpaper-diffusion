import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import copy
import sys
import wandb

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WallpaperDataset
from models.unet import SimpleUNet
from models.vae import load_vae
from models.text_encoder import load_clip, encode_text
from diffusion.ddpm import DDPMScheduler

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 8
LR              = 1e-4
TOTAL_STEPS     = 200_000
SAVE_EVERY      = 5_000
LOG_EVERY       = 100
CFG_DROP        = 0.1
GRAD_CHECKPOINT = (DEVICE == "cuda")
CKPT_DIR        = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

wandb.init(
    project="wallpaper-diffusion",
    config=dict(batch_size=BATCH_SIZE, lr=LR, total_steps=TOTAL_STEPS,
                cfg_drop=CFG_DROP, grad_checkpoint=GRAD_CHECKPOINT)
)

vae                = load_vae(DEVICE)
tokenizer, encoder = load_clip(DEVICE)
unet               = SimpleUNet().to(DEVICE)
scheduler          = DDPMScheduler().to(DEVICE)
ema_unet           = copy.deepcopy(unet)
optimizer          = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=0.01)
scaler             = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

dataset = WallpaperDataset("data/metadata_processed.jsonl")
loader  = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

print(f"Training on {len(dataset)} images  |  device={DEVICE}  |  steps={TOTAL_STEPS}")

step = 0
for epoch in range(9999):
    for imgs, captions in loader:
        if step >= TOTAL_STEPS:
            break
        imgs = imgs.to(DEVICE)

        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

        captions = ["" if torch.rand(1).item() < CFG_DROP else c for c in captions]
        context  = encode_text(captions, tokenizer, encoder, DEVICE)

        noise  = torch.randn_like(latents)
        t      = torch.randint(0, scheduler.T, (BATCH_SIZE,), device=DEVICE)
        noisy  = scheduler.add_noise(latents, noise, t)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            pred = unet(noisy, t, context, use_grad_checkpoint=GRAD_CHECKPOINT)
            loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            for p_ema, p in zip(ema_unet.parameters(), unet.parameters()):
                p_ema.data.mul_(0.9999).add_(p.data, alpha=1 - 0.9999)

        step += 1
        if step % LOG_EVERY == 0:
            print(f"Step {step:>6}/{TOTAL_STEPS}  loss={loss.item():.4f}")
            wandb.log({"loss": loss.item(), "step": step})
        if step % SAVE_EVERY == 0:
            torch.save(
                {"step": step, "unet": unet.state_dict(),
                 "ema_unet": ema_unet.state_dict(),
                 "optimizer": optimizer.state_dict()},
                CKPT_DIR / f"ckpt_{step}.pt"
            )
            print(f"  Checkpoint saved: checkpoints/ckpt_{step}.pt")

    if step >= TOTAL_STEPS:
        break

torch.save(ema_unet.state_dict(), CKPT_DIR / "final_ema_unet.pt")
print("Training complete. Final weights: checkpoints/final_ema_unet.pt")
wandb.finish()
