import torch
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet import SimpleUNet
from models.vae import load_vae
from models.text_encoder import load_clip, encode_text
from diffusion.ddpm import DDPMScheduler
from diffusion.ddim import DDIMSampler

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 2_000
OUT_DIR   = Path("eval/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

unet = SimpleUNet().to(DEVICE)
unet.load_state_dict(torch.load("checkpoints/final_ema_unet.pt", map_location=DEVICE))
unet.eval()

vae                = load_vae(DEVICE)
tokenizer, encoder = load_clip(DEVICE)
scheduler          = DDPMScheduler().to(DEVICE)
sampler            = DDIMSampler(scheduler, num_steps=50)

prompts = [
    "nature wallpaper", "abstract art", "cityscape at night",
    "space galaxy", "minimal design"
] * (N_SAMPLES // 5 + 1)

print(f"Generating {N_SAMPLES} images for FID evaluation...")
for i in range(N_SAMPLES):
    context = encode_text([prompts[i]], tokenizer, encoder, DEVICE)
    latent  = sampler.sample(unet, (1, 4, 64, 64), context, device=DEVICE)
    with torch.no_grad():
        img_t = vae.decode(latent / 0.18215).sample
    img_t  = (img_t.clamp(-1, 1) + 1) / 2
    img_np = (img_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(img_np).save(OUT_DIR / f"{i:05d}.png")
    if i % 100 == 0:
        print(f"  {i}/{N_SAMPLES}")

print(f"\nDone. Generated images saved to eval/generated/")
print(f"Now run FID:")
print(f"  python -m pytorch_fid eval/generated data/processed/512x512")
