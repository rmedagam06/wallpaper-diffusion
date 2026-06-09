import torch
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet import SimpleUNet
from models.vae import load_vae
from models.text_encoder import load_clip, encode_text
from diffusion.ddpm import DDPMScheduler
from diffusion.ddim import DDIMSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_ema_unet.pt"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "purple neon cityscape, wallpaper"

vae                = load_vae(DEVICE)
tokenizer, encoder = load_clip(DEVICE)
scheduler          = DDPMScheduler().to(DEVICE)

unet = SimpleUNet().to(DEVICE)

# Support both raw state dict and checkpoint dict (saved by train.py)
ckpt = torch.load(CKPT, map_location=DEVICE)
if isinstance(ckpt, dict) and "ema_unet" in ckpt:
    unet.load_state_dict(ckpt["ema_unet"])
else:
    unet.load_state_dict(ckpt)
unet.eval()

sampler = DDIMSampler(scheduler, num_steps=50)
context = encode_text([PROMPT], tokenizer, encoder, DEVICE)

print(f"Generating: '{PROMPT}'")
latent = sampler.sample(unet, (1, 4, 64, 64), context, cfg_scale=7.5, device=DEVICE)

with torch.no_grad():
    img_tensor = vae.decode(latent / 0.18215).sample
img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
img = Image.fromarray(img_np)
img.save("output.png")
print("Saved to output.png")
