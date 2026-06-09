import torch
from diffusers import AutoencoderKL


def load_vae(device="cpu"):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae
