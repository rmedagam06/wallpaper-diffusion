import torch
import math


class DDPMScheduler:
    def __init__(self, T=1000):
        self.T = T
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, noise, t):
        ab = self.alpha_bar[t].to(x0.device)
        while ab.ndim < x0.ndim:
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        return self
