import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        def split_heads(t):
            return t.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=128, time_dim=512, context_dim=768):
        super().__init__()
        ch = base_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        # Encoder
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.enc1 = ResBlock(ch,     ch,   time_dim)
        self.enc2 = ResBlock(ch,     ch*2, time_dim)
        self.down2 = nn.Conv2d(ch*2, ch*2, 2, stride=2)
        self.enc3 = ResBlock(ch*2,   ch*4, time_dim)
        self.down3 = nn.Conv2d(ch*4, ch*4, 2, stride=2)
        # Bottleneck
        self.mid1 = ResBlock(ch*4, ch*4, time_dim)
        self.mid_attn = CrossAttention(ch*4, context_dim)
        self.mid2 = ResBlock(ch*4, ch*4, time_dim)
        # Decoder
        self.up3 = nn.ConvTranspose2d(ch*4, ch*4, 2, stride=2)
        self.dec3 = ResBlock(ch*4 + ch*4, ch*2, time_dim)
        self.dec3_attn = CrossAttention(ch*2, context_dim)
        self.up2 = nn.ConvTranspose2d(ch*2, ch*2, 2, stride=2)
        self.dec2 = ResBlock(ch*2 + ch*2, ch, time_dim)
        self.dec1 = ResBlock(ch + ch,     ch, time_dim)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t, context, use_grad_checkpoint=False):
        t_emb = self.time_mlp(timestep_embedding(t, 128))
        h0  = self.conv_in(x)

        if use_grad_checkpoint:
            h1  = checkpoint(self.enc1, h0, t_emb, use_reentrant=False)
            h2  = checkpoint(self.enc2, h1, t_emb, use_reentrant=False)
            h2d = self.down2(h2)
            h3  = checkpoint(self.enc3, h2d, t_emb, use_reentrant=False)
            h3d = self.down3(h3)
            hm  = checkpoint(self.mid1, h3d, t_emb, use_reentrant=False)
            hm  = checkpoint(self.mid_attn, hm, context, use_reentrant=False)
            hm  = checkpoint(self.mid2, hm, t_emb, use_reentrant=False)
        else:
            h1  = self.enc1(h0, t_emb)
            h2  = self.enc2(h1, t_emb)
            h2d = self.down2(h2)
            h3  = self.enc3(h2d, t_emb)
            h3d = self.down3(h3)
            hm  = self.mid1(h3d, t_emb)
            hm  = self.mid_attn(hm, context)
            hm  = self.mid2(hm, t_emb)

        hd3 = self.dec3(torch.cat([self.up3(hm), h3], dim=1), t_emb)
        hd3 = self.dec3_attn(hd3, context)
        hd2 = self.dec2(torch.cat([self.up2(hd3), h2], dim=1), t_emb)
        hd1 = self.dec1(torch.cat([hd2, h1], dim=1), t_emb)
        return self.conv_out(hd1)
