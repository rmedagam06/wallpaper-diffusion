import torch
from transformers import CLIPTokenizer, CLIPTextModel


def load_clip(device="cpu"):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    encoder   = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return tokenizer, encoder


def encode_text(captions, tokenizer, encoder, device="cpu"):
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        return encoder(**tokens).last_hidden_state
