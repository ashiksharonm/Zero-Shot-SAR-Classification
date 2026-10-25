"""Model loading and embedding extraction (open_clip backend)."""
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .config import MODELS

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(key):
    _, arch, pretrained, ckpt = MODELS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    if ckpt is not None:
        state = torch.load(hf_hub_download(*ckpt), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all("logit_bias" in m for m in missing), (missing, unexpected)
    # fp16 on GPU/MPS: ~2.6x faster for ViT-L/14; embeddings match fp32 to cosine >= 0.9999.
    model = model.to(DEVICE).eval()
    if DEVICE != "cpu":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(arch)
    norm = next(t for t in preprocess.transforms if t.__class__.__name__ == "Normalize")
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    return model, tokenizer, (norm.mean, norm.std), size


@torch.no_grad()
def encode_images(model, imgs_u8, norm, size, batch_size=128):
    """imgs_u8: [N,H,W,3] uint8. Bicubic resize to the model resolution (patches are
    square, so this equals the default resize+center-crop), then CLIP normalisation."""
    mean = torch.tensor(norm[0], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(norm[1], device=DEVICE).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs_u8), batch_size):
        x = torch.from_numpy(imgs_u8[i:i + batch_size]).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        f = model.encode_image(((x - mean) / std).to(next(model.parameters()).dtype))
        out.append(F.normalize(f, dim=-1).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_texts(model, tokenizer, texts):
    f = model.encode_text(tokenizer(texts).to(DEVICE))
    return F.normalize(f, dim=-1).float().cpu().numpy()


def class_embeddings(model, tokenizer, templates, names):
    """Average of normalised per-template text embeddings, re-normalised (prompt ensembling).
    Returns [K, D]."""
    embs = np.stack([encode_texts(model, tokenizer, [t.format(n) for n in names]) for t in templates])
    m = embs.mean(0)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)
