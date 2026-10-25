"""Step 1: compute and cache L2-normalised image and text embeddings for every model.

Usage: python extract_features.py [model_key ...]   (default: all models in config.MODELS)
"""
import json
import sys
import time

import numpy as np
import torch

from sarzs.config import (CACHE_DIR, HIERARCHIES, INPUT_MODES, MODELS, PHYSICS_DESCRIPTIONS,
                          PROMPT_TEMPLATES, READABLE, ROTATIONS)
from sarzs.data import db_ranges, encode, load_raw
from sarzs.models import DEVICE, encode_images, encode_texts, load_model


def text_bank(model, tok):
    bank = {f"tmpl:{k}": encode_texts(model, tok, [t.format(n) for n in READABLE])
            for k, t in PROMPT_TEMPLATES.items()}
    bank["physics:aerial"] = encode_texts(model, tok, [f"an aerial view of {d}." for d in PHYSICS_DESCRIPTIONS])
    bank["physics:sar"] = encode_texts(model, tok, [f"a SAR image of {d}." for d in PHYSICS_DESCRIPTIONS])
    for h, spec in HIERARCHIES.items():
        for k, t in PROMPT_TEMPLATES.items():
            bank[f"{h}:tmpl:{k}"] = encode_texts(model, tok, [t.format(n) for n in spec["readable"]])
    return bank


def main(keys):
    X, y, _ = load_raw()
    ranges = db_ranges(X)
    print(f"device={DEVICE}  N={len(y)}  dB ranges={json.dumps(ranges)}", flush=True)
    encoded = {m: encode(X, m, ranges) for m in INPUT_MODES + ["naive_uint8"]}

    for key in keys:
        out_dir = CACHE_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        model, tok, norm, size = load_model(key)
        bs = 256 if "b32" in key else 128
        np.savez(out_dir / "text.npz", **text_bank(model, tok))

        jobs = [(m, 0) for m in INPUT_MODES]
        if "b32" in key:  # TTA rotations and failure-mode ablation: ViT-B/32 models only (compute budget)
            jobs += [("vv", r) for r in ROTATIONS if r] + [("naive_uint8", 0)]
        for mode, rot in jobs:
            f = out_dir / f"img_{mode}_r{rot}.npy"
            if f.exists():
                continue
            t0 = time.time()
            imgs = np.ascontiguousarray(np.rot90(encoded[mode], k=rot, axes=(1, 2)))
            feats = encode_images(model, imgs, norm, size, batch_size=bs)
            np.save(f, feats.astype(np.float16))
            print(f"{key:16s} {mode:12s} rot={rot * 90:3d}  {len(feats)} imgs  {time.time() - t0:6.1f}s", flush=True)
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
