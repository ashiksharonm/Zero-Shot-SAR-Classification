"""Positive control: the identical zero-shot pipeline on OPTICAL EuroSAT (Sentinel-2 RGB).

Verifies that near-chance SAR results are not caused by a pipeline bug, and quantifies the
optical -> SAR modality gap. Uses the EuroSAT test split from HuggingFace (tanganke/eurosat).
"""
import io
import json
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from sarzs import metrics as M
from sarzs.config import PROMPT_TEMPLATES, READABLE, RESULTS_DIR
from sarzs.models import encode_images, encode_texts, load_model


def load_optical():
    df = pd.read_parquet(hf_hub_download("tanganke/eurosat", "data/test-00000-of-00001.parquet",
                                         repo_type="dataset"))
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["image"]])
    gray = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in df["image"]])
    return rgb, np.repeat(gray[..., None], 3, -1), df["label"].to_numpy()


def main(keys):
    rgb, gray, y = load_optical()
    out = {}
    for key in keys:
        model, tok, norm, size = load_model(key)
        for name, imgs in [("optical_rgb", rgb), ("optical_gray", gray)]:
            F = encode_images(model, imgs, norm, size)
            for t, tmpl in PROMPT_TEMPLATES.items():
                txt = encode_texts(model, tok, [tmpl.format(c) for c in READABLE])
                r = M.summary(y, (F @ txt.T).argmax(1), 10)
                out[f"{key}|{name}|{t}"] = r
            best = max((v["accuracy"], k) for k, v in out.items() if k.startswith(f"{key}|{name}|"))
            aer = out[f"{key}|{name}|aerial"]
            print(f"{key:16s} {name:13s} aerial acc {100 * aer['accuracy']:5.2f} macroF1 {100 * aer['macro_f1']:5.2f}"
                  f" | best template {best[1].split('|')[-1]} {100 * best[0]:5.2f}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "positive_control.json").write_text(json.dumps({"n": int(len(y)), **out}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:] or ["clip_b32"])
