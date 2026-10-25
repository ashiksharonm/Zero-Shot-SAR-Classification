"""Loading EuroSAT-SAR and encoding dual-pol dB patches as 3-channel images."""
import json

import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

from .config import CACHE_DIR, CLASSES, DATA_DIR, DEV_FRACTION, SEED


def load_raw():
    """Return (X [N,64,64,2] float32 dB with channels (VV, VH), y [N], paths [N]).

    Cached as a single .npz after the first call.
    """
    cache = CACHE_DIR / "eurosat_sar_raw.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["y"], d["paths"]
    paths, labels = [], []
    for k, c in enumerate(CLASSES):
        files = sorted((DATA_DIR / c).glob("*.tif"), key=lambda p: int(p.stem.split("_")[-1]))
        paths += [str(p.relative_to(DATA_DIR)) for p in files]
        labels += [k] * len(files)
    X = np.stack([tifffile.imread(DATA_DIR / p) for p in paths]).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    paths = np.asarray(paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, paths=paths)
    return X, y, paths


def dev_test_split(y):
    """Stratified dev/test split. Dev is used only to select prompts; test for reporting."""
    idx = np.arange(len(y))
    dev, test = train_test_split(idx, test_size=1 - DEV_FRACTION, stratify=y, random_state=SEED)
    return np.sort(dev), np.sort(test)


def db_ranges(X):
    """Label-free, dataset-wide 1st/99th percentile clipping ranges (dB) for VV, VH, VV-VH."""
    cache = CACHE_DIR / "db_ranges.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rng = np.random.default_rng(SEED)
    sub = X[rng.choice(len(X), size=min(len(X), 5000), replace=False)]
    bands = {"vv": sub[..., 0], "vh": sub[..., 1], "ratio": sub[..., 0] - sub[..., 1]}
    r = {k: [float(np.percentile(v, 1)), float(np.percentile(v, 99))] for k, v in bands.items()}
    cache.write_text(json.dumps(r, indent=2))
    return r


def _to_uint8(band, lo, hi):
    return (np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def encode(X, mode, ranges):
    """Map dB patches [N,H,W,2] to uint8 RGB [N,H,W,3].

    vv   : VV replicated in all three channels (single-polarisation setting)
    vh   : VH replicated in all three channels
    dual : standard dual-pol false-colour composite R=VV, G=VH, B=VV-VH (dB ratio)
    """
    vv = _to_uint8(X[..., 0], *ranges["vv"])
    vh = _to_uint8(X[..., 1], *ranges["vh"])
    if mode == "vv":
        return np.repeat(vv[..., None], 3, axis=-1)
    if mode == "vh":
        return np.repeat(vh[..., None], 3, axis=-1)
    if mode == "dual":
        ratio = _to_uint8(X[..., 0] - X[..., 1], *ranges["ratio"])
        return np.stack([vv, vh, ratio], axis=-1)
    if mode == "naive_uint8":
        # Reproduces the failure mode of casting raw dB values straight to 8-bit:
        # negative dB values clip to 0, producing near-black images.
        raw = np.clip(X[..., 0], 0, 255).astype(np.uint8)
        return np.repeat(raw[..., None], 3, axis=-1)
    raise ValueError(mode)
