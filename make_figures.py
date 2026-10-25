"""Step 3: paper figures from results/results.json and the raw data."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sarzs.config import CLASSES, HIERARCHIES, RESULTS_DIR
from sarzs.data import db_ranges, encode, load_raw

OUT = RESULTS_DIR / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (all-pairs)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
SHORT = ["Annual\ncrop", "Forest", "Herb.\nveg.", "Highway", "Indus-\ntrial", "Pasture",
         "Perm.\ncrop", "Resi-\ndential", "River", "Sea/\nlake"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_examples():
    X, y, _ = load_raw()
    r = db_ranges(X)
    idx = [int(np.flatnonzero(y == k)[7]) for k in range(len(CLASSES))]
    rows = [("VV", encode(X[idx], "vv", r)), ("VH", encode(X[idx], "vh", r)),
            ("Dual-pol\ncomposite", encode(X[idx], "dual", r))]
    fig, ax = plt.subplots(3, 10, figsize=(7.2, 2.6))
    for i, (name, imgs) in enumerate(rows):
        for j in range(10):
            ax[i, j].imshow(imgs[j])
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                ax[i, j].set_title(SHORT[j], fontsize=7, color=INK)
        ax[i, 0].set_ylabel(name, fontsize=7.5, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.08, right=0.995, top=0.86, bottom=0.01)
    fig.savefig(OUT / "fig_examples.png", dpi=300)


def heat(ax, cm, xt, yt, title):
    cm = np.asarray(cm, float)
    rn = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(rn, cmap=SEQ, vmin=0, vmax=100)
    for i in range(rn.shape[0]):
        for j in range(rn.shape[1]):
            if rn[i, j] >= 0.5:
                ax.text(j, i, f"{rn[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if rn[i, j] > 55 else INK)
    ax.set_xticks(range(len(xt)), [t.replace("\n", " ") for t in xt], fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(yt)), yt, fontsize=6.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=8.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_confusion(R):
    c = R["clip_b32_vv"]
    h = HIERARCHIES["H3"]["names"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [2.4, 1]})
    heat(ax[0], c["confusion"], SHORT, [s.replace("\n", " ") for s in SHORT],
         f"(a) 10 fine classes")
    heat(ax[1], c["H3"]["confusion"], h, h, f"(b) 3 coarse groups (H3)")
    fig.text(0.995, 0.01, "Row-normalised (% of each true class). CLIP ViT-B/32, VV input, dev-selected templates, test split.",
             ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_confusion.png", dpi=300)


def fig_modes(R):
    rows = R["model_mode"]
    models = list(R["models"])
    modes = [("vv", "VV"), ("vh", "VH"), ("dual", "Dual-pol (VV, VH, VV−VH)")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = 0.26
    for i, (m, lab) in enumerate(modes):
        vals = [100 * next(r["macro_f1"] for r in rows if r["model"] == k and r["mode"] == m) for k in models]
        x = np.arange(len(models)) + (i - 1) * w
        ax.bar(x, vals, w - 0.02, color=SERIES[i], label=lab, zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=6, color=INK2)
    maj = 100 * R["baselines"]["fine"]["majority"]["macro_f1"]
    ax.axhline(10, color=INK2, lw=1, ls="--", zorder=1, label="uniform random (≈10.0)")
    ax.axhline(maj, color=INK2, lw=1, ls=":", zorder=1, label=f"majority class ({maj:.1f})")
    ax.set_xticks(range(len(models)), [R["models"][k] for k in models], fontsize=7.5)
    ax.set_ylabel("Macro-F1 (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left", handlelength=1.8, columnspacing=1.2)
    ax.set_ylim(0, max(ax.get_ylim()[1], 20) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "fig_modes.png", dpi=300)


def fig_pipeline():
    """Schematic of the evaluated zero-shot pipeline, drawn from a real EuroSAT-SAR patch."""
    from matplotlib.patches import FancyBboxPatch
    X, y, _ = load_raw()
    r = db_ranges(X)
    i = int(np.flatnonzero(y == 8)[7])  # a River patch
    vv, dual = encode(X[[i]], "vv", r)[0], encode(X[[i]], "dual", r)[0]

    fig = plt.figure(figsize=(7.2, 2.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis("off")

    def box(x, y0, w, h, title, lines, fc="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y0 + h - 1.6, title, ha="center", va="top", fontsize=7.2, weight="bold", color=INK)
        for k, s in enumerate(lines):
            ax.text(x + w / 2, y0 + h - 4.6 - 2.6 * k, s, ha="center", va="top", fontsize=6.2, color=INK2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))

    for j, (img, lab) in enumerate([(vv, "VV ×3"), (dual, "dual-pol")]):
        a = fig.add_axes([0.02 + j * 0.08, 0.52, 0.068, 0.068 * 7.2 / 2.35])
        a.imshow(img); a.set_xticks([]); a.set_yticks([])
        a.set_title(lab, fontsize=6, color=INK2, pad=2)
    ax.text(9.2, 15.8, "Sentinel-1, 64×64 px, 10 m\n(EuroSAT-SAR); dB clipped\nto 1–99th pct. → 0–255",
            ha="center", va="top", fontsize=5.6, color=INK2)
    box(21, 18, 21, 13, "Image encoder (frozen)", ["ViT-B/32 or ViT-L/14", "CLIP / RemoteCLIP / GeoRSCLIP",
                                                   "input resized to 224×224"])
    box(21, 1, 21, 13, "Text encoder (frozen)", ["12-layer Transformer", "same VLM as image encoder",
                                                 "one prompt per class"])
    ax.text(9.2, 5.5, "“an aerial view of\n{class name}.”", ha="center", va="center", fontsize=6.4,
            color=INK, style="italic")
    arrow(17.4, 24.5, 20.6, 24.5); arrow(16.5, 5.5, 20.6, 5.5)
    box(47, 18, 15, 13, "Image embedding", ["$z_{img}\\in\\mathbb{R}^d$", "d = 512 (B/32)", "d = 768 (L/14)"])
    box(47, 1, 15, 13, "Class embeddings", ["$z^{(k)}_{text}$, k = 1..K", "optional: prompt", "ensembling"])
    arrow(42.4, 24.5, 46.6, 24.5); arrow(42.4, 7.5, 46.6, 7.5)
    box(67, 9, 15, 15, "Cosine similarity", ["$\\hat{y}=\\arg\\max_k$", "$\\cos(z_{img}, z^{(k)}_{text})$",
                                            "optional: rotation TTA"])
    arrow(62.4, 24.5, 66.6, 19); arrow(62.4, 7.5, 66.6, 13)
    box(86, 9, 13, 15, "Prediction", ["fine: 10 classes", "or coarse:", "3 or 5 groups"])
    arrow(82.4, 16.5, 85.6, 16.5)
    fig.savefig(OUT / "fig_pipeline.png", dpi=300)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads((RESULTS_DIR / "results.json").read_text())
    fig_pipeline()
    fig_examples()
    fig_confusion(R)
    fig_modes(R)
    print("figures ->", OUT)
