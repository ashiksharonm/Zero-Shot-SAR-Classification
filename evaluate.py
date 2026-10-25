"""Step 2: run every experiment from cached embeddings and write results/.

All selection (best prompt template, best configuration) is done on the stratified dev
split (20%); every reported number is computed on the held-out test split (80%).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sarzs import metrics as M
from sarzs.config import (CACHE_DIR, CLASSES, HIERARCHIES, MODELS, PROMPT_CATEGORY, PROMPT_TEMPLATES,
                          RESULTS_DIR, ROTATIONS, SEED)
from sarzs.data import dev_test_split, load_raw

K = len(CLASSES)
TEMPLATES = list(PROMPT_TEMPLATES)


def norm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def predict(img, txt):
    return (img @ txt.T).argmax(1)


def ensemble(text, keys):
    return norm(np.mean([text[k] for k in keys], axis=0))


def coarse_map(h):
    m = np.empty(K, dtype=np.int64)
    for g, members in enumerate(HIERARCHIES[h]["groups"]):
        m[members] = g
    return m


class Model:
    def __init__(self, key):
        self.key, self.name = key, MODELS[key][0]
        d = CACHE_DIR / key
        self.text = dict(np.load(d / "text.npz"))
        self.img = {}
        for f in d.glob("img_*.npy"):
            mode, rot = f.stem[4:].rsplit("_r", 1)
            self.img[(mode, int(rot))] = np.load(f).astype(np.float32)

    def feats(self, mode, tta=False):
        if not tta:
            return self.img[(mode, 0)]
        return norm(np.mean([self.img[(mode, r)] for r in ROTATIONS], axis=0))


def evaluate(y, p, n_classes, ci=True):
    r = M.summary(y, p, n_classes)
    if ci:
        r["accuracy_ci"] = M.bootstrap_ci(y, p, "accuracy")
        r["macro_f1_ci"] = M.bootstrap_ci(y, p, "macro_f1", n_classes=n_classes, n_boot=300)
    return r


def best_template(feats_dev, y_dev, text, prefix="tmpl:", n_classes=K):
    """Select the single template with the highest dev macro-F1."""
    scores = {t: M.summary(y_dev, predict(feats_dev, text[prefix + t]), n_classes)["macro_f1"] for t in TEMPLATES}
    return max(scores, key=scores.get)


def main():
    _, y, _ = load_raw()
    dev, test = dev_test_split(y)
    yd, yt = y[dev], y[test]
    models = {k: Model(k) for k in MODELS if (CACHE_DIR / k / "text.npz").exists()}
    R = {"models": {k: m.name for k, m in models.items()}}

    # ---- dataset description -------------------------------------------------------
    R["dataset"] = {
        "classes": CLASSES,
        "counts_total": np.bincount(y, minlength=K).tolist(),
        "counts_dev": np.bincount(yd, minlength=K).tolist(),
        "counts_test": np.bincount(yt, minlength=K).tolist(),
        "hierarchies": {h: {"names": s["names"], "members": [[CLASSES[i] for i in g] for g in s["groups"]],
                            "counts_test": np.bincount(coarse_map(h)[yt], minlength=len(s["names"])).tolist()}
                        for h, s in HIERARCHIES.items()},
    }
    R["baselines"] = {"fine": M.baselines(yt, K)}
    for h, s in HIERARCHIES.items():
        R["baselines"][h] = M.baselines(coarse_map(h)[yt], len(s["names"]))

    # ---- E1: prompt templates (VV, every model) ------------------------------------
    rows = []
    for k, m in models.items():
        F = m.feats("vv")
        for t in TEMPLATES + ["ensemble"]:
            txt = ensemble(m.text, [f"tmpl:{x}" for x in TEMPLATES]) if t == "ensemble" else m.text[f"tmpl:{t}"]
            r = evaluate(yt, predict(F[test], txt), K, ci=(k == "clip_b32"))
            r["dev_macro_f1"] = M.summary(yd, predict(F[dev], txt), K)["macro_f1"]
            rows.append({"model": k, "template": t, "category": PROMPT_CATEGORY.get(t, "Ensemble"),
                         "prompt": PROMPT_TEMPLATES.get(t, "mean of all 10 templates"), **r})
    R["prompts"] = rows

    # ---- E2/E3: models x polarisation, dev-selected template -------------------------
    rows, preds = [], {}
    for k, m in models.items():
        for mode in ["vv", "vh", "dual"]:
            F = m.feats(mode)
            t = best_template(F[dev], yd, m.text)
            p = predict(F[test], m.text[f"tmpl:{t}"])
            preds[(k, mode)] = p
            rows.append({"model": k, "mode": mode, "template": t, **evaluate(yt, p, K)})
    R["model_mode"] = rows

    R["mcnemar"] = {}
    for a, b in [("clip_b32", "remoteclip_b32"), ("clip_l14", "remoteclip_l14"), ("clip_b32", "georsclip_b32"),
                 ("clip_b32", "clip_l14"), ("remoteclip_b32", "remoteclip_l14")]:
        if a in models and b in models:
            R["mcnemar"][f"{a}_vs_{b}_vv"] = M.mcnemar(yt, preds[(a, "vv")], preds[(b, "vv")])
    for k in models:
        R["mcnemar"][f"{k}_vv_vs_dual"] = M.mcnemar(yt, preds[(k, "vv")], preds[(k, "dual")])

    # ---- E4: zero-shot techniques (VV) -----------------------------------------------
    rows = []
    for k, m in models.items():
        has_tta = all(("vv", r) in m.img for r in ROTATIONS)
        F, Ft = m.feats("vv"), (m.feats("vv", tta=True) if has_tta else None)
        t = best_template(F[dev], yd, m.text)
        ens = ensemble(m.text, [f"tmpl:{x}" for x in TEMPLATES])
        phys = "physics:aerial" if M.summary(yd, predict(F[dev], m.text["physics:aerial"]), K)["macro_f1"] >= \
            M.summary(yd, predict(F[dev], m.text["physics:sar"]), K)["macro_f1"] else "physics:sar"
        configs = {
            "baseline": (F, m.text[f"tmpl:{t}"]),
            "ensemble": (F, ens),
            "physics": (F, m.text[phys]),
            "tta": (Ft, m.text[f"tmpl:{t}"]),
            "ensemble+tta": (Ft, ens),
            "physics+ensemble+tta": (Ft, ensemble(m.text, ["physics:aerial", "physics:sar"] +
                                                  [f"tmpl:{x}" for x in TEMPLATES])),
        }
        for name, (feat, txt) in configs.items():
            if feat is None:
                continue
            rows.append({"model": k, "technique": name, "template": t, **evaluate(yt, predict(feat[test], txt), K)})
    R["techniques"] = rows

    # ---- E5: hierarchical classification ---------------------------------------------
    rows = []
    for k, m in models.items():
        F = m.feats("vv")
        t_fine = best_template(F[dev], yd, m.text)
        p_fine = predict(F[test], m.text[f"tmpl:{t_fine}"])
        for h, s in HIERARCHIES.items():
            cm, nc = coarse_map(h), len(s["names"])
            yc_d, yc_t = cm[yd], cm[yt]
            t_c = best_template(F[dev], yc_d, m.text, prefix=f"{h}:tmpl:", n_classes=nc)
            rows.append({"model": k, "hierarchy": h, "method": "direct coarse prompts", "template": t_c,
                         **evaluate(yc_t, predict(F[test], m.text[f"{h}:tmpl:{t_c}"]), nc)})
            rows.append({"model": k, "hierarchy": h, "method": "fine prediction mapped to coarse",
                         "template": t_fine, **evaluate(yc_t, cm[p_fine], nc)})
    R["hierarchical"] = rows

    # ---- E6: linear-probe reference (NOT zero-shot: trained on the 5,400 dev labels) ---
    rows = []
    for k, m in models.items():
        for mode in ["vv", "dual"]:
            F = m.feats(mode)
            clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED).fit(F[dev], yd)
            rows.append({"model": k, "mode": mode, **evaluate(yt, clf.predict(F[test]), K, ci=False)})
    R["linear_probe"] = rows

    # ---- E7: failure-mode ablation (raw dB cast to uint8) -----------------------------
    rows = []
    for k, m in models.items():
        if ("naive_uint8", 0) in m.img:
            F = m.feats("naive_uint8")
            t = "aerial"
            p = predict(F[test], m.text[f"tmpl:{t}"])
            rows.append({"model": k, "template": t, **evaluate(yt, p, K, ci=False),
                         "predicted_class_counts": np.bincount(p, minlength=K).tolist()})
    R["naive_uint8"] = rows

    # ---- best zero-shot configuration (selected on dev), per-class + confusion --------
    cands = []
    for k, m in models.items():
        for mode in ["vv", "vh", "dual"]:
            for tta in ([False, True] if mode == "vv" and all(("vv", r) in m.img for r in ROTATIONS) else [False]):
                F = m.feats(mode, tta)
                for t in TEMPLATES + ["ensemble"]:
                    txt = ensemble(m.text, [f"tmpl:{x}" for x in TEMPLATES]) if t == "ensemble" else m.text[f"tmpl:{t}"]
                    cands.append((M.summary(yd, predict(F[dev], txt), K)["macro_f1"], k, mode, tta, t, F, txt))
    _, k, mode, tta, t, F, txt = max(cands, key=lambda c: c[0])
    p = predict(F[test], txt)
    R["best_config"] = {"model": k, "mode": mode, "tta": tta, "template": t, **evaluate(yt, p, K),
                        "per_class": M.per_class(yt, p, CLASSES),
                        "confusion": M.confusion(yt, p, K).tolist()}
    # per-class + confusion for the paper's original setting (CLIP ViT-B/32, VV, dev-best template)
    m = models["clip_b32"]
    F = m.feats("vv")
    t = best_template(F[dev], yd, m.text)
    p = predict(F[test], m.text[f"tmpl:{t}"])
    R["clip_b32_vv"] = {"template": t, "per_class": M.per_class(yt, p, CLASSES),
                        "confusion": M.confusion(yt, p, K).tolist()}
    h = "H3"
    cm, nc = coarse_map(h), len(HIERARCHIES[h]["names"])
    t_c = best_template(F[dev], cm[yd], m.text, prefix=f"{h}:tmpl:", n_classes=nc)
    pc = predict(F[test], m.text[f"{h}:tmpl:{t_c}"])
    R["clip_b32_vv"]["H3"] = {"template": t_c, "per_class": M.per_class(cm[yt], pc, HIERARCHIES[h]["names"]),
                              "confusion": M.confusion(cm[yt], pc, nc).tolist()}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(R, indent=1, default=float))
    for name in ["prompts", "model_mode", "techniques", "hierarchical", "linear_probe"]:
        pd.DataFrame(R[name]).to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    print_summary(R)


def pct(x):
    return f"{100 * x:5.2f}"


def print_summary(R):
    b = R["baselines"]
    print("test counts", R["dataset"]["counts_test"])
    print(f"fine majority acc {pct(b['fine']['majority']['accuracy'])}  macroF1 {pct(b['fine']['majority']['macro_f1'])}"
          f"  uniform {pct(b['fine']['uniform_random_expected_accuracy'])}")
    for h in HIERARCHIES:
        print(f"{h} majority acc {pct(b[h]['majority']['accuracy'])}  balacc {pct(b[h]['majority']['balanced_accuracy'])}"
              f"  macroF1 {pct(b[h]['majority']['macro_f1'])}  counts {R['dataset']['hierarchies'][h]['counts_test']}")
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "n_predicted_classes"]
    for name, idx in [("prompts", ["model", "template"]), ("model_mode", ["model", "mode", "template"]),
                      ("techniques", ["model", "technique"]), ("hierarchical", ["model", "hierarchy", "method", "template"]),
                      ("linear_probe", ["model", "mode"]), ("naive_uint8", ["model"])]:
        df = pd.DataFrame(R[name])
        if df.empty:
            continue
        df[cols[:3]] = (df[cols[:3]] * 100).round(2)
        print(f"\n== {name}\n" + df[idx + cols].to_string(index=False))
    bc = R["best_config"]
    print(f"\nbest config: {bc['model']} {bc['mode']} tta={bc['tta']} {bc['template']}  acc {pct(bc['accuracy'])}"
          f"  macroF1 {pct(bc['macro_f1'])}")
    print(json.dumps(R["mcnemar"], indent=1))


if __name__ == "__main__":
    main()
