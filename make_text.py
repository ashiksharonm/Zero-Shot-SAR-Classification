"""Step 5: number-bearing sentences for the paper and response letter, generated from results/*.json.

Writes results/results_text.tex (LaTeX macros \\RES... used by the manuscript).
"""
import json

import numpy as np

from sarzs.config import CLASSES, HIERARCHIES, RESULTS_DIR

R = json.loads((RESULTS_DIR / "results.json").read_text())
PC = json.loads((RESULTS_DIR / "positive_control.json").read_text())
NAME = R["models"]
ORDER = [k for k in ["clip_b32", "clip_l14", "remoteclip_b32", "remoteclip_l14", "georsclip_b32"] if k in NAME]
MODE = {"vv": "VV", "vh": "VH", "dual": "dual-pol"}
B = R["baselines"]
MAJ = B["fine"]["majority"]


def p(x, d=1):
    return f"{100 * x:.{d}f}"


def rng(vals, d=1):
    return f"{p(min(vals), d)}--{p(max(vals), d)}\\%"


def mm(k, m):
    return next(r for r in R["model_mode"] if r["model"] == k and r["mode"] == m)


def pval(x):
    if x < 1e-3:
        return "$p<0.001$"
    return f"$p={x:.3f}$"


T = {}
mmrows = R["model_mode"]
best = R["best_config"]
probe = R["linear_probe"]
probe_acc = [r["accuracy"] for r in probe]

T["RESabstract"] = (f"over all {len(mmrows)} model--input combinations, test accuracy ranges from "
                    f"{rng([r['accuracy'] for r in mmrows])} and macro-F1 from {rng([r['macro_f1'] for r in mmrows])}, "
                    f"against {p(MAJ['accuracy'])}\\% accuracy for always predicting the most frequent class and "
                    f"10.0\\% for random guessing; the best configuration selected on the dev split "
                    f"({NAME[best['model']]}, {MODE[best['mode']]} input) reaches {p(best['accuracy'])}\\% accuracy "
                    f"and {p(best['macro_f1'])}\\% macro-F1")
T["RESprobe"] = f"{rng(probe_acc)} accuracy from only 5,400 labeled images"

# ---- prompts (CLIP B/32, VV) ----------------------------------------------------------
pr = [r for r in R["prompts"] if r["model"] == "clip_b32" and r["template"] != "ensemble"]
acc = {r["template"]: r for r in pr}
sar = [acc[t]["accuracy"] for t in ["sar_basic", "sar_backscatter", "sar_texture"]]
topt = max(pr, key=lambda r: r["accuracy"])
lowt = min(pr, key=lambda r: r["accuracy"])
npred = [r["n_predicted_classes"] for r in pr]
T["RESpromptsText"] = (
    f"Accuracy ranges from {p(lowt['accuracy'])}\\% (\\texttt{{{lowt['template'].replace('_', chr(92) + '_')}}}) to "
    f"{p(topt['accuracy'])}\\% (\\texttt{{{topt['template'].replace('_', chr(92) + '_')}}}), a spread of "
    f"{100 * (topt['accuracy'] - lowt['accuracy']):.1f} percentage points around the majority-class baseline of "
    f"{p(MAJ['accuracy'])}\\%. Macro-F1 lies between {rng([r['macro_f1'] for r in pr])}, below the "
    f"$\\approx$10\\% expected from random guessing, and balanced accuracy between "
    f"{rng([r['balanced_accuracy'] for r in pr])}. With 21,600 test images the 95\\% confidence intervals are about "
    f"$\\pm$0.4 points wide, so some differences between templates are statistically detectable, but none is "
    f"practically meaningful: no template brings CLIP above chance level on SAR. The spatial template "
    f"(``an aerial view of'', {p(acc['aerial']['accuracy'])}\\%) and the three SAR-specific templates "
    f"({rng(sar)}) perform alike, so the data do not support a claim that spatial prompts substantially outperform "
    f"radar terminology. Several templates never predict some classes at all "
    f"({min(npred)}--{max(npred)} of 10 classes are ever predicted).")

# ---- models and polarisation ------------------------------------------------------------
mc = R["mcnemar"]
best_vv = max(ORDER, key=lambda k: mm(k, "vv")["macro_f1"])
lines = []
for a, b in [("clip_b32", "remoteclip_b32"), ("clip_l14", "remoteclip_l14")]:
    key = f"{a}_vs_{b}_vv"
    if key in mc:
        ra, rb = mm(a, "vv"), mm(b, "vv")
        lines.append(f"{NAME[b]} versus {NAME[a]}: {p(rb['accuracy'])}\\% vs.\\ {p(ra['accuracy'])}\\% accuracy and "
                     f"{p(rb['macro_f1'])}\\% vs.\\ {p(ra['macro_f1'])}\\% macro-F1 (McNemar {pval(mc[key]['p'])})")
dual_delta = {k: mm(k, "dual")["macro_f1"] - mm(k, "vv")["macro_f1"] for k in ORDER}
up = [k for k in ORDER if dual_delta[k] > 0.01]
down = [k for k in ORDER if dual_delta[k] < -0.01]
l14 = [k for k in ORDER if "l14" in k]
l14_txt = ", ".join(f"{NAME[k]}: {p(mm(k, 'vv')['macro_f1'])}" for k in l14)
T["RESmodelsText"] = (
    f"Table~\\ref{{tab:models}} and Fig.~\\ref{{fig:modes}} compare all models and inputs. Every configuration stays "
    f"close to chance: accuracy {rng([r['accuracy'] for r in mmrows])}, macro-F1 {rng([r['macro_f1'] for r in mmrows])}. "
    f"With VV input the highest macro-F1 is obtained by {NAME[best_vv]} ({p(mm(best_vv, 'vv')['macro_f1'])}\\%). "
    f"Optical remote-sensing pre-training does not reliably help or hurt: " + "; ".join(lines) + ". "
    f"Domain-specific pre-training on optical imagery therefore does not translate into a consistent advantage on SAR; "
    f"the differences are small in either direction. "
    + (f"The larger ViT-L/14 backbones do not change this picture ({l14_txt}\\% macro-F1 with VV). " if l14 else "")
    + f"The effect of polarization depends on the model: the dual-pol composite raises macro-F1 for "
    f"{', '.join(NAME[k] for k in up) or 'no model'} "
    f"({', '.join(f'{100 * dual_delta[k]:+.1f}' for k in up)} points) and lowers it for "
    f"{', '.join(NAME[k] for k in down) or 'no model'} ({', '.join(f'{100 * dual_delta[k]:+.1f}' for k in down)} points). "
    f"A false-color composite therefore gives the image encoder more information but also moves the input further "
    f"from the images these models were trained on.")

# ---- techniques ------------------------------------------------------------------------
tr = {(r["model"], r["technique"]): r for r in R["techniques"]}
parts = []
for t, lab in [("ensemble", "prompt ensembling"), ("physics", "physics-informed descriptions"), ("tta", "rotation TTA"),
               ("physics+ensemble+tta", "the combination of all three")]:
    d = [tr[(k, t)]["accuracy"] - tr[(k, "baseline")]["accuracy"] for k in ORDER if (k, t) in tr]
    parts.append(f"{lab} {min(d) * 100:+.1f} to {max(d) * 100:+.1f}")
allacc = [r["accuracy"] for r in R["techniques"]]
T["REStechText"] = (
    f"Table~\\ref{{tab:techniques}} shows that none of the standard zero-shot techniques closes the gap. Relative to "
    f"the single dev-selected template, accuracy changes by {'; '.join(parts)} percentage points across models, and "
    f"all results remain between {rng(allacc)}. The physics-informed descriptions, which encode expected radar "
    f"behavior (e.g.\\ volume scattering in forests, specular reflection on water), do not help the text encoders, "
    f"which were never trained to associate such descriptions with radar images.")

# ---- hierarchy ---------------------------------------------------------------------------
hs = []
for h in ["H3", "H5"]:
    rows = [r for r in R["hierarchical"] if r["hierarchy"] == h]
    bm = B[h]["majority"]
    nc = len(HIERARCHIES[h]["names"])
    above = [r for r in rows if r["accuracy"] > bm["accuracy"]]
    top = max(rows, key=lambda r: r["accuracy"])
    hs.append(dict(h=h, nc=nc, maj=bm["accuracy"], majname=HIERARCHIES[h]["names"][bm["majority_class"]],
                   top=top, n_above=len(above), n=len(rows),
                   bal=[r["balanced_accuracy"] for r in rows]))
h3, h5 = hs
HMARGIN = {d["h"]: max(r["accuracy"] for r in R["hierarchical"] if r["hierarchy"] == d["h"]) - d["maj"] for d in hs}


def coarse_claim():
    """Standalone clause comparing coarse accuracy with the majority baseline, valid for the actual numbers."""
    def part(m, h):
        if m <= 0:
            return f"stays below the majority-class baseline for {h} (best result {100 * -m:.1f} points below)"
        return f"exceeds the majority-class baseline by at most {100 * m:.1f} points for {h}"
    return f"coarse accuracy {part(HMARGIN['H3'], 'the 3-group hierarchy')} and {part(HMARGIN['H5'], 'the 5-group hierarchy')}"


T["REScoarse"] = None  # filled below once coarse_claim() is defined
METHOD = {"direct coarse prompts": "direct prompts", "fine prediction mapped to coarse": "fine$\\to$coarse"}
T["REShierText"] = (
    f"Coarse accuracies must be read against the majority-class baseline, which is much higher than at the fine level "
    f"because the groups are unequal in size. For H3, always predicting ``{h3['majname']}'' already gives "
    f"{p(h3['maj'])}\\% accuracy; the best zero-shot coarse result is {p(h3['top']['accuracy'])}\\% "
    f"({NAME[h3['top']['model']]}, {METHOD[h3['top']['method']]}), "
    f"and {h3['n_above']} of {h3['n']} configurations exceed the baseline. Balanced accuracy ({rng(h3['bal'])}) "
    f"stays close to the 33.3\\% of random guessing. For H5 the majority baseline is {p(h5['maj'])}\\%, the best "
    f"result {p(h5['top']['accuracy'])}\\%, and {h5['n_above']} of {h5['n']} configurations exceed it. "
    f"The large jump from fine to coarse accuracy is therefore mostly due to having fewer, larger classes rather "
    f"than to better recognition. A classifier that assigns every image to the largest group "
    f"reaches exactly this baseline, so coarse accuracy alone, without the baseline, can be badly misleading.")

T["REScoarse"] = coarse_claim()

# ---- per class -------------------------------------------------------------------------
pc_best = best["per_class"]
f1s = sorted(pc_best, key=lambda x: -x["f1"])
low = [x["class"] for x in pc_best if x["f1"] < 0.05]
cb = R["clip_b32_vv"]["per_class"]
cm = np.array(R["clip_b32_vv"]["confusion"])
col = cm.sum(0)
dom = int(col.argmax())
T["RESperclassText"] = (
    f"Per-class results (Table~\\ref{{tab:perclass}}, Fig.~\\ref{{fig:confusion}}) show how the errors arise. For CLIP "
    f"ViT-B/32 with VV input, {p(col[dom] / col.sum())}\\% of all test images are assigned to ``{CLASSES[dom]}'', and "
    f"recall is low for every class (at most {p(max(x['recall'] for x in cb))}\\%). In the best "
    f"zero-shot configuration the highest F1 scores are obtained for {f1s[0]['class']} ({p(f1s[0]['f1'])}\\%), "
    f"{f1s[1]['class']} ({p(f1s[1]['f1'])}\\%) and {f1s[2]['class']} ({p(f1s[2]['f1'])}\\%)"
    + (f", while {len(low)} classes ({', '.join(low)}) have F1 below 5\\%" if low else "") + ". "
    f"Large, calm water bodies (SeaLake) fill the whole patch with very low, homogeneous backscatter "
    f"(Fig.~\\ref{{fig:examples}}) and are the most distinctive class; narrow rivers occupy only part of a patch and "
    f"are not recognized. Most other classes, including the urban classes with bright double-bounce returns, are "
    f"recognized poorly, which suggests that the text embeddings do not capture these SAR signatures.")

# ---- gap -----------------------------------------------------------------------------
opt = [PC[f"{k}|optical_rgb|aerial"]["accuracy"] for k in ORDER if f"{k}|optical_rgb|aerial" in PC]
gray = [PC[f"{k}|optical_gray|aerial"]["accuracy"] for k in ORDER if f"{k}|optical_gray|aerial" in PC]
sar_aer = [next(r for r in R["prompts"] if r["model"] == k and r["template"] == "aerial")["accuracy"] for k in ORDER]
pv = [r["accuracy"] for r in probe if r["mode"] == "vv"]
pdual = [r["accuracy"] for r in probe if r["mode"] == "dual"]
T["RESgapText"] = (
    f"Table~\\ref{{tab:gap}} locates the failure. As a positive control, the identical pipeline applied to optical "
    f"EuroSAT reaches {rng(opt)} zero-shot accuracy with RGB input and {rng(gray)} with grayscale input "
    f"(template ``an aerial view of''), confirming that the implementation works and that removing color costs "
    f"part of the performance. On SAR the same template gives {rng(sar_aer)}. A logistic-regression linear probe "
    f"trained on the 5,400 dev images (not zero-shot) reaches {rng(pv)} on VV and {rng(pdual)} on dual-pol input, "
    f"far above chance. The frozen image encoders therefore do produce SAR embeddings that separate the land-cover "
    f"classes; what fails is the match between these embeddings and the text embeddings of the class names. SAR "
    f"images look unlike anything the text encoders were aligned with during pre-training, so prompt wording alone "
    f"cannot recover the class information that the image embeddings contain.")

# ---- pitfall -------------------------------------------------------------------------
nv = R["naive_uint8"]
T["RESpitfallText"] = (
    f"SAR products such as EuroSAT-SAR store backscatter in decibels, and almost every value is negative. If the "
    f"floating-point array is cast directly to 8-bit integers (for example by clipping to 0--255 before saving or "
    f"displaying it), every pixel becomes 0 and all images are black. We reproduced this error: the ViT-B/32 models then predict only "
    f"{min(r['n_predicted_classes'] for r in nv)}--{max(r['n_predicted_classes'] for r in nv)} distinct classes, with "
    f"accuracies of {rng([r['accuracy'] for r in nv])} and macro-F1 of {rng([r['macro_f1'] for r in nv])}. Such "
    f"results look stable and are close to the majority-class rate, so they can easily be mistaken for a weak but "
    f"genuine baseline. Checking the number of distinct predicted classes and comparing against a majority-class "
    f"baseline exposes the problem immediately; we recommend both as routine checks for SAR experiments with VLMs.")

T["RESdiscussionText"] = (
    f"Our results do not support the use of current CLIP-style VLMs as zero-shot classifiers for single-date "
    f"Sentinel-1 land cover. Fine-grained zero-shot accuracy is close to chance for every model, input and prompt we "
    f"tested; {coarse_claim()}; and the best configuration "
    f"({p(best['accuracy'])}\\% accuracy) would not be useful in practice. The linear-probe results suggest that the "
    f"more promising direction is to keep the frozen image encoder and learn the class representations from a few "
    f"labeled SAR examples, for instance through prompt learning~\\cite{{zhou2022learning}} or SAR--text "
    f"pre-training. \\textit{{Limitations.}} (i) We evaluate one dataset (EuroSAT-SAR): European sites, $64\\times64$ "
    f"pixel patches at 10\\,m, a single Sentinel-1 acquisition mode and season per patch, and 10 classes inherited from "
    f"optical labels; results may differ for other regions, higher-resolution SAR, time series or other class "
    f"definitions. (ii) The VLMs require three-channel input. Duplicating VV or VH, or building a false-color "
    f"composite, are pragmatic mappings that do not match the statistics of the images the models were trained on; "
    f"this modality mismatch is one of the causes of the poor zero-shot performance. (iii) We test five dual-encoder "
    f"models and hand-written prompts; SAR-specific foundation models, generative VLMs and learned prompts are outside "
    f"the scope of this study. (iv) Rotation TTA and the optical positive control were run for the ViT-B/32 models only.")

T["RESconclusionText"] = (
    f"This study provides a reproducible baseline for zero-shot vision-language classification of real "
    f"dual-polarization Sentinel-1 imagery. Across five frozen VLMs, ten prompt templates, three input encodings and "
    f"four standard zero-shot techniques, fine-grained accuracy on EuroSAT-SAR stays close to chance "
    f"({rng([r['accuracy'] for r in mmrows])}, majority baseline {p(MAJ['accuracy'])}\\%). Coarse hierarchical "
    f"classification is no remedy: {coarse_claim()}. Prompt wording matters little, and optical remote-sensing pre-training does not transfer reliably to SAR. The image embeddings "
    f"nevertheless contain land-cover information (linear probe {rng(probe_acc)}), which identifies image--text "
    f"alignment as the bottleneck. We therefore caution against using general-purpose or optical remote-sensing VLMs "
    f"as zero-shot SAR classifiers, and we see few-shot adaptation and SAR-specific vision-language pre-training as "
    f"the necessary next steps. Finally, we recommend reporting majority-class baselines, class-balanced metrics and "
    f"the number of predicted classes, which together reveal degenerate results such as those caused by "
    f"decibel-to-8-bit conversion.")

out = ["%% Auto-generated by code/make_text.py from code/results/*.json. Do not edit by hand."]
out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in T.items()]
(RESULTS_DIR / "results_text.tex").write_text("\n".join(out) + "\n")
print("wrote", RESULTS_DIR / "results_text.tex", len(T), "macros")
