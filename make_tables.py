"""Step 4: LaTeX tables for the paper, generated from results/*.json (no hand transcription)."""
import json

from sarzs.config import CLASSES, HIERARCHIES, PROMPT_CATEGORY, PROMPT_TEMPLATES, RESULTS_DIR

R = json.loads((RESULTS_DIR / "results.json").read_text())
PC = json.loads((RESULTS_DIR / "positive_control.json").read_text()) \
    if (RESULTS_DIR / "positive_control.json").exists() else {}
OUT = RESULTS_DIR / "tables"
MODEL_ORDER = [k for k in ["clip_b32", "clip_l14", "remoteclip_b32", "remoteclip_l14", "georsclip_b32"] if k in R["models"]]
NAME = R["models"]
MODE = {"vv": "VV", "vh": "VH", "dual": "Dual-pol"}
TECH = {"baseline": "Baseline (single dev-selected template)", "ensemble": "Prompt ensembling (10 templates)",
        "physics": "Physics-informed class descriptions", "tta": "Rotation TTA (0/90/180/270\\textdegree)",
        "ensemble+tta": "Ensembling + TTA", "physics+ensemble+tta": "Physics + ensembling + TTA"}


def p(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(r, key="accuracy_ci"):
    lo, hi = r[key]
    return f"[{p(lo)}, {p(hi)}]"


def tab(name, caption, label, colspec, header, body, size="\\footnotesize"):
    s = (f"\\begin{{table}}[t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
         f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}\n\\toprule\n{header} \\\\\n"
         f"\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    (OUT / f"{name}.tex").write_text(s)


def quote(s):
    return "``" + esc(s) + "''" if s else "mean of all 10 templates"


def esc(s):
    return s.replace("{}", "\\{class\\}").replace("_", "\\_")


def t_dataset():
    d = R["dataset"]
    h3 = HIERARCHIES["H3"]
    grp = {i: h3["names"][g] for g, mem in enumerate(h3["groups"]) for i in mem}
    h5 = HIERARCHIES["H5"]
    grp5 = {i: h5["names"][g] for g, mem in enumerate(h5["groups"]) for i in mem}
    body = [f"{c} & {grp[i]} & {grp5[i]} & {d['counts_total'][i]:,} & {d['counts_dev'][i]:,} & {d['counts_test'][i]:,} \\\\"
            for i, c in enumerate(CLASSES)]
    body += ["\\midrule", f"Total (10 classes) & & & {sum(d['counts_total']):,} & {sum(d['counts_dev']):,} & "
             f"{sum(d['counts_test']):,} \\\\"]
    tab("t_dataset", "EuroSAT-SAR classes, the two coarse hierarchies (H3: 3 groups; H5: 5 groups) and the "
        "stratified dev/test split (seed 0). Dev is used only to select prompt templates/configurations; "
        "all reported results are computed on test.", "tab:dataset", "lllrrr",
        "Fine class & H3 group & H5 group & Total & Dev & Test", body)


def t_prompts():
    rows = [r for r in R["prompts"] if r["model"] == "clip_b32"]
    b = R["baselines"]["fine"]
    body = [f"{r['category']} & {esc(r['template'])} & {quote(r['prompt'])} & {p(r['accuracy'])} {ci(r)} & {p(r['balanced_accuracy'])} & "
            f"{p(r['macro_f1'])} & {r['n_predicted_classes']} \\\\" for r in rows]
    body += ["\\midrule",
             f"\\multicolumn{{3}}{{l}}{{Majority class (always ``{CLASSES[b['majority']['majority_class']]}'')}} & "
             f"{p(b['majority']['accuracy'])} & {p(b['majority']['balanced_accuracy'])} & {p(b['majority']['macro_f1'])} & 1 \\\\",
             f"\\multicolumn{{3}}{{l}}{{Uniform random guess (expected)}} & {p(b['uniform_random_expected_accuracy'])} & "
             f"{p(0.1)} & $\\approx${p(0.1)} & 10 \\\\"]
    tab("t_prompts", "Prompt templates: zero-shot results of CLIP ViT-B/32 with VV input on the test split "
        "(21,600 images, 10 classes). Acc.\\ = overall (micro) accuracy with 95\\% bootstrap CI; Bal.\\ = balanced "
        "accuracy (macro-averaged recall); F1 = macro-F1; \\#Pred = number of distinct classes ever predicted.",
        "tab:prompts", "lllcccc", "Category & ID & Template & Acc. [95\\% CI] & Bal. & F1 & \\#Pred", body, "\\scriptsize")


def t_models():
    body = []
    rows = {(r["model"], r["mode"]): r for r in R["model_mode"]}
    for k in MODEL_ORDER:
        for i, m in enumerate(["vv", "vh", "dual"]):
            r = rows[(k, m)]
            body.append(f"{NAME[k] if i == 0 else ''} & {MODE[m]} & {esc(r['template'])} & {p(r['accuracy'])} {ci(r)} & "
                        f"{p(r['balanced_accuracy'])} & {p(r['macro_precision'])} & {p(r['macro_recall'])} & "
                        f"{p(r['macro_f1'])} \\\\")
        body.append("\\midrule")
    b = R["baselines"]["fine"]["majority"]
    body.append(f"Majority class & -- & -- & {p(b['accuracy'])} & {p(b['balanced_accuracy'])} & {p(b['macro_precision'])} & "
                f"{p(b['macro_recall'])} & {p(b['macro_f1'])} \\\\")
    tab("t_models", "Zero-shot results per model and SAR input encoding on the test split (10 classes). For each "
        "model and input the single prompt template with the highest dev macro-F1 is used. Macro-P/R/F1 are "
        "unweighted means over the 10 classes.", "tab:models", "lllccccc",
        "Model & Input & Template & Acc. [95\\% CI] & Bal. & Macro-P & Macro-R & Macro-F1", body, "\\scriptsize")


def t_techniques():
    rows = {(r["model"], r["technique"]): r for r in R["techniques"]}
    body = []
    for t, lab in TECH.items():
        cells = " & ".join(f"{p(rows[(k, t)]['accuracy'])} / {p(rows[(k, t)]['macro_f1'])}" if (k, t) in rows else "--"
                           for k in MODEL_ORDER)
        body.append(f"{lab} & {cells} \\\\")
    b = R["baselines"]["fine"]["majority"]
    body += ["\\midrule", f"Majority class & " + " & ".join([f"{p(b['accuracy'])} / {p(b['macro_f1'])}"] * len(MODEL_ORDER)) + " \\\\"]
    hdr = "Technique (VV input) & " + " & ".join(NAME[k].replace(" ViT", "\\newline ViT") for k in MODEL_ORDER)
    tab("t_techniques", "Zero-shot techniques (VV input, 10 classes, test split). Each cell gives accuracy / "
        "macro-F1 (\\%). TTA was run for the ViT-B/32 models only (--: not run).", "tab:techniques", "l" + "c" * len(MODEL_ORDER),
        "Technique (VV input) & " + " & ".join(NAME[k] for k in MODEL_ORDER), body, "\\scriptsize")


def t_hier():
    body = []
    for h in ["H3", "H5"]:
        rows = {(r["model"], r["method"]): r for r in R["hierarchical"] if r["hierarchy"] == h}
        nc = len(HIERARCHIES[h]["names"])
        b = R["baselines"][h]["majority"]
        body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{h}: {nc} coarse groups ({', '.join(HIERARCHIES[h]['names'])})}}}} \\\\")
        body.append(f"Majority class (always ``{HIERARCHIES[h]['names'][b['majority_class']]}'') & -- & "
                    f"{p(b['accuracy'])} & {p(b['balanced_accuracy'])} & {p(b['macro_f1'])} & 1 \\\\")
        body.append(f"Uniform random guess (expected) & -- & {p(1 / nc)} & {p(1 / nc)} & -- & {nc} \\\\")
        for k in MODEL_ORDER:
            for meth, lab in [("direct coarse prompts", "direct"), ("fine prediction mapped to coarse", "fine$\\to$coarse")]:
                r = rows[(k, meth)]
                body.append(f"{NAME[k]}, {lab} & {esc(r['template'])} & {p(r['accuracy'])} & {p(r['balanced_accuracy'])} & "
                            f"{p(r['macro_f1'])} & {r['n_predicted_classes']} \\\\")
        if h == "H3":
            body.append("\\midrule")
    tab("t_hier", "Hierarchical (coarse) classification on the test split, VV input, compared with the "
        "majority-class baseline. ``direct'': coarse group names are prompted directly; ``fine$\\to$coarse'': "
        "the 10-class prediction is mapped to its group. Acc./Bal./F1 in \\%.", "tab:hier", "llcccc",
        "Method & Template & Acc. & Bal. & Macro-F1 & \\#Pred", body, "\\scriptsize")


def t_perclass():
    c = R["clip_b32_vv"]["per_class"]
    bc = R["best_config"]
    b = bc["per_class"]
    body = [f"{x['class']} & {x['support']:,} & {p(x['precision'])} & {p(x['recall'])} & {p(x['f1'])} & "
            f"{p(y['precision'])} & {p(y['recall'])} & {p(y['f1'])} \\\\" for x, y in zip(c, b)]
    tab("t_perclass", f"Per-class precision, recall and F1 (\\%) on the test split for CLIP ViT-B/32 with VV input "
        f"(template ``{esc(PROMPT_TEMPLATES[R['clip_b32_vv']['template']])}'') and for the best zero-shot configuration "
        f"selected on dev ({NAME[bc['model']]}, {MODE[bc['mode']]} input{', TTA' if bc['tta'] else ''}, "
        f"template: {esc(bc['template'])}).", "tab:perclass", "lrcccccc",
        "& & \\multicolumn{3}{c}{CLIP ViT-B/32, VV} & \\multicolumn{3}{c}{Best zero-shot} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nClass & Support & P & R & F1 & P & R & F1", body, "\\scriptsize")


def t_gap():
    rows = {(r["model"], r["mode"]): r for r in R["linear_probe"]}
    zs = {(r["model"], r["mode"]): r for r in R["model_mode"]}
    body = []
    for k in MODEL_ORDER:
        opt = PC.get(f"{k}|optical_rgb|aerial")
        gray = PC.get(f"{k}|optical_gray|aerial")
        sar_aer = next(r for r in R["prompts"] if r["model"] == k and r["template"] == "aerial")
        body.append(f"{NAME[k]} & {p(opt['accuracy']) if opt else '--'} & {p(gray['accuracy']) if gray else '--'} & "
                    f"{p(sar_aer['accuracy'])} & {p(zs[(k, 'dual')]['accuracy'])} & "
                    f"{p(rows[(k, 'vv')]['accuracy'])} & {p(rows[(k, 'dual')]['accuracy'])} \\\\")
    tab("t_gap", "Modality gap and information content of the frozen embeddings (accuracy, \\%, 10 classes). "
        "Zero-shot columns use the template ``an aerial view of \\{class\\}.'' (optical, VV) or the dev-selected "
        "template (dual-pol). Optical results use the 2,700-image EuroSAT (Sentinel-2) test split (run for the ViT-B/32 models only; --: not run); SAR results use "
        "the EuroSAT-SAR test split. The linear probe (logistic regression on the 5,400 labelled dev images) is "
        "\\emph{not} zero-shot and is reported only as an upper reference.", "tab:gap", "lcccccc",
        "& \\multicolumn{4}{c}{Zero-shot} & \\multicolumn{2}{c}{Linear probe} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\nModel & Optical RGB & Optical gray & SAR VV & SAR dual & SAR VV & SAR dual",
        body, "\\scriptsize")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in [t_dataset, t_prompts, t_models, t_techniques, t_hier, t_perclass, t_gap]:
        f()
    print("tables ->", OUT)
