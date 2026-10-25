# Zero-Shot Land Cover Classification of SAR Imagery using Vision-Language Models

Code and results for:

> Arivoli A, Ashik Sharon M, Mannadithya. *Zero-Shot Land Cover Classification of SAR Imagery using
> Vision-Language Models: A Prompt Engineering Study.* 5th International Conference on Machine Learning and
> Data Engineering (ICMLDE 2026), Procedia Computer Science (to appear).

We evaluate whether frozen CLIP-style vision-language models can classify real dual-polarization Sentinel-1
SAR patches without any training. Five models (CLIP ViT-B/32 and ViT-L/14, RemoteCLIP ViT-B/32 and ViT-L/14,
GeoRSCLIP ViT-B/32), ten prompt templates, three SAR input encodings, prompt ensembling, physics-informed class
descriptions, rotation test-time augmentation and two class hierarchies are compared on EuroSAT-SAR.

## Main findings

* **Zero-shot accuracy is close to chance** for every configuration (10 classes, test split of 21,600 images).
* **A linear probe on the same frozen features reaches 59–70%**: the image embeddings carry land-cover
  information, but they are poorly aligned with the text embeddings of the class names.
* Prompt wording, ensembling, physics-informed descriptions and TTA change accuracy by only a few points.
* Coarse (hierarchical) accuracy must be read against the majority-class baseline: 50.0% vs. best 51.3% for
  3 groups, 29.6% vs. best 36.4% for 5 groups.
* Casting SAR backscatter in dB directly to 8-bit images makes every pixel 0 and collapses predictions to
  one or two classes. Always check the number of predicted classes and a majority-class baseline.

| Model | Zero-shot VV acc / macro-F1 | Zero-shot dual-pol acc / macro-F1 | Linear probe VV / dual-pol acc |
|---|---|---|---|
| CLIP ViT-B/32 | 9.8 / 6.3 | 18.9 / 11.3 | 59.3 / 63.7 |
| CLIP ViT-L/14 | 12.7 / 10.3 | 22.8 / 14.4 | 62.6 / 69.8 |
| RemoteCLIP ViT-B/32 | 12.0 / 6.1 | 11.5 / 2.7 | 65.0 / 68.8 |
| RemoteCLIP ViT-L/14 | 12.0 / 4.7 | 12.8 / 5.6 | 65.8 / 70.1 |
| GeoRSCLIP ViT-B/32 | 20.8 / 12.8 | 11.6 / 2.9 | 63.7 / 66.6 |
| Majority class | 11.1 / 2.0 | 11.1 / 2.0 | – |

Accuracy and macro-F1 in %, test split, dev-selected prompt template. Full results: `results/`.

## Data

[EuroSAT-SAR](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR) (Wang et al., 2023; MIT licence):
27,000 Sentinel-1 patches (64×64 px, 10 m, VV and VH backscatter in dB) with the 10 EuroSAT land-cover labels.

```bash
mkdir -p data && cd data
curl -L -o EuroSAT-SAR.zip https://huggingface.co/datasets/wangyi111/EuroSAT-SAR/resolve/main/EuroSAT-SAR.zip
unzip EuroSAT-SAR.zip && cd ..
```

## Protocol

* **No training.** One stratified split (seed 0): 20% *dev* (5,400) used only to select prompt templates and
  configurations, 80% *test* (21,600) for every reported number.
* **Preprocessing.** Each band clipped to its dataset-wide 1st–99th percentile in dB (label-free; stored in
  `cache/db_ranges.json`), scaled to 0–255, resized bicubically to 224 px, CLIP-normalized.
* **Inputs.** `vv` and `vh`: one band copied to three channels; `dual`: R=VV, G=VH, B=VV−VH.
* **Models** via `open_clip`: OpenAI CLIP weights, `chendelong/RemoteCLIP`, `Zilun/GeoRSCLIP` (RS5M_ViT-B-32).
  fp16 inference on Apple MPS (embeddings match fp32 to cosine ≥ 0.9999).
* **Metrics.** Accuracy (= micro P/R/F1), balanced accuracy, macro P/R/F1, per-class P/R/F1, 95% bootstrap CIs,
  McNemar tests, majority-class and random baselines. A logistic-regression linear probe trained on the dev
  labels is reported as a non-zero-shot reference.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python extract_features.py   # embeddings -> cache/ (a few hours on an Apple M-series GPU)
.venv/bin/python positive_control.py   # optical EuroSAT control (ViT-B/32 models in the paper)
.venv/bin/python evaluate.py           # all experiments -> results/results.json, *.csv
.venv/bin/python make_tables.py        # LaTeX tables  -> results/tables/
.venv/bin/python make_figures.py       # figures       -> results/figures/
.venv/bin/python make_text.py          # numbers used in the paper text -> results/results_text.tex
```

Rotation TTA and the optical positive control were run for the ViT-B/32 models only.

## Repository history

Earlier commits (tag `original-submission`) contain preliminary code that evaluated grayscale *optical* EuroSAT
images and a README describing results that we could not reproduce. That material is superseded by this
version, which is the code behind the published paper.

## Licence

Code: MIT (see `LICENSE`). Dataset and model weights are subject to their own licences.
