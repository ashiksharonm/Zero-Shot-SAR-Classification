# Zero-Shot Land Cover Classification of SAR-like Imagery using Vision-Language Models (VLMs)

**Ashik Sharon M.**  
*Manuscript submitted to IEEE Geoscience and Remote Sensing Letters (GRSL)*

---

## 📄 Abstract
This repository contains the code and reproducibility materials for the paper **"Zero-Shot Land Cover Classification of SAR-like Imagery using Vision-Language Models"**. We investigate the capability of Contrastive Language-Image Pre-training (CLIP) models to classify SAR-like (Sentinel-1 mimicking) imagery without any training samples. By analyzing 10 distinct prompt templates and implementing a hierarchical classification strategy, we demonstrate significant improvements over random baselines.

## 🚀 Key Results

Our experiments on the **EuroSAT** dataset (27,000 images) yielded the following findings:

### 1. Zero-Shot Accuracy by Prompt Type
Using the best-performing model (**CLIP ViT-L/14**), we achieved:
- **Best Prompt ("Aerial View"):** **31.5%** Top-1 Accuracy
- **Hierarchy-Aware Accuracy:** **44.26%** (Coarse-level classification)
- **Baseline Improvement:** **3.4x** improvement over random chance (10%).

| Prompt Template | Accuracy (ViT-L/14) |
| :--- | :--- |
| **"an aerial view of [CLASS]"** | **31.5%** |
| "a satellite image of [CLASS]" | 23.8% |
| "radar imagery showing [CLASS]" | 24.6% |
| "grayscale texture pattern of [CLASS]" | 29.1% |

### 2. t-SNE Visualization
The t-SNE projection of CLIP embeddings reveals distinct clustering for semantic categories (e.g., *Forest* vs. *River*), even in the zero-shot SAR domain.

![t-SNE Embeddings](figures/fig1_tsne.png)
*(Figure 1: t-SNE visualization of CLIP embeddings for SAR-like EuroSAT images)*

### 3. Confusion Matrix
Hierarchical grouping significantly reduces misclassification between semantically similar classes (e.g., *Annual Crop* vs. *Permanent Crop*).

![Confusion Matrix](figures/fig2_conf_matrix.png)

---

## 🛠️ Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.x (GPU recommended)
- `clip`, `datasets`, `scikit-learn`

### Installation
```bash
pip install -r requirements.txt
```

### Running the Experiment
To reproduce the full results (27k images):
```bash
python src/main.py
```
*Note: By default `DEMO_MODE=False` runs the full experiment. Toggle to `True` in `main.py` for a quick 10% subset test.*

---

## 📜 Citation
If you use this code or results, please cite our IEEE GRSL paper:

> **Ashik Sharon M.**, *"Zero-Shot Land Cover Classification of SAR-like Imagery using Vision-Language Models"*, IEEE Geoscience and Remote Sensing Letters, 2026. (Under Review)

[**📄 View Full Paper (PDF)**]([Zero_Shot_SAR_Classification.pdf](https://drive.google.com/file/d/1plfSqjQZku4OMYDqDCjTYExJ9pEtIOgH/view))
   - If using the "RGB" version converted to Grayscale, ensure the path in the notebook matches `../data/EuroSAT`.

2. **Environment:**
   - The environment is defined in `requirements.txt`.
   - It requires a GPU-enabled instance for efficient CLIP inference.
3. **Running:**
   - Open `code/sar_zero_shot_clip_v3.ipynb` in JupyterLab.
   - Run all cells to reproduce the Zero-Shot classification results.

## Files
- `code/sar_zero_shot_clip_v3.ipynb`: Main experiment notebook.
