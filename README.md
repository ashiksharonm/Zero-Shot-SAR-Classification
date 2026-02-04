
# Zero-Shot SAR Classification with CLIP

This capsule contains the code for the IEEE GRSL submission "Zero-Shot Land Cover Classification of SAR-like Imagery using Vision-Language Models".

## Instructions

1. **Data:**
   - The EuroSAT dataset should be placed in the `../data` folder.
   - If using the "RGB" version converted to Grayscale, ensure the path in the notebook matches `../data/EuroSAT`.

2. **Environment:**
   - The environment is defined in `requirements.txt`.
   - It requires a GPU-enabled instance for efficient CLIP inference.

3. **Running:**
   - Open `code/sar_zero_shot_clip_v3.ipynb` in JupyterLab.
   - Run all cells to reproduce the Zero-Shot classification results.

## Files
- `code/sar_zero_shot_clip_v3.ipynb`: Main experiment notebook.
