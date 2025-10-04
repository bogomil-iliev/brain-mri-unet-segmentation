# Brain MRI Tumor Segmentation — U-Net (TensorFlow/Keras)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/bogomil-iliev/brain-mri-unet-segmentation/blob/main/notebooks/brain_mri_unet_segmentation.ipynb)
![Python](https://img.shields.io/badge/python-3.10+-informational)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproduction of a **2D U-Net** for **brain tumour MRI segmentation** on **BraTS 2020** using **FLAIR + T1ce** modalities, generating **100 slices per case** at **128×128** and reporting **Dice / IoU / Accuracy / Precision / Sensitivity / Specificity**.  
Based on my MSc Computer Vision & Deep Learning portfolio. Full report: `docs/ain7302_report.pdf`.

## Highlights
- **Input:** 2 channels (FLAIR, T1ce), 128×128; **100 slices** per case starting at slice **22**.  
- **Splits:** ≈ **Train 68% / Val 20% / Test 12%** (stratified by case IDs).  
- **Training:** TensorFlow/Keras, Adam (1e-3), early stopping & LR on plateau; model checkpointing.  
- **Metrics:** Dice, Mean IoU, Accuracy, Precision, Sensitivity, Specificity.
- **Clean, scriptable stages:** **download → splits → train → evaluate → inspect**. 
- **Plots and CSV logs** to track training & validation.

> Code is organised under `scripts/`:
> - `download_data.py` – fetch data to `data/raw/`. :contentReference[oaicite:0]{index=0}  
> - `data_preparation_splits.py` – create train/val/test splits (and any caching).  
> - `train_unet.py` – train U-Net; saves weights + history. :contentReference[oaicite:1]{index=1}  
> - `predict_and_evaluate.py` – run inference & compute metrics/figures. :contentReference[oaicite:2]{index=2}  
> - `examine_training_metrics.py` – load history CSV, plot curves. :contentReference[oaicite:3]{index=3}  
> - `model_definition.py` – U-Net model (Keras). :contentReference[oaicite:4]{index=4}  
> - `metrics_and_loss.py` – Dice & CE utilities. :contentReference[oaicite:5]{index=5}  
> - `data_analysis.py` – quick EDA utilities.

## Dataset
Use **BraTS 2020** (see dataset license). Place files as described in **data/README.md**.  
> Note: a one-off filename fix may be needed for `BraTS20_Training_355` (see `scripts/data_preparation_splits.py`).
---

## Quickstart
**Colab (recommended):** click the badge above and run:
1) Data Preparation > 2) Data Analysis > 3) Data Splitting > 4) Define the Evaluation Metrics and Loss Function. > 5) Model Definition > 6) Train and Save the Model > 7) Load the Trained Model > 8) Metrics Analysis > 9) Make Predictions > 10) Evaluation

**or**

**Local**

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Dataset
This repo does not include images. See data/README.md.

Put MRI images (and masks) under data/raw/ (e.g. data/raw/images/*.png, data/raw/masks/*.png) or edit the constant paths at the top of the scripts (see “Config blocks” below).
**scripted download:** ```bash python scripts/download_data.py```

### 3) Make splits
```bash python scripts/data_preparation_splits.py ```
This writes split files (and any cache) under data/ (paths/names are set in the script header).

### 4) Train U-Net
```bash python scripts/train_unet.py```
Outputs:

  - models/ (or path you set) → best/last weights

  - logs/ → training_log.csv, loss/F1 (or Dice) plots

### 5) Evaluate & Predict
```bash python scripts/predict_and_evaluate.py```
Produces aggregate metrics (e.g., Dice/IoU), confusion-style summaries for classes if defined, and example predictions.

### 6) Inspect training curves (optional)
```bash python scripts/examine_training_metrics.py```

### Config blocks (important)
These scripts are notebook-style and use constants near the top for paths and hyper-params (e.g. data roots, split paths, image size, batch size, epochs). Open the file and adjust the “CONFIG” section before running:
  - train_unet.py – data dirs, IMG_SIZE, batch size, epochs, output dirs.
  - predict_and_evaluate.py – path to saved model(s) and test split.
  - model_definition.py, metrics_and_loss.py – network and loss/metric choices.

### Repo layout
```graphql
.
├─ scripts/
│  ├─ download_data.py
│  ├─ data_preparation_splits.py
│  ├─ train_unet.py
│  ├─ predict_and_evaluate.py
│  ├─ examine_training_metrics.py
│  ├─ model_definition.py
│  ├─ metrics_and_loss.py
│  └─ data_analysis.py
├─ data/                # created by you/scripts (raw, splits, cache)
├─ models/              # saved weights (by training script)
├─ logs/                # CSV logs and plots
├─ requirements.txt
├─ LICENSE
└─ README.md
```

### Results
**Training Graphs of Reconstructed Model**

<img width="890" height="442" alt="image" src="https://github.com/user-attachments/assets/56127416-e9d2-4ae8-b610-98bfaa823865" />

**Results from the Reconstructed Model**

<img width="273" height="288" alt="image" src="https://github.com/user-attachments/assets/7d151384-a55f-4bb9-9ce4-00101af785e4" />

**Comparison of the Recreated model with the original results reported**

<img width="1271" height="69" alt="image" src="https://github.com/user-attachments/assets/79b3300d-134e-4570-ad9e-acb69c255666" />

**Predictions Made with the Reconstructed Model**

<img width="1475" height="1279" alt="image" src="https://github.com/user-attachments/assets/b68276ba-d5b2-40cb-978f-b6e7d67c0d7e" />

### Ethics & limitations

Research/education only—not a medical device. BraTS 2020 is single-site curated; external generalisation is not guaranteed. Dice is the primary segmentation metric; accuracy is less informative when background dominates.

### License
**MIT**

### Citation
[➡️ Cite this repository](./CITATION.cff)

