# Brain MRI Tumor Segmentation — U-Net (TensorFlow/Keras)

![Python](https://img.shields.io/badge/python-3.10+-informational)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end U-Net pipeline for brain MRI tumor segmentation:
- Clean, scriptable stages: **download → splits → train → evaluate → inspect**.
- Loss/metrics include **Dice** (and CE) for segmentation.  
- Plots and CSV logs to track training & validation.

> Code is organised under `scripts/`:
> - `download_data.py` – fetch data to `data/raw/`. :contentReference[oaicite:0]{index=0}  
> - `data_preparation_splits.py` – create train/val/test splits (and any caching).  
> - `train_unet.py` – train U-Net; saves weights + history. :contentReference[oaicite:1]{index=1}  
> - `predict_and_evaluate.py` – run inference & compute metrics/figures. :contentReference[oaicite:2]{index=2}  
> - `examine_training_metrics.py` – load history CSV, plot curves. :contentReference[oaicite:3]{index=3}  
> - `model_definition.py` – U-Net model (Keras). :contentReference[oaicite:4]{index=4}  
> - `metrics_and_loss.py` – Dice & CE utilities. :contentReference[oaicite:5]{index=5}  
> - `data_analysis.py` – quick EDA utilities.

---

## Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Dataset

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
Mean Dice (val): TBD

Mean Dice (test): TBD

Example predictions and metric tables are saved by predict_and_evaluate.py.

Replace the numbers once you’ve run your final experiment.

