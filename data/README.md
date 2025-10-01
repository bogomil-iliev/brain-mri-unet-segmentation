# Data (not included)

This project uses **BraTS 2020** (training/validation) (Link -> https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/data). Download per the dataset license and place like:

BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii
    BraTS20_Training_001_t1ce.nii
    BraTS20_Training_001_t1.nii
    BraTS20_Training_001_t2.nii
    BraTS20_Training_001_seg.nii
  BraTS20_Training_002/
  ...

Optional index CSV (one row per case):
```csv
case_id,folder
BraTS20_Training_001,BraTS2020_TrainingData/.../BraTS20_Training_001
