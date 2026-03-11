# Pneumonia Chest X-ray Augmentation with WGAN-GP and Mini-DDPM

This repository contains the code for a lightweight generative augmentation study on pneumonia chest X-ray images. The project compares **WGAN-GP** and **Mini-DDPM** from both the **generative** and **downstream classification** perspectives.

## Overview

The goal of this project is to generate synthetic **pneumonia** chest X-ray images and test whether they can replace part of the real pneumonia training data without hurting downstream classification performance.

The full pipeline includes:

1. Train **WGAN-GP** on pneumonia images only
2. Train **Mini-DDPM** on pneumonia images only
3. Generate synthetic pneumonia images from the best checkpoint
4. Build balanced downstream experiment datasets
5. Train **ResNet-18** for binary classification:
   - **Normal vs. Pneumonia**
   - **ImageNet-pretrained** or **from scratch**

## Repository Files

- `WGAN_GP.py` — WGAN-GP training and image generation
- `mini_ddpm.py` — Mini-DDPM training and image generation
- `build_datasets.py` — build downstream experiment datasets
- `resnet.py` — ResNet-18 training and evaluation

## Dataset

This project uses the **Kaggle Chest X-Ray Images (Pneumonia)** dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is **not included** in this repository.

Expected folder structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Environment

Recommended environment:

- Python 3.10+
- PyTorch
- torchvision
- tqdm
- scikit-learn
- matplotlib
- pytorch-ignite
- pillow

Install the dependencies manually in your own environment.

## 1. Train WGAN-GP

Example:

```bash
python WGAN_GP.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_wgan \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

## 2. Train Mini-DDPM

Example:

```bash
python mini_ddpm.py \
  --data_dir ./chest_xray/train/PNEUMONIA \
  --out_dir ./output_ddpm \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

## 3. Build Downstream Datasets

Run:

```bash
python build_datasets.py
```

This script creates:

```text
Experiment_Datasets/
├── Exp_A_Baseline/
├── Exp_B_WGAN/
└── Exp_C_DDPM/
```

### Dataset Settings

- **Baseline**: all pneumonia images are real
- **WGAN-GP**: pneumonia class is a mixture of real and WGAN-generated images
- **Mini-DDPM**: pneumonia class is a mixture of real and DDPM-generated images

## 4. Run ResNet-18

Example:

```bash
python resnet.py \
  --train_dir ./Experiment_Datasets/Exp_A_Baseline \
  --test_dir ./chest_xray/test \
  --epochs 15 \
  --batch_size 32 \
  --device cuda
```

## Important Note for ResNet Weights

In `resnet.py`, the backbone weights are changed **manually** inside the code.

Current line:

```python
model = models.resnet18(weights=None)
```

Manual options:

- `weights=None` → train from scratch
- `weights=ResNet18_Weights.IMAGENET1K_V1` → ImageNet pretrained

Please edit this line manually before running the experiment.

## Outputs

Typical outputs include:

- generated pneumonia images
- training logs
- CSV metric files
- ROC data
- downstream experiment datasets

