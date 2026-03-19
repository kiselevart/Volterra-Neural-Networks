# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a research implementation of **Volterra Neural Networks (VNN)** — polynomial-filtering-based neural networks for image classification (CIFAR-10) and video action recognition (UCF101, HMDB51, UCF10, UCF11). The core idea is replacing standard convolutions with 2nd- and 3rd-order polynomial (Volterra series) interactions to reduce over-parameterization.

## Important Note
A lot of imporvements have been identified in IMPROVEMENTS.md. If a number such as x.x is used it is likely referring to one of them. Make sure to update IMPROVEMENTS.md if you have completed one of these tasks or you identify a new issue.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train CIFAR-10:**
```bash
python3 train.py --task cifar --dataset cifar10 --model vnn_ortho --epochs 50 --batch_size 128 --lr 0.01
```

**Train video (UCF101):**
```bash
python3 train.py --task video --dataset ucf101 --model vnn_fusion_ho --num_workers 8 --batch_size 8 --lr 1e-4
```

**Evaluate only (no training):**
```bash
python3 train.py --task video --dataset ucf101 --model vnn_fusion_ho --test_only --resume runs/.../checkpoints/best_model.pth
```

There is no test suite — validation is done via training runs with different configs.

## Architecture

### Data Flow
1. `train.py` (`Trainer` class) is the single entrypoint for all tasks
2. `utils/model_factory.py` instantiates models by name string
3. `utils/data_factory.py` creates dataloaders; for fusion models it wraps `VideoDataset` with `FlowDatasetWrapper` to compute optical flow on-the-fly
4. `mypath.py` resolves dataset paths — checks environment variables first (`UCF101_ROOT`, etc.), falls back to `./data/`

### Model Hierarchy (video)

The primary video model family lives in `network/video_higher_order/`:

```
volterra_ops.py       — math primitives: volterra_quadratic, volterra_cubic_symmetric, volterra_cubic_general
blocks.py             — VolterraBlock3D (linear + quadratic ± cubic paths + residual + gate), MultiKernelBlock3D
backbone_4block.py    — 4-block 3D backbone (output: [B, 96, T/8, H/8, W/8])
backbone_7block.py    — deeper 7-block variant
fusion_head.py        — single VolterraBlock3D + ClassifierHead; applies 10× LR to final FC layer
```

`vnn_fusion_ho` (the main model) runs two backbone streams (RGB + optical flow), then fuses them through `fusion_head.py`.

### Volterra Math
Each order is factorized to keep parameter counts tractable:
- **Quadratic (2nd-order):** CP factorization → `left · right` (2·Q channels)
- **Cubic symmetric (3rd-order):** tied factors → `a² · b` (2·Q channels)
- **Cubic general (3rd-order):** independent factors → `a · b · c` (3·Q channels)

`--Q` controls quadratic rank; cubic rank (`Qc`) is set per-block. All ops clamp outputs to `[-50, 50]`.

### CIFAR Models

`network/cifar_ortho/res_vnn_ortho.py` uses Chebyshev polynomial T2(x) = 4xy − 2 with spectral normalization for stability. ResNet18-like backbone.

## Key Training Details

- **Optimizer:** Adam for video, SGD+Nesterov for CIFAR
- **Numerical stability:** gradient clipping (norm=1.0), skip batches with non-finite loss/outputs, gate initialization at 1e-4 to ease in polynomial terms
- **W&B logging:** enabled by default; use `--wandb_on_fail offline` for offline mode
- **Device:** auto-detects CUDA → MPS → CPU; override with `--device {cuda|mps|cpu|auto}`
- **Differential LR:** video models apply 10× LR multiplier to the final FC classifier layer

## Dataset Setup

Set environment variables to point to datasets, or place data in `./data/`:
- `UCF101_ROOT`, `UCF101_PREPROCESSED`
- `HMDB51_ROOT`, `HMDB51_PREPROCESSED`
- `CIFAR10_ROOT`

Video datasets expect directory structure: `[train|val|test]/[class_name]/[video_files]`
