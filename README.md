
Code release for [Volterra Neural Networks (VNNs)](https://arxiv.org/abs/1910.09616) and [Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6870).

Patent Information: [Volterra Neural Network and Method](https://patents.google.com/patent/US20210279519A1/en?q=(siddharth+roheda)&oq=siddharth+roheda)

# Citation
If you use our work please cite and acknowledge:

@inproceedings{roheda2020conquering,<br />
  title={Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},<br />
  volume={34},<br />
  number={07},<br />
  pages={11948--11956},<br />
  year={2020}<br />
}<br />

@article{roheda2019volterra,<br />
  title={Volterra Neural Networks (VNNs)},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  journal={arXiv preprint arXiv:1910.09616},<br />
  year={2019}<br />
}<br />

---

# Volterra Neural Networks — Extended Implementation

This repository extends the original VNN codebase with a research implementation focused on **video action recognition** (UCF101, HMDB51, UCF10, UCF11) and **image classification** (CIFAR-10). The core idea is replacing standard convolutions with 2nd- and 3rd-order polynomial (Volterra series) interactions to reduce over-parameterization while capturing richer feature interactions.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Models](#models)
- [Architecture](#architecture)
- [Volterra Math](#volterra-math)
- [Training Arguments](#training-arguments)
- [Weights & Biases Logging](#weights--biases-logging)
- [Resuming Training](#resuming-training)
- [Project Structure](#project-structure)

---

## Overview

A standard convolution computes a *linear* weighted sum of input features. Volterra series extend this to polynomial interactions:

```
y = h0 + Σ h1(i)·x(i)              # linear (1st order)
       + Σ h2(i,j)·x(i)·x(j)       # quadratic (2nd order)
       + Σ h3(i,j,k)·x(i)·x(j)·x(k) # cubic (3rd order)
```

Naively computing all cross-terms is intractable (O(C²) for quadratic, O(C³) for cubic). This implementation uses CP-style tensor factorizations to keep the parameter count tractable while preserving the expressive power of higher-order interactions.

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 is required for mixed-precision (AMP) support and `torch.compile` compatibility. For a specific CUDA version (e.g., CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Dataset Setup

Dataset paths are resolved by `mypath.py` in the following priority order:

1. **Environment variable** (preferred for server use)
2. **Repository-local default** under `./data/`

### Environment Variables

```bash
# UCF101
export UCF101_ROOT=/path/to/ucf101          # raw video directory
export UCF101_PREPROCESSED=/path/to/ucf101_pre  # extracted frames (auto-created on first run)

# HMDB51
export HMDB51_ROOT=/path/to/hmdb51
export HMDB51_PREPROCESSED=/path/to/hmdb51_pre

# UCF10 / UCF11
export UCF10_ROOT=/path/to/ucf10
export UCF10_PREPROCESSED=/path/to/ucf10_pre
export UCF11_ROOT=/path/to/ucf11
export UCF11_PREPROCESSED=/path/to/ucf11_pre

# CIFAR-10 (optional — torchvision downloads automatically)
export CIFAR10_ROOT=/path/to/cifar10
```

### Video Directory Structure

Video datasets must follow this layout:

```
ucf101/
  train/
    ApplyEyeMakeup/
      v_ApplyEyeMakeup_g01_c01.avi
      ...
    BaseballPitch/
      ...
  val/
    ...
  test/
    ...
```

Frame extraction runs automatically on the first training call and is cached in the preprocessed directory. Optical flow for fusion models is computed on-the-fly by default (no pre-extraction needed).

---

## Training

All training is handled through the single `train.py` entrypoint. The `--task` flag is inferred automatically from `--dataset` (CIFAR-10 → `cifar`, everything else → `video`).

### CIFAR-10

```bash
# ResNet18-equivalent VNN with Chebyshev orthogonal polynomial features
python3 train.py \
  --dataset cifar10 \
  --model vnn_ortho \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.01
```

### Video — RGB only

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_rgb_ho \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_workers 8
```

### Video — Two-stream fusion (RGB + optical flow)

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_workers 8
```

---

## Evaluation

To run evaluation only (no training) on a saved checkpoint:

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --test_only \
  --resume runs/<run_name>/checkpoints/best_model.pth
```

---

## Models

| Model name | Task | Description |
|---|---|---|
| `vnn_ortho` | CIFAR-10 | ResNet18-like backbone with Chebyshev T2(x) polynomial features and spectral normalization |
| `vnn_simple` | CIFAR-10 | Lightweight VNN without orthogonalization |
| `resnet18` | CIFAR-10 | Standard ResNet18 baseline (adapted for 32×32 input) |
| `vnn_rgb_ho` | Video | 4-block 3D backbone (quadratic + cubic) → fusion classifier head, RGB only |
| `vnn_fusion_ho` | Video | Two-stream: RGB backbone + flow backbone → cross-stream product → fusion head |
| `vnn_complex_ho` | Video | Deeper 7-block backbone, single RGB stream, includes classifier |
| `vnn_cubic_simple_toggle` | Video | 4-block backbone with `--disable_cubic` flag to ablate cubic contribution |
| `vnn_rgb` | Video | Legacy RGB-only model (older backbone) |
| `vnn_fusion` | Video | Legacy two-stream fusion (older backbone) |

---

## Architecture

### Video Model Data Flow

```
Input RGB:  [B, 3, T, H, W]
Input Flow: [B, 2, T, H, W]   ← computed on-the-fly for fusion models
     │                │
     ▼                ▼
 backbone_4block   backbone_4block
     │                │
  [B,96,T/8,H/8,W/8]  [B,96,T/8,H/8,W/8]
     │                │
     └────── × ───────┘   ← element-wise cross-stream product (vnn_fusion_ho)
     │        │       │
   rgb       flow   cross
     └────── cat ────┘
          [B, 288, T/8, H/8, W/8]
               │
          fusion_head
          (VolterraBlock3D → ClassifierHead)
               │
          [B, num_classes]
```

### 4-Block Backbone (`backbone_4block.py`)

```
Block 1: MultiKernelBlock3D  — parallel 5×5×5 + 3×3×3 + 1×1×1 convs, quadratic only, stride=2
         → [B, 24, T/2, H/2, W/2]

Block 2: VolterraBlock3D     — quadratic, stride=2, residual shortcut
         → [B, 32, T/4, H/4, W/4]

Block 3: VolterraBlock3D     — quadratic + symmetric cubic, no pool, residual shortcut
         → [B, 64, T/4, H/4, W/4]

Block 4: VolterraBlock3D     — quadratic + symmetric cubic, stride=2, residual shortcut
         → [B, 96, T/8, H/8, W/8]
```

### VolterraBlock3D

Each block has three parallel paths summed together:

```
x ──┬── conv_lin  ──→ BN ──────────────────────────────────┐
    ├── conv_quad ──→ BN → volterra_quadratic → quad_gate ──┤→ sum → out
    └── conv_cub  ──→ BN → volterra_cubic_sym → cubic_gate ─┘

Residual shortcut: x → 1×1×1 conv (if in_ch ≠ out_ch or stride > 1) → added to out
```

Gates are per-channel scalar parameters initialized at `1e-4`, easing in polynomial contributions gradually during training.

### Fusion Head (`fusion_head.py`)

A single `VolterraBlock3D` followed by `ClassifierHead`:

```
[B, C, T, H, W] → VolterraBlock3D(stride=2) → AdaptiveAvgPool → Dropout → FC → [B, num_classes]
```

The FC layer receives a 10× learning rate multiplier relative to the backbone.

### CIFAR Model (`cifar_ortho/res_vnn_ortho.py`)

ResNet18-equivalent with Chebyshev polynomial second-order features:

```
T2(x, y) = 4xy − 2   (Chebyshev T2 of the first kind)
```

Spectral normalization on conv weights provides Lipschitz stability. Uses SGD + Nesterov + CosineAnnealingLR.

---

## Volterra Math

Each order is factorized using CP (CANDECOMP/PARAFAC) decomposition to keep parameter counts tractable. All ops clamp outputs to `[-50, 50]` before summation for numerical stability.

### 2nd-order (Quadratic)

```
h2(i,j) ≈ Σ_q a_q(i) · b_q(j)

Implementation:
  conv_quad: Conv3d(in_ch, 2·Q·out_ch, ...)
  split into left [B, Q·C, T, H, W] and right [B, Q·C, T, H, W]
  product = left * right   → clamp → sum over Q → [B, C, T, H, W]
```

### 3rd-order Symmetric

```
h3(i,j,k) ≈ Σ_q a_q(i) · a_q(j) · b_q(k)

Implementation:
  conv_cub: Conv3d(in_ch, 2·Qc·out_ch, ...)
  product = a² * b   → clamp → sum over Qc → [B, C, T, H, W]

The a² term is an energy/magnitude detector (always ≥ 0); b modulates sign and scale.
```

### 3rd-order General (available via `cubic_mode='general'`)

```
h3(i,j,k) ≈ Σ_q a_q(i) · b_q(j) · c_q(k)

Implementation:
  conv_cub: Conv3d(in_ch, 3·Qc·out_ch, ...)
  product = a * b * c   → clamp → sum over Qc → [B, C, T, H, W]

More expressive than symmetric but uses 50% more parameters for the cubic path.
```

The `--Q` flag sets the quadratic rank for all blocks (default: 2). Cubic rank (`Qc`) is fixed per-block in the architecture.

---

## Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `cifar10`, `ucf10`, `ucf11`, `ucf101`, `hmdb51` |
| `--model` | required | Model name (see Models table) |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Base learning rate |
| `--momentum` | 0.9 | SGD momentum (CIFAR only) |
| `--weight_decay` | 5e-4 | L2 regularization strength |
| `--label_smoothing` | 0.0 | Cross-entropy label smoothing factor |
| `--Q` | 2 | Quadratic rank (number of CP components) |
| `--disable_cubic` | False | Ablate cubic path (for `vnn_cubic_simple_toggle`) |
| `--num_workers` | 0 | DataLoader worker processes |
| `--device` | auto | `auto`, `cuda`, `mps`, `cpu` |
| `--resume` | None | Path to checkpoint `.pth` to resume from |
| `--run_name` | auto | Override the run directory name (default: `model_dataset_timestamp`) |
| `--test_only` | False | Skip training, run evaluation on the test set only |
| `--wandb_name` | None | W&B run display name |
| `--wandb_mode` | online | `online`, `offline`, `disabled` |
| `--wandb_project` | auto | W&B project name override |
| `--wandb_entity` | None | W&B team or user override |
| `--wandb_on_fail` | abort | `abort` or `offline` — behavior if W&B init fails |

---

## Weights & Biases Logging

W&B is the default logging backend. Login once per machine:

```bash
wandb login
```

Then run training normally — project and run names are auto-generated from model/dataset/timestamp.

**Override run metadata:**
```bash
python3 train.py --dataset ucf101 --model vnn_fusion_ho \
  --wandb_project my-project \
  --wandb_name "experiment-1/fusion-ho" \
  --wandb_entity my-team
```

**Offline mode** (server without internet access):
```bash
python3 train.py --dataset ucf101 --model vnn_fusion_ho --wandb_mode offline

# Sync later:
wandb sync runs/<run_name>/wandb/
```

**Disable W&B entirely:**
```bash
python3 train.py --dataset ucf101 --model vnn_fusion_ho --wandb_mode disabled
```

Logged metrics include: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `weights/mean`, `weights/max`, `lr`.

---

## Resuming Training

Checkpoints are saved to `runs/<run_name>/checkpoints/best_model.pth` (best validation accuracy).

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --epochs 100 \
  --resume runs/vnn_fusion_ho_ucf101_<timestamp>/checkpoints/best_model.pth
```

The optimizer state is restored from the checkpoint. Training continues from where it left off.

---

## Numerical Stability

Several mechanisms guard against the instability inherent in polynomial feature interactions:

- **Output clamping:** All Volterra products are clamped to `[-50, 50]` before the rank-sum reduction.
- **Gate initialization:** Quadratic and cubic gates are initialized at `1e-4`, making the network behave like a pure linear model at the start of training. Gates grow gradually as training proceeds.
- **Gradient clipping:** Gradient norm is clipped to 1.0 each step.
- **Batch skipping:** Batches with non-finite loss or non-finite model outputs are skipped rather than propagating NaN gradients.
- **Mixed precision:** AMP (float16) is used automatically on CUDA; the GradScaler handles underflow/overflow.

---

## Project Structure

```
.
├── train.py                        # Unified training entrypoint (Trainer class)
├── mypath.py                       # Dataset path resolver (env vars + defaults)
├── requirements.txt
├── IMPROVEMENTS.md                 # Known issues and planned improvements
├── REMOTE.md                       # Instructions for remote server testing
│
├── network/
│   ├── cifar/
│   │   └── vnn_cifar.py            # Simple VNN for CIFAR-10
│   ├── cifar_ortho/
│   │   └── res_vnn_ortho.py        # ResNet18-like VNN with Chebyshev T2 + spectral norm
│   └── video/                      # Primary video model family
│       ├── volterra_ops.py         # Math primitives: quadratic, cubic_symmetric, cubic_general
│       ├── blocks.py               # VolterraBlock3D, MultiKernelBlock3D
│       ├── backbone_4block.py      # 4-block backbone → [B, 96, T/8, H/8, W/8]
│       ├── backbone_7block.py      # Deeper 7-block variant
│       ├── backbone_cubic_toggle.py # Ablation backbone (cubic on/off)
│       └── fusion_head.py          # Single VolterraBlock3D + ClassifierHead
│
├── utils/
│   ├── model_factory.py            # Instantiates all models by name string
│   └── data_factory.py             # Dataloaders; wraps VideoDataset with FlowDatasetWrapper
│
├── dataloaders/
│   └── ...                         # VideoDataset, FlowDatasetWrapper, preprocessing
│
├── data/                           # Default dataset root (gitignored)
└── runs/                           # Training outputs: checkpoints, W&B files (gitignored)
```
