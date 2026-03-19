# Codebase Improvement Ideas

Covers `vnn_fusion_ho` (primary target) and the broader codebase. Items marked **✓ DONE** are confirmed implemented in the current `main` branch.

---

## 1. Data Pipeline

### 1.1 `clip_len` hardcoded — no CLI arg
`data_factory.py` has `clip_len=16` in three places. It should be `--clip_len` so temporal context (8/16/32 frames) can be swept without editing source.

### 1.2 Short-video padding corrupts optical flow
`ensure_clip_len` pads short videos by repeating the final frame:
```python
pad_frames = np.repeat(buffer[-1:], repeats=pad_count, axis=0)
```
For RGB, a frozen last frame is harmless. For flow, a repeated flow frame means "motion stopped here" — an artifact that doesn't exist in the video. Actions defined by their motion pattern are degraded.

**Fix options:**
- Reflection padding (reverse-play the last N real frames; flow negates cleanly)
- Loop padding (wrap back to frame 0; semantically odd but motion-like)
- Filter out videos shorter than `clip_len` during preprocessing (UCF101 rarely has this)

### 1.3 `crop()` crashes on exact-size videos
```python
height_index = np.random.randint(buffer.shape[1] - crop_size)
```
`randint(0)` raises `ValueError: low >= high`. Triggered when source video resolution is exactly 112 (the crop size), which bypasses the resize step in `process_video`. `center_crop` already handles this with `max(0, ...)`. Fix:
```python
height_index = np.random.randint(max(1, buffer.shape[1] - crop_size))
width_index  = np.random.randint(max(1, buffer.shape[2] - crop_size))
```
Silent intermittent crash — only reproducible with specific source videos.

### 1.4 `of_skip` hardcoded to 1 (consecutive frames)
Optical flow is computed between adjacent frames. For fast actions (running, jumping), consecutive-frame displacements may be too small to distinguish motion direction reliably. Exposing `--of_skip` and ablating skip=1,2,4 is low effort and could matter for UCF101 categories like `SoccerJuggling` or `LongJump`.

### 1.5 No saturation/hue augmentation for RGB
Only `ColorJitter(brightness=0.3, contrast=0.3)` is applied. Adding saturation and hue jitter (`saturation=0.2, hue=0.1`) would help generalize across illumination variation. Flow stream is invariant to color so this affects only RGB quality.

---

## 2. Architecture

### 2.1 Hardcoded `fc_features=12544` in `ClassifierHead`
`fusion_head.py` passes `12544` directly:
```python
self.classifier = ClassifierHead(12544, num_classes)
```
This encodes `256 × 1 × 7 × 7` — the exact spatial/temporal output of the fusion block for `clip_len=16` and `crop_size=112`. Any of the following silently breaks this:
- Change `clip_len` to 8 or 32
- Change `crop_size` to 128
- Add or remove a pooling layer

**Fix — `AdaptiveAvgPool3d((1,1,1))`:**
```python
self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
# forward:
x = self.pool(x).view(x.size(0), -1)   # always [B, C]
```
FC input becomes exactly `out_ch` (256) regardless of resolution. The hardcoded `12544` disappears.

### 2.2 No activation on the linear path
`VolterraBlock3D` computes `out = BN(conv_lin(x))` with no activation. The linear output is unbounded before summing with the polynomial terms. Adding `ReLU` after the full summation `out = pool(relu(out))` would match standard ResNet behavior and improve gradient flow through the linear residual.

### 2.3 `VNN_F` fusion head accepts only one block configuration
The fusion head is a single `VolterraBlock3D(num_ch→256, Q=2, Qc=2, stride=2)`. Q and Qc are not exposed as constructor args. If `num_ch=288` (post cross-product fusion), the internal quadratic conv is `288→2·2·256=1024` channels — relatively large. Exposing `Q` and `Qc` as `VNN_F(num_classes, num_ch, Q=2, Qc=2)` would allow ablating fusion head capacity independently of backbone capacity.

### 2.4 Clamp range `[-50, 50]` potentially too wide
After BN, activations are ~O(-3, 3). The quadratic `left * right` product is ~O(9), and the cubic ~O(27). The clamp at ±50 therefore never triggers in normal operation. A tighter clamp like `[-10, 10]` would only engage in genuine instability events (gradient explosions that slip past clipping) and would provide a harder guard. Worth ablating.

### 2.5 No temporal attention before fusion
Both backbone streams pool time via `MaxPool3d(stride=2)` at each block, which gives uniform temporal weighting. A lightweight 1D temporal attention (e.g., squeeze-and-excite over the T axis before the cross-product) could let the model upweight frames where motion and appearance are most discriminative. Minimal parameter cost: `Linear(T, T)` with softmax.

### 2.6 Two streams use identical architecture regardless of modality
The RGB backbone (`num_ch=3`) and flow backbone (`num_ch=2`) share the exact same Q, Qc, and channel counts. Flow is noisier, lower-dynamic-range, and has only 2 input channels. A smaller Q or reduced depth for the flow stream might generalize better and reduce total parameters. Worth ablating.

---

## 3. Training Workflow

### 3.1 Weight stats computed every batch — expensive GPU→CPU syncs
`_get_weight_stats()` iterates over all parameters calling `.abs().max().item()` per tensor — each `.item()` is a GPU→CPU sync. For `vnn_fusion_ho` (~200 parameter tensors), this is ~200 syncs per batch. At 1000 batches/epoch that's 200k syncs per epoch purely for monitoring. The weight max changes slowly (epoch-scale), so per-batch reporting is informationally redundant.

**Fix:** call `_get_weight_stats()` once per epoch (or every 50 batches via `batch_idx % 50 == 0`), not every batch.

### 3.2 `vnn_fusion_ho` and `vnn_rgb_ho` miss differential LR
`train.py` checks `getattr(model, "get_1x_lr_params", None)`. `VideoVNNFusion_HO` and `VideoVNN_HO` in `model_factory.py` do not implement these methods, so all parameters train at 1× LR. The differential 10× multiplier on the final FC — intended to ramp up the classifier relative to the backbone — silently doesn't apply to the two production models. The `VNN_F` fusion head has `get_1x_lr_params` and `get_10x_lr_params` but they're only reached if the wrapper exposes them.

**Fix:** add `get_1x_lr_params` / `get_10x_lr_params` to `VideoVNNFusion_HO` and `VideoVNN_HO`, delegating to `self.model_fuse`/`self.head`.

### 3.3 `vnn_cubic_simple_toggle.get_1x_lr_params` adds 10× params into 1× group
```python
def get_1x_lr_params(self):
    p += list(vnn_fusion_ho.get_1x_lr_params(self.head))
    p += list(vnn_fusion_ho.get_10x_lr_params(self.head))  # ← wrong: added to 1x
    return p
```
The final FC parameters are added to the 1× group. Since `get_10x_lr_params` isn't defined on the model, train.py never creates a 10× group. Result: FC trains at 1× LR, same as backbone.

### 3.4 Scheduler state not saved on checkpoint
`torch.save({"optimizer": ...})` does not include `scheduler.state_dict()`. On resume, the scheduler restarts from epoch 0 — with `StepLR(step_size=5, gamma=0.9)` this means the LR jumps back to its initial value instead of where training left off.

**Fix:**
```python
torch.save({"scheduler": self.scheduler.state_dict(), ...}, path)
# on resume:
self.scheduler.load_state_dict(ckpt["scheduler"])
```

### 3.5 No LR warmup for gated polynomial networks
At epoch 0, gates are at `1e-4` — the model is nearly linear. The linear conv path updates aggressively from step 1 at full LR, establishing a feature space before polynomial terms have activated. The polynomial gates then try to learn on top of a rapidly shifting linear representation.

A 3–5 epoch linear warmup (`LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5)` chained via `SequentialLR` with the existing step/cosine schedule) lets the linear path stabilize first, then the full LR unlocks as the gates start growing. Standard for transformer/gated architectures.

### 3.6 `StepLR(step_size=5, gamma=0.9)` is a weak schedule for video
After 50 epochs, LR reaches `0.9^10 = 0.35` of its initial value — barely decayed. The step shape also creates visible loss spikes on step epochs. `CosineAnnealingLR(T_max=50)` decays smoothly to near-zero by epoch 50, gives faster final convergence, and is already used for CIFAR. Direct drop-in replacement.

### 3.7 Weight decay applied to BatchNorm — suppresses learned scales
The Adam optimizer uses `weight_decay=5e-4` applied to all parameters including BN `γ` and `β`. BN `γ` is the per-channel learned scale after normalization; L2 decay on it continuously pushes it toward zero, opposing the task gradient whenever a channel's useful amplitude is > 0. Standard practice excludes BN and bias parameters:
```python
decay = [p for n, p in model.named_parameters() if 'bn' not in n and 'bias' not in n]
no_decay = [p for n, p in model.named_parameters() if 'bn' in n or 'bias' in n]
optimizer = Adam([
    {'params': decay, 'weight_decay': args.weight_decay},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=args.lr)
```

### 3.8 No test evaluation at end of training
The training loop saves `best_model.pth` but never loads it and runs `_run_epoch(epoch, "test")` on the held-out test set. Final metrics require a manual `--test_only --resume` invocation.

### 3.9 `torch.compile()` unused
PyTorch 2.0+ `torch.compile()` can give 20–50% throughput improvement for element-wise-heavy models. Volterra interactions are exactly this: the quadratic/cubic paths are dominated by element-wise multiplications and reductions. Wrapping the model on CUDA with an `--no_compile` escape hatch is a no-risk speedup:
```python
if args.device == "cuda" and not args.no_compile:
    model = torch.compile(model)
```

---

## 4. Code Quality

### 4.1 Production model classes defined inline in `model_factory.py`
`VideoVNNFusion_HO`, `VideoVNN_HO`, `VideoVNNCubicToggle` are local class definitions inside `if/elif` blocks. They're invisible to grep, static analysis, and type checkers. They should be proper top-level classes in `network/video_higher_order/` (e.g., a `models.py`) or at minimum defined at module level in `model_factory.py`.

### 4.2 Legacy `network/video/` code still imported
`model_factory.py` imports `vnn_fusion_highQ` and `vnn_rgb_of_highQ` from `network/video/`. These GroupNorm-based older models are superseded by `network/video_higher_order/`. If they're kept as baselines, that's fine, but they should be moved to `network/legacy/` or clearly marked deprecated so new contributors don't accidentally use them.

### 4.3 `vnn_rgb_of_complex.py` has debug prints and wrong return value
`network/video/vnn_rgb_of_complex.py` line 240 has `print('x: ', x.shape)` in the forward pass. The inline quadratic loops (manual per-Q iteration) should also be replaced with vectorized `volterra_ops.py` calls. This file appears to return intermediate features rather than logits at line 250.

### 4.4 `num_classes` mapping duplicated in `train.py`
```python
num_classes_map = {"cifar10": 10, "ucf11": 11, "ucf101": 101, ...}
```
This should live in a shared `utils/constants.py` (or in `data_factory.py` alongside the loader construction) so it's a single source of truth if dataset support changes.

### 4.5 `check_preprocess()` only validates 10 classes
The integrity check in `dataset.py` loops over `ii` up to 10 classes:
```python
if ii == 10:
    break
```
A corrupt video in class 11+ passes silently. Remove the limit or at minimum raise it to the full dataset size.

### 4.6 `backbone_4block.py` `__main__` smoke test incomplete
The `__main__` block tests backbone output shape but not the full pipeline (backbone → fusion head → classifier). The hardcoded `12544` in `ClassifierHead` can break silently with shape changes that the backbone test wouldn't catch. The smoke test should run a dummy input through the complete `vnn_fusion_ho` stack.

