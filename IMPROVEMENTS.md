# Codebase Improvement Ideas

Focused on `vnn_fusion_ho` but covering architecture, data pipeline, training workflow, codebase organization, and the Mac-dev / Linux-train workflow.

---

## 1. Data Pipeline (Biggest Wins)

### 1.1 Pre-compute and cache optical flow to disk
**Current**: `FlowDatasetWrapper` runs Farneback on every `__getitem__` call in the dataloader worker. This means every training epoch recomputes flow for every clip from scratch — on CPU, in Python.

**Problem**: Farneback is slow (~50–200ms/clip). With `num_workers=8` and ~8K UCF101 train clips this is likely the main bottleneck, not the GPU.

**Suggestion**: Add a `preprocess_flow=True` mode to `VideoDataset` (alongside the existing frame-extraction step). Save flow as `.npy` or `.pt` files in a `flow/` subdirectory next to the RGB frames. `FlowDatasetWrapper` then becomes a trivial `np.load`. One-time cost, reused every epoch.

### 1.2 Fragile denormalization before flow computation

The full pipeline for a fusion batch currently is:

1. `VideoDataset.__getitem__` loads raw JPEG frames as `float32`, subtracts `[90, 98, 102]` (BGR channel means), then transposes to `[C, T, H, W]` and returns a normalized tensor.
2. `FlowDatasetWrapper.__getitem__` receives that normalized tensor and needs raw pixel values `[0, 255]` to feed into OpenCV's Farneback algorithm. So it does:
   ```python
   rgb_for_flow[0] = rgb_for_flow[0] + 90.0   # restore B channel
   rgb_for_flow[1] = rgb_for_flow[1] + 98.0   # restore G channel
   rgb_for_flow[2] = rgb_for_flow[2] + 102.0  # restore R channel
   rgb_for_flow = rgb_for_flow.clamp(min=0.0, max=255.0)
   ```
3. `calculate_video_flow` then calls `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` and computes flow.

**Why it's fragile:**

The denorm step is an undocumented coupling between two files. `VideoDataset.normalize()` uses the magic constants `[90, 98, 102]` and `FlowDatasetWrapper` must mirror them exactly. There is nothing enforcing this relationship — if you ever change normalization (switch to ImageNet standard `[0.485, 0.456, 0.406]` × 255, or add per-video normalization), `FlowDatasetWrapper` keeps adding back the old constants, producing corrupted pixel values that still look plausible but generate wrong optical flow. The model would then train on systematically incorrect motion cues with no warning.

The `clamp(0, 255)` compounds this: if a pixel was, say, 5 (close to the channel mean of 90), after subtracting 90 it becomes −85, and adding back 90 gives 5. Fine. But if normalization ever shifts the distribution, a value of −200 clamped to 0 is indistinguishable from a value of −5 clamped to 0, even though they came from completely different raw pixels. The clamp silently flattens regions of the frame, producing areas of artificially zero-motion in the flow.

**Fix**: Pre-computing flow during the preprocessing step (§1.1) eliminates this entirely — flow is computed once from the raw unprocessed frames before any normalization exists.

### 1.3 Missing DataLoader performance flags
All DataLoaders are missing:
```python
pin_memory=True          # faster host→device transfer on CUDA
persistent_workers=True  # avoids worker respawn between epochs (when num_workers>0)
prefetch_factor=2         # default is 2 in PyTorch but worth being explicit
```
`pin_memory` especially matters for GPU training and is a free 5–15% speedup.

### 1.4 `clip_len` hardcoded in `data_factory.py`
`clip_len=16` is hardcoded in three places inside `get_dataloaders()`. It should be a CLI arg (`--clip_len`) so you can quickly experiment with temporal context (8/16/32 frames) without editing source.

### 1.5 `ensure_clip_len` pads with last frame — corrupts optical flow

When a video is shorter than `clip_len=16`, `ensure_clip_len` pads it by repeating the final frame:

```python
pad_frames = np.repeat(buffer[-1:, :, :, :], repeats=pad_count, axis=0)
return np.concatenate([buffer, pad_frames], axis=0)
```

For the RGB stream this is visually harmless — the last frame just freezes. But consider what happens downstream in `FlowDatasetWrapper`:

`calculate_video_flow` iterates frame pairs and computes flow between consecutive frames. For the padding region, frame `T-1` and frame `T` are byte-for-byte identical. Farneback sees zero image gradient difference and returns a zero flow vector across the entire frame. The flow stream for that clip therefore looks like: `[real_flow, real_flow, ..., zeros, zeros, zeros]`. The clip artificially has a "motion stop" at the boundary that doesn't exist in the real video.

This is particularly bad for action recognition because many actions are defined by their motion patterns. A clip of "HandStands" that appears to freeze mid-sequence is a corrupted training example that the model must learn to ignore rather than learn from.

**Alternatives:**
- **Loop padding**: after the last real frame, wrap back to frame 0. Consecutive frames across the wrap boundary will have different pixel values, so flow is non-trivial (though it's semantically wrong — it implies a jump). Better than zeros.
- **Reflection padding**: pad with frames in reverse order (`..., frame_N, frame_{N-1}, frame_{N-2}, ...`). Flow in the reversed segment is the negation of real flow, which is at least motion-like.
- **Better fix**: filter out videos with fewer than `clip_len` real frames during preprocessing and don't include them in the dataset at all. UCF101 rarely has such short clips at 4-frame extraction intervals.

### 1.6 `crop()` can raise a `ValueError` on exact-size videos

The random spatial crop:

```python
height_index = np.random.randint(buffer.shape[1] - crop_size)
width_index  = np.random.randint(buffer.shape[2] - crop_size)
```

`np.random.randint(n)` draws from `[0, n-1]`. When `n = 0`, NumPy raises:
```
ValueError: low >= high
```

This happens when `buffer.shape[1] == crop_size` (112) exactly. Because `resize_height = 128` and `crop_size = 112`, a 16-pixel margin normally exists — `randint(128 - 112) = randint(16)` is fine. However:

- A video whose source resolution is already 112×_ or _×112 will be skipped by the resize-if-different-size check in `process_video`:
  ```python
  if (frame_height != self.resize_height) or (frame_width != self.resize_width):
      frame = cv2.resize(frame, (self.resize_width, self.resize_height))
  ```
  This only resizes if dimensions don't match 128×171. A 112×171 source frame passes through unchanged at height=112. After `load_frames` returns a buffer of shape `[T, 112, 171, 3]`, `height_index = randint(112 - 112) = randint(0)` crashes.

- Similarly, the width crop fails if source width is exactly 171 − wait, crop_size is 112 and resize_width is 171, so width has a 59-pixel margin. But height is the risk.

`center_crop` defensively handles this:
```python
height_index = max(0, (buffer.shape[1] - crop_size) // 2)
```
`crop` needs the same guard:
```python
height_index = np.random.randint(max(1, buffer.shape[1] - crop_size))
width_index  = np.random.randint(max(1, buffer.shape[2] - crop_size))
```

This is a silent intermittent crash — only triggered by specific source videos, which makes it hard to reproduce and debug in a long training run.

### 1.7 No color/photometric augmentation
Only spatial flip + random crop are applied. Adding `ColorJitter` (brightness/contrast/saturation) and random grayscale would improve generalization for RGB, especially since flow handles motion invariantly.

---

## 2. Architecture

### 2.1 Hardcoded `fc_features=12544` in `ClassifierHead`
`fusion_head.py` passes `12544` directly:
```python
self.classifier = ClassifierHead(12544, num_classes)
```
This magic number encodes the assumption that the input to the fusion head is `[B, 192, 2, 14, 14]` after the VolterraBlock3D with stride=2. If you change input resolution, `Q`, channel counts, or add/remove blocks, this silently produces wrong shapes or crashes.

Fix: replace `ClassifierHead`'s `view` with `AdaptiveAvgPool3d((1,1,1))` before the FC, or compute the feature size dynamically with a dummy forward pass at construction time.

### 2.2 Dead `activation=False` parameter
Both `backbone_4block.VNN.forward()` and `fusion_head.VNN_F.forward()` have `def forward(self, x, activation=False)` but `activation` is never used. Either use it (e.g., for returning intermediate features/activations for visualization) or remove it.

### 2.3 Two identical kernels in `backbone_7block` Block 1
`backbone_7block.py` Block 1 uses `kernels=[(3,3,3), (3,3,3), (1,1,1)]` — two identical 3×3×3 kernels. This doubles parameters for Block 1 without adding multi-scale diversity. `backbone_4block` correctly uses `(5,5,5), (3,3,3), (1,1,1)`. Looks like a copy-paste error.

### 2.4 Block numbering in `backbone_7block` is misleading
The blocks are defined as `block2, block3, block4, block6, block7, block5` but called in forward as `block1 → block2 → block3 → block4 → block6 → block7 → block5`. Block 5 is defined last in `__init__` but called last in `forward` too — it just has a confusing number. Consider renaming sequentially.

### 2.5 Scalar gate vs per-channel gate

**What the gate currently does:**

Each `VolterraBlock3D` has:
```python
self.quad_gate = nn.Parameter(torch.ones(1) * 1e-4)
```
In the forward pass:
```python
q = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
if self.gate_quadratic:
    q = self.quad_gate * q   # shape: scalar * [B, out_ch, T, H, W]
out = out + q
```

The `quad_gate` is a single floating-point number. Multiplying `[B, 96, T, H, W]` by a scalar scales every channel, every spatial location, every sample in the batch by the exact same amount. The gate controls the entire quadratic branch as a unit — either all channels get more quadratic contribution or none do.

**What per-channel gates would mean:**

```python
self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)
# in forward:
q = self.quad_gate.view(1, -1, 1, 1, 1) * q   # [1, out_ch, 1, 1, 1] * [B, out_ch, T, H, W]
```

Now each of the `out_ch` output channels has its own independent gate value. Consider what this enables for a block with 96 output channels:

- Channel 7 might learn to detect horizontal edges. Horizontal edges already have a strong linear signal (conv captures them well), so the network may learn `gate[7] → very small` — keep channel 7 mostly linear.
- Channel 23 might detect motion boundaries where two objects cross. This is inherently a quadratic interaction (you need to detect two different features *together*). The network may learn `gate[23] → larger` — let channel 23 rely heavily on the quadratic path.
- Channel 51 might detect texture patterns that benefit from the `a²` energy-detector property of the symmetric cubic. `gate[51]` can grow accordingly.

With a scalar gate, the network must find a single compromise for all 96 channels simultaneously. It can never selectively boost quadratic interactions for channels that benefit from it while suppressing them for channels that don't.

The parameter cost is trivial: 96 floats instead of 1 float per block. The expressiveness gain is substantial, and the initialization at `1e-4` for all channels preserves the stable warmup property (§3.4).

The same argument applies to `cubic_gate`.

### 2.6 No ReLU on linear path
`VolterraBlock3D` computes `out = BN(conv_lin(x))` with no activation. The quadratic path already provides nonlinearity, but the linear path output is unbounded-before-summation. At minimum, adding a ReLU after the block's final sum (before pooling) would match standard ResNet behavior and could help gradient flow.

### 2.7 Fusion is naive concatenation — the quadratic interaction is misaligned

In `vnn_fusion_ho`, the fusion step is:
```python
out_rgb  = self.model_rgb(rgb)    # [B, 96, 2, 14, 14]
out_of   = self.model_of(flow)    # [B, 96, 2, 14, 14]
fused    = torch.cat((out_rgb, out_of), dim=1)  # [B, 192, 2, 14, 14]
return self.model_fuse(fused)
```

Inside `VNN_F.block1`, which is `VolterraBlock3D(192, 256, Q=2, ...)`, the quadratic convolution is:
```python
self.conv_quad = Conv3d(192, 2 * Q * 256, kernel_size=3, ...)
# = Conv3d(192, 1024, 3, ...)
```

This produces a 1024-channel tensor which is split into `left` (channels 0–511) and `right` (channels 512–1023), then multiplied element-wise: `left * right`. Each channel of `left` and `right` is a learned linear combination of all 192 input channels — meaning all 96 RGB feature channels and all 96 flow feature channels are mixed together before multiplication.

**The problem:** For the quadratic interaction to capture a meaningful cross-stream signal like "there's motion (flow feature) in the same location as a particular shape (RGB feature)", the network needs `left` to specialize toward RGB features and `right` toward flow features (or vice versa). But nothing in the architecture encourages this. The conv weights can learn it in principle, but the learning signal is indirect and the gradient must discover this factorization from scratch.

**What explicit cross-stream interaction looks like:**

The most direct Volterra-style cross-stream product is:
```python
cross = out_rgb * out_of   # [B, 96, 2, 14, 14]   element-wise
fused = torch.cat([out_rgb, out_of, cross], dim=1)  # [B, 288, 2, 14, 14]
```

This gives the fusion head both the individual stream features *and* their pixel-wise product — which directly captures "motion AND shape at this location". The cross term is exactly the 2nd-order Volterra interaction between the two streams, computed explicitly rather than hoping the conv learns to extract it.

**Alternatively**, learnable stream weighting before fusion:
```python
alpha = torch.sigmoid(self.stream_weight)   # scalar in (0, 1)
fused = alpha * out_rgb + (1 - alpha) * out_of
```
This is much simpler and is what classical two-stream networks (Simonyan & Zisserman) do at test time. The model learns how much to trust each stream globally.

The current naive concat is not wrong, just the hardest way to get the network to discover the cross-stream interaction that is the whole point of two-stream fusion.

### 2.8 Classifier is rigidly coupled to input spatial and temporal dimensions

**The full shape pipeline for `vnn_fusion_ho` with `clip_len=16` and `crop_size=112`:**

```
Input (each stream): [B, 3, 16, 112, 112]

backbone_4block:
  block1 (MultiKernelBlock, stride=2): [B, 24, 8, 56, 56]
  block2 (VolterraBlock, stride=2):    [B, 32, 4, 28, 28]
  block3 (VolterraBlock, no pool):     [B, 64, 4, 28, 28]
  block4 (VolterraBlock, stride=2):    [B, 96, 2, 14, 14]

After concat (RGB + flow):             [B, 192, 2, 14, 14]

fusion_head VNN_F.block1 (stride=2):   [B, 256, 1, 7, 7]

ClassifierHead.view(-1):               [B, 256 * 1 * 7 * 7] = [B, 12544]
FC:                                    [B, num_classes]
```

The 12544 is exactly `256 × 1 × 7 × 7`. This entire chain breaks silently in any of the following scenarios:

- **Change `clip_len` to 8**: backbone outputs `[B, 96, 1, 14, 14]`. The fusion head's MaxPool3d(2,2) on a tensor with temporal dim=1 produces temporal dim=0 on some PyTorch versions, or dim=1 on others. Either way you get `256 × 0 × 7 × 7 = 0` or `256 × 1 × 7 × 7 = 12544` but the shape is wrong. You'd need to audit every stride.
- **Change `crop_size` to 128**: backbone spatial dimensions become `128/8 = 16` at each dim. Fusion head output is `[B, 256, 1, 8, 8]`. Flatten = `256 × 64 = 16384 ≠ 12544`. Runtime crash at the FC layer.
- **Add another pooling layer**: any extra stride anywhere changes the downstream spatial size and breaks the flatten.

**Fix — `AdaptiveAvgPool3d`:**

Replace the flatten in `ClassifierHead` with:
```python
self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
# in forward:
x = self.pool(x)           # [B, C, 1, 1, 1] regardless of input spatial/temporal dims
x = x.view(x.size(0), -1) # [B, C]
x = self.dropout(x)
return self.fc(x)
```
Now `fc` takes `C` features (just the channel count of the last conv), which is stable across resolutions. The 12544 disappears. You can freely change clip length, crop size, or number of pooling layers without touching the classifier.

The tradeoff: adaptive average pooling discards spatial information (you can't tell *where* in the frame the feature was). For action recognition this is usually fine since the label is global. If spatial localization matters, a learnable spatial pooling (e.g., attention pooling) would be better.

### 2.9 `num_classes` and `pretrained` args on backbone are dead
`backbone_4block.VNN.__init__(self, num_classes, num_ch=3, pretrained=False)` — `num_classes` is unused (the backbone outputs features, not logits) and `pretrained=False` is always False (no pretrained weights exist). These are API leftovers from older code that add confusion.

### 2.10 Clamp range `[-50, 50]` may be too loose
Volterra interactions multiply features: for quadratic, if inputs are ~O(10) after BN, `left*right` could reach O(100), and the clamp at 50 still allows very large values. Given that BN output is typically in `[-3, 3]` range, a tighter clamp like `[-10, 10]` or `[-5, 5]` post-product would cost nothing and might improve stability. Worth ablating.

---

## 3. Training Workflow

### 3.1 Weight stats computed every batch — measurable overhead

The current inner loop:
```python
for batch_idx, (inputs, targets) in pbar:
    ...
    _, w_max = self._get_weight_stats()   # ← called every batch
    pbar.set_postfix({"L": ..., "A": ..., "W": f"{w_max:.2e}"})
```

`_get_weight_stats` does:
```python
for p in self.model.parameters():
    w_sum += p.data.sum().item()
    w_count += p.data.numel()
    w_max = max(w_max, p.data.abs().max().item())
```

For `vnn_fusion_ho` — two backbone streams plus fusion head — the total parameter count is on the order of several million. Each `.item()` call forces a GPU→CPU synchronization, stalling the GPU pipeline while the CPU processes the result. The `.abs().max()` also triggers a full GPU reduction on each parameter tensor.

**Rough cost estimate**: suppose the model has 5M parameters spread across 200 tensors. Each batch calls this 200 times (one per parameter tensor), with each call doing a GPU reduction + sync. On a training run with batch_size=8 and ~1000 batches/epoch, that's 200,000 GPU→CPU syncs per epoch, purely for monitoring.

The data that's expensive to collect — `w_max` — changes slowly during training (weights drift across epochs, not dramatically across individual batches). Seeing it update every batch in the tqdm bar is visually useful but informationally redundant. Nothing changes between batch 5 and batch 6 that would make you act differently.

**Fix**: call `_get_weight_stats` once per epoch (after the epoch loop) rather than once per batch. The per-epoch W&B log already captures this. The tqdm bar can show loss and accuracy without `W`.

If you want intra-epoch monitoring for catching instabilities early, call it every 50 or 100 batches instead, gated on `batch_idx % 50 == 0`.

### 3.2 Scheduler state not saved/restored on resume
When resuming a checkpoint, `ckpt["optimizer"]` is loaded but there's no `ckpt["scheduler"]` saved or loaded. This means the learning rate restarts from its initial-epoch value rather than where it was. For `StepLR(step_size=5)` this could cause a significant LR jump. Save `scheduler.state_dict()` in the checkpoint.

### 3.3 Only `best_model.pth` saved — no `last.pth`
If training crashes mid-epoch or `best_model.pth` gets partially written, the last good checkpoint is gone. Add a `last.pth` save at the end of every epoch (just overwrite). Optionally add `--save_every N` for periodic numbered checkpoints.

### 3.4 No LR warmup — especially problematic for gated polynomial networks

**The initialization state at epoch 0:**

- Gates: `quad_gate = 1e-4`, `cubic_gate = 1e-4`
- Conv weights: Kaiming normal
- BN: weight=1, bias=0

The forward pass of a fresh `VolterraBlock3D`:
```
linear_out  ≈ BN(conv_lin(x))          # full magnitude, ~O(1) after BN
quad_out    ≈ 1e-4 * BN(left*right)    # 10,000× smaller
cubic_out   ≈ 1e-4 * BN(a²*b)         # 10,000× smaller
total_out   ≈ linear_out               # model is effectively linear at initialization
```

The model starts as a nearly pure linear network. The quadratic/cubic contributions are negligible until the gates grow. This is intentional for stability — it prevents the polynomial terms from exploding before the linear path establishes a reasonable feature space.

**What happens without warmup at the start of training:**

At epoch 0, with full LR (e.g., `1e-4`), the gradients are determined almost entirely by the linear path. The linear conv weights update aggressively toward whatever features minimize the loss for a linear model. BN statistics are computed and accumulated.

The problem is that this early aggressive linear-path update may establish a feature space that the quadratic/cubic terms then struggle to complement. The gates start growing (backpropagating through `gate * BN(left*right)`) while the conv weights underneath them are still rapidly changing. This is a moving-target problem: the quadratic path is trying to learn a useful interaction of features that themselves are changing each step.

**What warmup does:**

A linear LR warmup from `lr/10` to `lr` over the first 3–5 epochs means:
- Steps 0–N: small gradients → conv weights make small updates → BN statistics stabilize → a consistent base feature space forms
- After warmup: LR is full → conv weights can now make larger updates, but they're updating *around* an already-stable base
- Simultaneously, the gates have begun growing from `1e-4` toward meaningful values in a stable feature landscape

The quadratic/cubic terms therefore "activate into" an already-well-characterized feature space rather than chasing a moving target.

**Practical implementation**: PyTorch's `LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)` combined with `SequentialLR` would chain it with the existing `StepLR`.

### 3.5 `StepLR(step_size=5, gamma=0.9)` for video is the weakest sensible schedule

**What the current schedule actually does:**

Starting from `lr = 1e-4`:

| Epoch | LR |
|-------|----|
| 0–4   | 1.00e-4 |
| 5–9   | 9.00e-5 |
| 10–14 | 8.10e-5 |
| 20–24 | 6.56e-5 |
| 50    | 3.49e-5 |

The decay is `0.9^(epoch//5)`. After 50 epochs the LR is 35% of the initial value. The decay is gradual enough that the model barely notices the steps — the 10% drop every 5 epochs is within the noise of the loss curve.

The step shape itself is a problem: the LR is constant for 5 epochs, then drops discontinuously. This creates small periodic loss spikes on the epoch of the step, which are meaningless but visible in W&B and can be misinterpreted as instability.

**CosineAnnealingLR (already used for CIFAR):**

`CosineAnnealingLR(optimizer, T_max=50)` decays as:
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
```

For `T_max=50`, `lr_min=0`, `lr_max=1e-4`:

| Epoch | LR |
|-------|----|
| 0     | 1.00e-4 |
| 10    | 9.05e-5 |
| 25    | 5.00e-5 |
| 40    | 9.55e-6 |
| 50    | ~0 |

The smooth monotonic decay means the model is always exploring slightly less aggressively than the previous step — no discontinuities, no spikes.

**OneCycleLR:**

`OneCycleLR` does: linear warmup from `lr/div_factor` to `lr`, then cosine decay to `lr/final_div_factor`. It was designed for fixed-epoch training, which is exactly what this repo does. It consistently outperforms step schedules in the literature and requires knowing `total_steps` upfront — which is `epochs × len(train_loader)`, easily computed before training starts.

**Recommendation**: replace `StepLR` with `CosineAnnealingLR` as the minimal improvement. Combine with warmup (§3.4) for the full benefit.

### 3.6 Weight decay applied to BatchNorm parameters — actively harmful

**What L2 weight decay does:**

Weight decay adds a penalty term to the loss: `L_total = L_task + λ/2 * Σ||w||²`. The optimizer gradient update for parameter `w` becomes:
```
w ← w - lr * (grad_L_task + λ * w)
```
This continuously pushes every parameter toward zero unless the task loss gradient opposes it. For a conv weight, this is regularization: it prevents individual filters from growing arbitrarily large, encouraging the network to spread information across many filters (implicit rank reduction, generalization).

**Why BatchNorm parameters are different:**

BatchNorm has two learnable parameters per channel:
- `γ` (weight): scales the normalized output. `γ = 1.0` at initialization.
- `β` (bias): shifts the normalized output. `β = 0.0` at initialization.

The normalization step produces zero-mean unit-variance activations. `γ` and `β` then re-introduce the learned scale and mean that the layer needs to be expressive. If `γ` is pulled toward zero by weight decay, the activations for that channel are suppressed toward zero — the channel effectively dies. If `β` is pulled toward zero, the learned bias disappears and the channel must keep reinventing it through other means.

**Concrete example for a Volterra block:**

After `BN(volterra_quadratic(...))`, channel 7's output has been normalized to zero-mean, unit-variance. Then `γ[7]` rescales it to whatever magnitude is useful for the next block. Suppose gradient descent has pushed `γ[7]` to 2.5 because that channel is highly informative — its quadratic feature gets amplified. Weight decay continuously applies `γ[7] ← γ[7] - lr * λ * 2.5` every step, pulling it back toward 0. The task gradient then has to fight this decay every step just to maintain the informative amplitude. It's wasted gradient budget.

**Standard practice:**

```python
# Separate BN params (and biases) from weight-decayed params
decay_params    = [p for n, p in model.named_parameters()
                   if 'bn' not in n and 'bias' not in n]
no_decay_params = [p for n, p in model.named_parameters()
                   if 'bn' in n or 'bias' in n]

optimizer = optim.Adam([
    {'params': decay_params,    'weight_decay': args.weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=args.lr)
```

This is standard in ResNet training, Vision Transformers, and virtually all modern architectures.

### 3.7 No test evaluation at end of training
The training loop tracks train/val accuracy but never runs the test set automatically. After the training loop completes, it should run `_run_epoch(final_epoch, "test")` on the best checkpoint and log it.

### 3.8 `torch.compile()` not used
PyTorch 2.0+ `torch.compile()` can give 20–50% speedup on CUDA for models with repeated elementwise ops (exactly what Volterra interactions are). Worth wrapping the model: `model = torch.compile(model)` after construction on CUDA, with a `--no_compile` escape hatch.

### 3.9 Non-finite batch skipping is correctly implemented — no bug

This item exists because the skip logic looks asymmetric on first read. To trace through each case:

**Case A: non-finite input**
```python
if not self._check_finite(inputs, "input", ...): continue
```
`optimizer.zero_grad()` was called at the top of the batch. No forward pass has happened. No gradients exist. `continue` moves to the next batch. `stats["batches"]` does not increment, so the running loss and grad-norm averages exclude this batch. Clean skip.

**Case B: non-finite output or loss**
```python
with autocast(...):
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)

if not self._check_finite(outputs, ...) or not self._check_finite(loss, ...):
    if is_train: self.optimizer.zero_grad(set_to_none=True)
    continue
```
The forward pass ran but produced NaN/Inf somewhere. `zero_grad(set_to_none=True)` discards the computation graph (setting grads to `None` rather than zero, freeing memory). No backward pass happens, no parameter update happens. `stats["batches"]` doesn't increment. Clean skip.

**Case C: valid batch (the normal path)**
Backward runs, `clip_grad_norm_` runs, optimizer steps. `stats["batches"]` increments.

The average `grad_norm = stats["grad_norm"] / stats["batches"]` is therefore computed only over valid batches — it correctly excludes skipped iterations. The skip counter `skipped_stats` separately tracks how many batches were dropped and why, which is logged to the pbar and (implicitly) could be logged to W&B.

The only potential concern: if the skip rate is very high (e.g., 50% of batches are non-finite), `stats["batches"]` significantly underestimates the wall-clock time represented by the epoch, and the reported "average" loss is over a biased subset of the data. But at that point the skip rate itself is the signal that something is wrong, and the `skipped_stats` tracking would surface it.

---

## 4. Codebase Organization

### 4.1 Model classes defined inline in `model_factory.py`
`VideoVNNFusion_HO`, `VideoVNN_HO`, `VideoVNNCubicToggle` etc. are defined as local classes inside `if/elif` blocks inside `get_model()`. They should either be proper module-level classes in `network/video_higher_order/` (since that's where they logically belong) or at minimum be defined at the top of `model_factory.py`. Local class definitions inside conditionals make them invisible to introspection tools, type checkers, and grep.

### 4.2 `video_dataset_refactored.py` is dead code
`dataloaders/video_dataset_refactored.py` exists alongside `dataset.py` but is never imported. It's unclear which is the "real" one. Either delete it or rename the active one clearly.

### 4.3 `network/video/` legacy code still imported
`model_factory.py` still imports from `network.video.vnn_fusion_highQ` and `network.video.vnn_rgb_of_highQ` for the `vnn_rgb` and `vnn_fusion` model variants. These are older non-higher-order models. If they're kept as baselines, that's fine, but if `vnn_rgb_ho` and `vnn_fusion_ho` have superseded them, the old directory could be archived or moved to `network/legacy/`.

### 4.4 `find_max_batch.py` doesn't list `vnn_fusion_ho`
`utils/find_max_batch.py` has a hardcoded `choices=["vnn_simple", "vnn_ortho", "resnet18", "vnn_rgb", "vnn_fusion"]` — none of the `_ho` models are included. The main model being worked on (`vnn_fusion_ho`) can't be probed for max batch size with this tool.

### 4.5 `num_classes` mapping duplicated
The dataset → num_classes mapping exists in two places:
- `train.py`: `ds_map = {"cifar10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51, "ucf10": 10}`
- `find_max_batch.py`: another copy

A single source of truth (e.g., a dict in a `utils/constants.py`) would prevent drift.

### 4.6 `get_1x_lr_params` in `backbone_4block` and `backbone_7block` is trivial
Both backbones have:
```python
def get_1x_lr_params(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p
```
This is just `model.parameters()`. The function exists only for API symmetry with the fusion head's differential LR logic. Either enforce a consistent interface (all backbone modules implement `get_1x_lr_params` and `get_10x_lr_params`) or remove it from the backbones and just call `.parameters()` directly.

### 4.7 `backbone_7block` block numbering bug in docstring
The docstring says:
```
Block 6: Quadratic + Cubic → Pool
Block 7: Quadratic + Cubic (no pool)
Block 5: Quadratic + Cubic → Pool
```
This matches the code but the ordering is confusing: blocks are defined and called as 1→2→3→4→6→7→5. This is a renaming accident and should be fixed (rename to 1→2→3→4→5→6→7 sequentially).

---

## 5. Mac Dev / Linux Train Workflow

### 5.1 No sync mechanism documented
`data/` and `runs/` are gitignored (correctly). But there's no documented way to:
- Push a trained checkpoint from Linux back to Mac for inspection
- Push datasets from Mac to Linux
- Sync W&B offline runs

A `Makefile` or `scripts/sync.sh` with `rsync` commands (parameterized by `$REMOTE_HOST`) would make this much less manual:
```makefile
sync-runs:
    rsync -avz user@server:/path/to/runs/ ./runs/

sync-data-up:
    rsync -avz ./data/ user@server:/path/to/data/
```

### 5.2 No `.env.example`
Seven environment variables control dataset paths (`UCF101_ROOT`, `UCF101_PREPROCESSED`, etc.). New environments (fresh Linux server setup) require knowing all of these by reading `mypath.py`. An `.env.example` file listing them all with placeholder values would make onboarding a new machine take minutes instead of hunting through code.

### 5.3 No `requirements-dev.txt` or lockfile
`requirements.txt` uses `>=` version pins. PyTorch/CUDA version compatibility is notoriously tricky, and `torch>=2.0.0` on a fresh Linux server might install a CUDA-incompatible version. Consider pinning exact versions in a `requirements-lock.txt` or using a `setup.sh` that includes the correct `pip install torch --index-url https://download.pytorch.org/whl/cu118` for your server's CUDA version.

### 5.4 `wandb_mode` default is "online" but Linux server may not have internet
If the training server is behind a firewall or doesn't have wandb credentials configured, the run will fail or hang at `wandb.init()`. The default should be `offline` with a note that you can sync later with `wandb sync runs/<run_name>/wandb/`. Or add an `--no_wandb` flag that replaces wandb with a simple JSON log file.

### 5.5 Preprocessing blocks on first run
`VideoDataset.__init__` will run `preprocess()` on the first call if the preprocessed directory doesn't exist. This takes a long time (minutes to hours for UCF101) and happens silently inside dataset construction. On a new Linux server this would stall the training script without warning. Consider a standalone `python -m dataloaders.dataset --preprocess ucf101` command, or at least print a clear message that preprocessing is running and estimated time.

### 5.6 Run output directory uses timestamp on both machines
`self.run_name = f"{args.model}_{args.dataset}_{timestamp}"` — when you start a run on Linux, the `runs/` directory is named by Linux's timestamp. If you later rsync it to Mac and want to resume, `--resume` must point to the exact timestamp path. Using `--run_name` explicitly (and making it consistent between machines) is safer.

### 5.7 W&B `dir` set to `self.out_dir`
```python
self.wandb.init(dir=self.out_dir, ...)
```
W&B writes its files inside the run output directory, which is good. But if `wandb_mode="offline"`, the `.wandb/` files also go there. When you rsync the run back to Mac and run `wandb sync`, you need the correct relative path. Document this in the workflow.

---

## 6. Minor / Low-hanging Fruit

- **`ensure_clip_len` in `__getitem__` after crop**: crop already returns `clip_len` frames if the video is long enough. `ensure_clip_len` is only hit for short videos. The code path is correct but the call order (crop → ensure) means `ensure_clip_len` gets the already-cropped buffer, which can only be shorter than `clip_len` if the video was shorter to begin with. This is fine, just worth a comment.

- **`check_preprocess()` only checks 10 classes**: The integrity check loops over at most `ii == 10` classes. A dataset with corrupt frames in class 11 would pass the check silently.

- **`process_video` EXTRACT_FREQUENCY fallback is three nested ifs**: Could be `max(1, 4 - (3 - min(3, frame_count // 16)))` or a simple loop, but the current triple-nested if has an off-by-one risk if `frame_count` is very small.

- **`wandb.config.update` called after `wandb.init` with `config=vars(args)`**: Both set config. `total_params` (the extra update) could just be added to `vars(args)` before init, making init the single point of config setup.

- **`backbone_4block.py` module `__main__` block**: The smoke test creates a model with `num_classes=101` but `num_classes` is unused. The smoke test should test the actual output shape pipeline including the fusion head, since that's the bottleneck for the magic `12544`.
