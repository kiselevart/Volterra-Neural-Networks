# Remote Testing Instructions — `arch-fixes` branch

## Step 1 — Commit and push the branch

On local machine:

```bash
cd /Users/kisel/projects/VNN/Volterra-Neural-Networks

git add network/video_higher_order/backbone_4block.py \
        network/video_higher_order/blocks.py \
        utils/model_factory.py

git commit -m "arch: residual shortcuts, cross-stream product, per-channel gates"

git push -u origin arch-fixes
```

---

## Step 2 — Pull branch on remote server

SSH in, then:

```bash
cd <project_dir>
git fetch origin
git checkout arch-fixes
# If venv is managed by uv:
uv sync   # or: source .venv/bin/activate
```

---

## Step 3 — Run the comparison experiment

Run both models on the same branch so the backbone improvements are shared and only the fusion architecture is the differentiator. Use explicit `--wandb_name` so the two runs are easy to compare in W&B.

**Run A — RGB-only baseline (arch-fixes backbone):**
```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_rgb_ho \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_workers 8 \
  --wandb_name "arch-fixes/rgb-ho" \
  --wandb_mode online
```

**Run B — Two-stream fusion (all three fixes active):**
```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_workers 8 \
  --wandb_name "arch-fixes/fusion-ho" \
  --wandb_mode online
```

Run them in separate tmux/screen panes so they can run in parallel if GPU memory allows; otherwise sequentially.

If W&B is unavailable on the server, add `--wandb_mode offline` and sync later with:
```bash
wandb sync runs/<run_name>/wandb/
```

---

## Step 4 — What to watch for

| Signal | Expected outcome |
|--------|-----------------|
| `val/acc` at epoch 50 | fusion-ho > rgb-ho by ≥ 2% (was ~1% gap in *wrong* direction) |
| `train/loss` early | fusion-ho should converge faster — residual shortcuts ease gradient flow for flow stream |
| `weights/mean` on quad/cubic gates | gates should grow away from 1e-4 faster with per-channel freedom |
| NaN/inf losses | none expected; clamp [-50,50] + gradient clip unchanged |

---

## Files involved

| File | Change |
|------|--------|
| `network/video_higher_order/backbone_4block.py` | Fix 1: use_shortcut=True on blocks 2/3/4 |
| `network/video_higher_order/blocks.py` | Fix 3: per-channel gates |
| `utils/model_factory.py` | Fix 2: cross-stream product, 288ch fusion input |

---

## After the run

Items 2.2 (scalar gates) and 2.4 (naive concatenation) in `IMPROVEMENTS.md` are addressed by this branch. Update them to "done" once the run confirms the improvement.
