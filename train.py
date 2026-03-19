import argparse
import atexit
import contextlib
import os
import time
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.data_factory import get_dataloaders
from utils.model_factory import get_model

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Core Training Args
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "ucf10", "ucf101", "hmdb51", "ucf11"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--fc_lr_mult", type=float, default=10.0, help="LR multiplier for the final FC layer (video models)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs before cosine decay (video only)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--Q", type=int, default=2)
    parser.add_argument("--disable_cubic", action="store_true")

    # Logging & System
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation on the test set")
    
    args = parser.parse_args()
    
    # Auto-determine Task and Classes
    args.task = "cifar" if args.dataset == "cifar10" else "video"
    ds_map = {"cifar10": 10, "ucf11": 11, "ucf101": 101, "hmdb51": 51, "ucf10": 10}
    args.num_classes = ds_map.get(args.dataset, 101)
    
    return args

class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0
        
        self._setup_device(args.device)
        self._setup_logging(args)
        
        # Initialize Model, Data, and Criterion
        self.model = get_model(args, self.device)
        self.loaders = get_dataloaders(args)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(self.device)
        self._setup_optimizer(args)
        
        # Mixed Precision Setup
        self.amp_enabled = (self.device.type == "cuda")
        self.scaler = GradScaler(device="cuda") if self.amp_enabled else None
        
        # Numerical Stability & Parameter Tracking
        self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}
        self.finite_debug_prints = 0
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.wandb.config.update({"total_params": self.total_params})
        print(f"==> Total Parameters: {self.total_params:,}")

        # Optional Resume
        if args.resume and os.path.isfile(args.resume):
            print(f"==> Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.start_epoch = ckpt["epoch"]
            self.best_acc = ckpt.get("best_acc", 0.0)
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

    def _setup_device(self, device_pref):
        if device_pref == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_pref)
        print(f"==> Device: {self.device}")

    def _setup_logging(self, args):
        import wandb
        self.wandb = wandb

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        self.run_name = args.run_name or f"{args.model}_{args.dataset}_{timestamp}"
        self.out_dir = os.path.join("runs", self.run_name)
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)

        init_kwargs = {
            "name": args.wandb_name or self.run_name,
            "mode": args.wandb_mode,
            "dir": self.out_dir,
            "config": vars(args)
        }

        # Resume the same W&B run if the checkpoint recorded a run id
        if args.resume and os.path.isfile(args.resume):
            try:
                peek = torch.load(args.resume, map_location="cpu")
                wandb_run_id = peek.get("wandb_run_id")
                if wandb_run_id:
                    init_kwargs["id"] = wandb_run_id
                    init_kwargs["resume"] = "must"
            except Exception:
                pass

        self.wandb.init(**init_kwargs)
        atexit.register(lambda: self.wandb.finish() if self.wandb else None)

    def _setup_optimizer(self, args):
        if args.task == "video":
            get_1x = getattr(self.model, "get_1x_lr_params", None)
            get_10x = getattr(self.model, "get_10x_lr_params", None)

            if callable(get_1x):
                params = [{"params": get_1x(), "lr": args.lr}]
                if callable(get_10x):
                    params.append({"params": get_10x(), "lr": args.lr * args.fc_lr_mult})
            else:
                params = self.model.parameters()

            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

            cosine_epochs = max(1, args.epochs - args.warmup_epochs)
            cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cosine_epochs)
            if args.warmup_epochs > 0:
                warmup = optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
                )
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
                )
            else:
                self.scheduler = cosine
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay, nesterov=True)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

    def _get_weight_stats(self):
        """Calculates global mean and max absolute weight across the model."""
        w_sum, w_count, w_max = 0.0, 0, 0.0
        with torch.no_grad():
            for p in self.model.parameters():
                w_sum += p.data.sum().item()
                w_count += p.data.numel()
                w_max = max(w_max, p.data.abs().max().item())
        return w_sum / w_count if w_count > 0 else 0.0, w_max

    def _check_finite(self, tensor, name, epoch, batch_idx, mode):
        tensors = tensor if isinstance(tensor, (list, tuple)) else [tensor]
        if all(torch.isfinite(t).all() for t in tensors):
            return True
            
        self.skipped_stats["total"] += 1
        self.skipped_stats[name] += 1
        
        if self.finite_debug_prints < 3:
            print(f"[{mode.upper()}][Ep {epoch+1}][Batch {batch_idx}] Skipping: non-finite {name}")
            self.finite_debug_prints += 1
        return False

    def _run_epoch(self, epoch, mode="train"):
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[mode if is_train else "val" if mode == "val" else "test"]
        
        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0, "grad_norm": 0.0}
        self.finite_debug_prints = 0
        
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Ep {epoch+1} [{mode.upper()}]")
        
        for batch_idx, (inputs, targets) in pbar:
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, (list, tuple)) else inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)
            
            if not self._check_finite(inputs, "input", epoch, batch_idx, mode): continue
            
            if is_train: self.optimizer.zero_grad()
            
            with (autocast(device_type=self.device.type) if self.amp_enabled else contextlib.nullcontext()):
                with contextlib.nullcontext() if is_train else torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            if not self._check_finite(outputs, "output", epoch, batch_idx, mode) or \
               not self._check_finite(loss, "loss", epoch, batch_idx, mode):
                if is_train: self.optimizer.zero_grad(set_to_none=True)
                continue

            if is_train:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
                    self.optimizer.step()

            stats["loss"] += loss.item()
            stats["batches"] += 1
            stats["total"] += targets.size(0)
            stats["correct"] += outputs.max(1)[1].eq(targets).sum().item()
            
            # Update terminal with current max weight every batch for monitoring
            _, w_max = self._get_weight_stats()
            pbar.set_postfix({
                "L": f"{stats['loss']/stats['batches']:.3f}", 
                "A": f"{100.*stats['correct']/stats['total']:.1f}%", 
                "W": f"{w_max:.2e}"
            })

        res = {k: (v/stats["batches"] if k in ["loss", "grad_norm"] else v) for k, v in stats.items()}
        res["acc"] = 100. * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        return res

    def run(self):
        if self.args.test_only:
            print(f"==> Running Test Evaluation...")
            test_stats = self._run_epoch(0, "test")
            print(f"Test Result | Loss: {test_stats['loss']:.3f} | Acc: {test_stats['acc']:.2f}%")
            return

        start_time_total = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            t_stats, v_stats = self._run_epoch(epoch, "train"), self._run_epoch(epoch, "val")
            self.scheduler.step()
            
            w_mean, w_max = self._get_weight_stats()
            print(f"Ep {epoch+1} | T_Loss: {t_stats['loss']:.3f} T_Acc: {t_stats['acc']:.1f}% | V_Loss: {v_stats['loss']:.3f} V_Acc: {v_stats['acc']:.1f}% | W_Max: {w_max:.2e}")
            
            log_data = {f"train/{k}": v for k, v in t_stats.items()}
            log_data.update({f"val/{k}": v for k, v in v_stats.items()})
            log_data.update({
                "epoch": epoch+1, 
                "lr": self.optimizer.param_groups[0]["lr"],
                "weights/mean": w_mean,
                "weights/max": w_max
            })
            if self.scaler: log_data["amp/scale"] = self.scaler.get_scale()
            self.wandb.log(log_data)

            if v_stats["acc"] > self.best_acc:
                self.best_acc = v_stats["acc"]
                torch.save({"epoch": epoch+1, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "best_acc": self.best_acc, "wandb_run_id": self.wandb.run.id if self.wandb.run else None},
                           os.path.join(self.out_dir, "checkpoints", "best_model.pth"))

        total_runtime = time.time() - start_time_total
        self.wandb.summary["total_runtime_sec"] = total_runtime
        print(f"==> Training Complete. Total Runtime: {total_runtime/60:.2f} mins")

if __name__ == "__main__":
    Trainer(parse_args()).run()
