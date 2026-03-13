import argparse
import os
import time
import platform
import socket
import contextlib
import warnings
import atexit
from typing import Any
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from utils.model_factory import get_model
from utils.data_factory import get_dataloaders

def parse_args():
    desc = """Unified VNN Training & Benchmarking Script
==========================================
This script trains various Volterra Neural Network (VNN) architectures and baselines
on CIFAR-10 and Video datasets (UCF101, HMDB51).

Supported Models:
  - vnn_simple: Basic VNN
  - vnn_ortho:  Orthogonal VNN
  - resnet18:   Standard ResNet18 baseline
  - vnn_rgb:    RGB-only VNN for video
  - vnn_fusion: Two-stream fusion (RGB + Optical Flow) for video"""

    epilog = """Examples:
  1. Train VNN Ortho on CIFAR-10:
     python train.py --task cifar --dataset cifar10 --model vnn_ortho --epochs 50 --batch_size 128 --lr 0.01

  2. Train ResNet18 baseline on CIFAR-10:
     python train.py --task cifar --dataset cifar10 --model resnet18

  3. Train VNN Fusion on UCF101 (Video):
     python train.py --task video --dataset ucf101 --model vnn_fusion --num_workers 4 --batch_size 16

  4. Train VNN Fusion with 3rd-order cubic terms (higher-order):
     python train.py --task video --dataset ucf101 --model vnn_fusion_ho --num_workers 4 --batch_size 8

  5. Resume from checkpoint:
     python train.py --task cifar --dataset cifar10 --model vnn_ortho --resume runs/vnn_ortho_.../checkpoints/last.pth"""

    parser = argparse.ArgumentParser(
        description=desc, 
        epilog=epilog, 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Task & Model
    parser.add_argument('--task', type=str, required=True, choices=['cifar', 'video'], help='Task type')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'ucf10', 'ucf101', 'hmdb51', 'ucf11'], help='Dataset name')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['vnn_simple', 'vnn_ortho', 'resnet18', 'vnn_rgb', 'vnn_fusion',
                                 'vnn_rgb_ho', 'vnn_fusion_ho', 'vnn_complex_ho', 'vnn_cubic_simple_toggle'], 
                        help='Model architecture (append _ho for higher-order cubic variants)')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--Q', type=int, default=2, help='Volterra interaction factor (for VNNs)')
    parser.add_argument('--disable_cubic', action='store_true',
                        help='Disable cubic term for vnn_cubic_simple_toggle (quadratic-only ablation)')

    # Experiment Tracking (Weights & Biases) - enabled by default
    parser.add_argument('--wandb_project', type=str, default=os.getenv('WANDB_PROJECT', 'vnn-research'), help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=os.getenv('WANDB_ENTITY'), help='W&B entity/team (optional)')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name (optional; defaults to auto run_name)')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline'],
                        help='W&B mode')
    parser.add_argument('--wandb_on_fail', type=str, default='abort', choices=['abort', 'offline'],
                        help='Behavior when W&B init fails: abort training or continue in offline mode')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Optional W&B tags')
    
    # System
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for safe Mac usage)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--run_name', type=str, default=None, help='Custom name for this run')
    
    args = parser.parse_args()
    
    # Derived attributes
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'ucf11':
        args.num_classes = 11
    elif args.dataset == 'ucf101':
        args.num_classes = 101
    elif args.dataset == 'hmdb51':
        args.num_classes = 51
        
    return args

class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0.0
        self.wandb: Any = None
        self.wandb_run: Any = None
        self.epoch_bench = []
        self.run_name = None
        
        # 1. Device Setup
        if args.device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device)
            
        print(f"==> Using Device: {self.device}")

        # 2. Directories & Logging
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        run_name = args.run_name if args.run_name else f"{args.model}_{args.dataset}_{timestamp}"
        self.run_name = run_name
        self.out_dir = os.path.join('runs', run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)

        # 2.1 W&B (initialize early to fail fast before expensive dataset setup)
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "W&B is required for this training script. "
                "Install it with: `uv add wandb` or `pip install wandb`."
            ) from exc

        wandb_name = args.wandb_name if args.wandb_name else run_name
        self.wandb = wandb
        init_kwargs = {
            'project': args.wandb_project,
            'entity': args.wandb_entity,
            'name': wandb_name,
            'mode': args.wandb_mode,
            'tags': args.wandb_tags,
            'dir': self.out_dir,
            'reinit': True,
            'config': {
                **vars(args),
                'device': str(self.device),
                'hostname': socket.gethostname(),
                'output_dir': self.out_dir,
            },
        }

        try:
            self.wandb_run = self.wandb.init(**init_kwargs)
        except Exception as exc:
            if args.wandb_on_fail == 'offline' and args.wandb_mode != 'offline':
                warnings.warn(
                    f"W&B init failed in online mode ({exc}). Falling back to offline mode.",
                    RuntimeWarning,
                )
                init_kwargs['mode'] = 'offline'
                self.wandb_run = self.wandb.init(**init_kwargs)
            else:
                raise RuntimeError(
                    "W&B initialization failed. "
                    "Login with `wandb login`, verify network access, or set --wandb_on_fail offline."
                ) from exc

        atexit.register(self._cleanup)
        
        # 3. Model & Data
        self.model = get_model(args, self.device)
        self.loaders = get_dataloaders(args)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.wandb.config.update({
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
        }, allow_val_change=True)
        self.wandb.define_metric('epoch')
        self.wandb.define_metric('train/*', step_metric='epoch')
        self.wandb.define_metric('val/*', step_metric='epoch')
        self.wandb.define_metric('bench/*', step_metric='epoch')
        self.wandb.define_metric('weights/*', step_metric='epoch')
        self.wandb.define_metric('grads/*', step_metric='epoch')
        
        # 4. Optimization
        # Handle specific optimizer needs (Video VNNs use Adam with specific groups, CIFAR uses SGD)
        if args.task == 'video':
            get_1x = getattr(self.model, 'get_1x_lr_params', None)
            if callable(get_1x):
                params = [
                    {'params': get_1x(), 'lr': args.lr},
                    # If model has other params not in 1x, add them here or ensure get_1x covers all
                ]
            else:
                params = self.model.parameters()
            
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
            
        else: # CIFAR
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
            
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # Mixed precision with conservative loss scaling
        self.amp_enabled = self.device.type == 'cuda'
        self.autocast_device = self.device.type if self.amp_enabled else 'cpu'
        # GradScaler is only supported on CUDA
        self.scaler = GradScaler(device='cuda', init_scale=2**16, growth_interval=100) if self.device.type == 'cuda' else None

        # 5. Resume
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"==> Loading checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=self.device)
                self.start_epoch = checkpoint['epoch']
                self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"==> Resuming from epoch {self.start_epoch}")

    def _cleanup(self):
        """Ensure W&B run is closed on exit or crash."""
        try:
            if self.wandb is not None:
                self.wandb.finish()
        except Exception:
            pass

    def _get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def _autocast(self):
        if self.amp_enabled:
            return autocast(device_type=self.autocast_device, enabled=True)
        return contextlib.nullcontext()

    def _log_model_stats(self, epoch):
        # Aggregate weight/grad summaries for lightweight W&B logging.
        stats = {'epoch': epoch + 1}
        with torch.no_grad():
            weight_means = []
            weight_stds = []
            grad_norms = []
            for name, param in self.model.named_parameters():
                if param is None:
                    continue
                if param.data is not None:
                    weight_means.append(param.data.mean().item())
                    weight_stds.append(param.data.std().item())
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    grad_norms.append(grad_norm)

            if weight_means:
                stats['weights/mean'] = sum(weight_means) / len(weight_means)
            if weight_stds:
                stats['weights/std'] = sum(weight_stds) / len(weight_stds)
            if grad_norms:
                stats['grads/norm_mean'] = sum(grad_norms) / len(grad_norms)

        # Scaler value (if using AMP)
        if self.scaler is not None:
            stats['amp/scale'] = float(self.scaler.get_scale())

        self.wandb.log(stats)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        skipped_nonfinite = 0
        step_times = []
        epoch_start = time.time()
        
        pbar = tqdm(enumerate(self.loaders['train']), total=len(self.loaders['train']), 
                    desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (inputs, targets) in pbar:
            step_start = time.time()
            # Handle Video Fusion Tuple (rgb, flow)
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(self.device) for x in inputs]
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)
            
            self.optimizer.zero_grad()
            if self.scaler:
                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                if (not torch.isfinite(loss)) or (not torch.isfinite(outputs).all()):
                    skipped_nonfinite += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                if (not torch.isfinite(loss)) or (not torch.isfinite(outputs).all()):
                    skipped_nonfinite += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            step_times.append(time.time() - step_start)
            
            display_loss = running_loss / max(valid_batches, 1)
            display_acc = (100. * correct / total) if total > 0 else 0.0
            pbar.set_postfix({'Loss': f"{display_loss:.3f}", 'Acc': f"{display_acc:.2f}%", 'SkipNF': skipped_nonfinite})
            
        epoch_time = time.time() - epoch_start
        samples_per_sec = total / epoch_time if epoch_time > 0 else 0.0
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
        if valid_batches == 0:
            warnings.warn("All training batches were non-finite; returning zeroed metrics for this epoch.")
        return {
            'loss': (running_loss / valid_batches) if valid_batches > 0 else 0.0,
            'acc': (100. * correct / total) if total > 0 else 0.0,
            'samples': total,
            'epoch_time': epoch_time,
            'samples_per_sec': samples_per_sec,
            'avg_step_time': avg_step_time,
            'valid_batches': valid_batches,
            'skipped_nonfinite': skipped_nonfinite,
        }

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        skipped_nonfinite = 0
        step_times = []
        epoch_start = time.time()
        
        pbar = tqdm(enumerate(self.loaders['val']), total=len(self.loaders['val']), 
                    desc=f"Epoch {epoch+1}/{self.args.epochs} [Val  ]")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                step_start = time.time()
                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) for x in inputs]
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.long).view(-1)
                
                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                if (not torch.isfinite(loss)) or (not torch.isfinite(outputs).all()):
                    skipped_nonfinite += 1
                    continue
                
                running_loss += loss.item()
                valid_batches += 1
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                step_times.append(time.time() - step_start)
                
                display_loss = running_loss / max(valid_batches, 1)
                display_acc = (100. * correct / total) if total > 0 else 0.0
                pbar.set_postfix({'Loss': f"{display_loss:.3f}", 'Acc': f"{display_acc:.2f}%", 'SkipNF': skipped_nonfinite})
                
        epoch_time = time.time() - epoch_start
        samples_per_sec = total / epoch_time if epoch_time > 0 else 0.0
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
        if valid_batches == 0:
            warnings.warn("All validation batches were non-finite; returning zeroed metrics for this epoch.")
        return {
            'loss': (running_loss / valid_batches) if valid_batches > 0 else 0.0,
            'acc': (100. * correct / total) if total > 0 else 0.0,
            'samples': total,
            'epoch_time': epoch_time,
            'samples_per_sec': samples_per_sec,
            'avg_step_time': avg_step_time,
            'valid_batches': valid_batches,
            'skipped_nonfinite': skipped_nonfinite,
        }

    def run(self):
        print(f"==> Starting training: {self.run_name}")
        full_start = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            start_time = time.time()
            
            # Train & Val
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(epoch)
            train_loss, train_acc = train_stats['loss'], train_stats['acc']
            val_loss, val_acc = val_stats['loss'], val_stats['acc']
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            current_lrs = self._get_lr()
            current_lr = current_lrs[0] if current_lrs else 0.0
            
            # Logging
            scaler_val = float(self.scaler.get_scale()) if self.scaler is not None else None
            scaler_str = f" | AMP_Scale: {scaler_val:.1f}" if scaler_val is not None else ""
            lr_str = ", ".join([f"{lr:.6f}" for lr in current_lrs])
            print(f"    Summary | T_Loss: {train_loss:.4f} T_Acc: {train_acc:.2f}% | "
                f"V_Loss: {val_loss:.4f} V_Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s | LR: [{lr_str}]{scaler_str}")
            print(
                f"    Bench   | T_SPS: {train_stats['samples_per_sec']:.2f} | V_SPS: {val_stats['samples_per_sec']:.2f} "
                f"| T_step: {train_stats['avg_step_time']*1000:.1f}ms | V_step: {val_stats['avg_step_time']*1000:.1f}ms"
            )
            if train_stats.get('skipped_nonfinite', 0) > 0 or val_stats.get('skipped_nonfinite', 0) > 0:
                print(
                    f"    Stable  | Skipped non-finite batches -> "
                    f"train: {train_stats.get('skipped_nonfinite', 0)}, val: {val_stats.get('skipped_nonfinite', 0)}"
                )
            
            self._log_model_stats(epoch)

            self.wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'lr': current_lr,
                'epoch/time_sec': epoch_time,
                'bench/train_samples_per_sec': train_stats['samples_per_sec'],
                'bench/val_samples_per_sec': val_stats['samples_per_sec'],
                'bench/train_avg_step_ms': train_stats['avg_step_time'] * 1000.0,
                'bench/val_avg_step_ms': val_stats['avg_step_time'] * 1000.0,
                'bench/train_skipped_nonfinite': train_stats.get('skipped_nonfinite', 0),
                'bench/val_skipped_nonfinite': val_stats.get('skipped_nonfinite', 0),
                'amp/scale': scaler_val if scaler_val is not None else 0.0,
            })

            self.epoch_bench.append({
                'epoch': epoch + 1,
                'train_samples_per_sec': train_stats['samples_per_sec'],
                'val_samples_per_sec': val_stats['samples_per_sec'],
                'train_avg_step_ms': train_stats['avg_step_time'] * 1000.0,
                'val_avg_step_ms': val_stats['avg_step_time'] * 1000.0,
                'epoch_time_sec': epoch_time,
            })
            
            # Checkpointing
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
                'args': vars(self.args)
            }
            
            # Save periodic
            if (epoch + 1) % 10 == 0:
                 torch.save(state, os.path.join(self.out_dir, 'checkpoints', f'checkpoint_ep{epoch+1}.pth'))
            
            # Save Best
            if val_acc > self.best_acc:
                print(f"    New Best Accuracy! ({self.best_acc:.2f}% -> {val_acc:.2f}%) Saving model...")
                self.best_acc = val_acc
                state['best_acc'] = val_acc
                torch.save(state, os.path.join(self.out_dir, 'checkpoints', 'best_model.pth'))

        total_runtime = time.time() - full_start
        avg_train_sps = sum(x['train_samples_per_sec'] for x in self.epoch_bench) / len(self.epoch_bench) if self.epoch_bench else 0.0
        avg_val_sps = sum(x['val_samples_per_sec'] for x in self.epoch_bench) / len(self.epoch_bench) if self.epoch_bench else 0.0

        self.wandb.summary['best_val_acc'] = self.best_acc
        self.wandb.summary['epochs_ran'] = len(self.epoch_bench)
        self.wandb.summary['total_runtime_sec'] = total_runtime
        self.wandb.summary['avg_train_samples_per_sec'] = avg_train_sps
        self.wandb.summary['avg_val_samples_per_sec'] = avg_val_sps
        self.wandb.summary['device'] = str(self.device)
        self.wandb.summary['hostname'] = socket.gethostname()
        self.wandb.summary['platform'] = platform.platform()
        self.wandb.summary['total_params'] = self.total_params
        self.wandb.summary['trainable_params'] = self.trainable_params
        self.wandb.summary['epoch_benchmarks'] = self.epoch_bench

        self.wandb.finish()
        print(f"==> Training Complete. Results saved to {self.out_dir}")

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()
