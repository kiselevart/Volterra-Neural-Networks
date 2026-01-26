import argparse
import os
import time
import json
import csv
import socket
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.model_factory import get_model
from utils.data_factory import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Unified VNN Training & Benchmarking Script')
    
    # Task & Model
    parser.add_argument('--task', type=str, required=True, choices=['cifar', 'video'], help='Task type')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'ucf10', 'ucf101', 'hmdb51'], help='Dataset name')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['vnn_simple', 'vnn_ortho', 'resnet18', 'vnn_rgb', 'vnn_fusion'], 
                        help='Model architecture')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--Q', type=int, default=2, help='Volterra interaction factor (for VNNs)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for safe Mac usage)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--run_name', type=str, default=None, help='Custom name for this run')
    
    args = parser.parse_args()
    
    # Derived attributes
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'ucf10':
        args.num_classes = 10
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
        self.out_dir = os.path.join('runs', run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, 'logs'))
        
        # CSV Logger
        self.csv_file = open(os.path.join(self.out_dir, 'metrics.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time', 'lr'])
        
        # Save Config
        with open(os.path.join(self.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # 3. Model & Data
        self.model = get_model(args, self.device)
        self.loaders = get_dataloaders(args)
        
        # 4. Optimization
        # Handle specific optimizer needs (Video VNNs use Adam with specific groups, CIFAR uses SGD)
        if args.task == 'video':
            if hasattr(self.model, 'get_1x_lr_params'):
                params = [
                    {'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                    # If model has other params not in 1x, add them here or ensure get_1x covers all
                ]
            else:
                params = self.model.parameters()
            
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
            
        else: # CIFAR
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
            
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(self.loaders['train']), total=len(self.loaders['train']), 
                    desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (inputs, targets) in pbar:
            # Handle Video Fusion Tuple (rgb, flow)
            if isinstance(inputs, list) and len(inputs) == 2:
                # inputs is [rgb, flow]
                inputs = [x.to(self.device) for x in inputs]
                targets = targets.to(self.device)
                outputs = self.model(inputs) # Model wrapper handles tuple unpacking
            else:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'Loss': f"{running_loss/(batch_idx+1):.3f}", 'Acc': f"{100.*correct/total:.2f}%"})
            
        return running_loss / len(self.loaders['train']), 100. * correct / total

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(self.loaders['val']), total=len(self.loaders['val']), 
                    desc=f"Epoch {epoch+1}/{self.args.epochs} [Val  ]")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                if isinstance(inputs, list) and len(inputs) == 2:
                    inputs = [x.to(self.device) for x in inputs]
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                else:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'Loss': f"{running_loss/(batch_idx+1):.3f}", 'Acc': f"{100.*correct/total:.2f}%"})
                
        return running_loss / len(self.loaders['val']), 100. * correct / total

    def run(self):
        print(f"==> Starting training: {self.args.run_name}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            start_time = time.time()
            
            # Train & Val
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"    Summary | T_Loss: {train_loss:.4f} T_Acc: {train_acc:.2f}% | "
                  f"V_Loss: {val_loss:.4f} V_Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Info/LearningRate', current_lr, epoch)
            
            self.csv_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, epoch_time, current_lr])
            self.csv_file.flush()
            
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
                
        self.writer.close()
        self.csv_file.close()
        print(f"==> Training Complete. Results saved to {self.out_dir}")

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()
