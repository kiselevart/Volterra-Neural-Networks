import os
import argparse
import time
import socket
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataloaders.video_dataset_refactored import VideoDatasetRefactored
from network.fusion_refactored import vnn_rgb_of_highQ, vnn_fusion_highQ

def main():
    parser = argparse.ArgumentParser(description='Train VNN Fusion HighQ Refactored')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root (e.g. ucf101 raw videos)')
    parser.add_argument('--dataset_output', type=str, required=True, help='Path to dataset output/cache (preprocessed frames)')
    parser.add_argument('--dataset_name', type=str, default='ucf101', help='Dataset name (ucf101, hmdb51)')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classes')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='runs/vnn_fusion_refactored', help='Directory to save checkpoints and logs')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume from')
    parser.add_argument('--test_interval', type=int, default=5, help='Interval for testing')
    parser.add_argument('--snapshot_interval', type=int, default=5, help='Interval for saving snapshots')
    parser.add_argument('--download', action='store_true', help='Download UCF101 dataset automatically')
    
    args = parser.parse_args()

    if args.download:
        from utils.download_ucf101 import download_ucf101
        print(f"Ensuring dataset is downloaded to {args.dataset_root}...")
        download_ucf101(args.dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Save Dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    log_dir = os.path.join(args.save_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize Models
    # Note: vnn_rgb_of_highQ defines VNN (for RGB and OF branches)
    # vnn_fusion_highQ defines VNN_F (Fusion)
    
    print("Initializing models...")
    model_RGB = vnn_rgb_of_highQ.VNN(num_classes=args.num_classes, num_ch=3, pretrained=False)
    model_OF = vnn_rgb_of_highQ.VNN(num_classes=args.num_classes, num_ch=2, pretrained=False)
    # 192 input channels for fusion? Original code says: num_ch=192
    model_fuse = vnn_fusion_highQ.VNN_F(num_classes=args.num_classes, num_ch=192, pretrained=False)

    model_RGB.to(device)
    model_OF.to(device)
    model_fuse.to(device)

    # Optimizer
    # Original code had different params for different layers. Replicating broadly.
    train_params = [
        {'params': vnn_rgb_of_highQ.get_1x_lr_params(model_RGB), 'lr': args.lr},
        {'params': vnn_rgb_of_highQ.get_1x_lr_params(model_OF), 'lr': args.lr},
        {'params': vnn_fusion_highQ.get_1x_lr_params(model_fuse), 'lr': args.lr},
        {'params': vnn_fusion_highQ.get_10x_lr_params(model_fuse), 'lr': args.lr}
    ]
    
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    # Resume logic could be added here similar to original

    # Data Loaders
    print("Setting up dataloaders...")
    train_dataset = VideoDatasetRefactored(
        root_dir=args.dataset_root,
        output_dir=args.dataset_output,
        split='train',
        clip_len=16,
        compute_flow=True # Enable flow computation in dataloader
    )
    
    val_dataset = VideoDatasetRefactored(
        root_dir=args.dataset_root,
        output_dir=args.dataset_output,
        split='val',
        clip_len=16,
        compute_flow=True
    )
    
    test_dataset = VideoDatasetRefactored(
        root_dir=args.dataset_root,
        output_dir=args.dataset_output,
        split='test',
        clip_len=16,
        compute_flow=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Training started for {args.epochs} epochs.")
    
    for epoch in range(args.resume_epoch, args.epochs):
        scheduler.step()
        
        # Train
        model_RGB.train()
        model_OF.train()
        model_fuse.train()
        
        running_loss = 0.0
        running_corrects = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, inputs_of, labels in pbar:
            inputs = inputs.float().to(device)
            inputs_of = inputs_of.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            out_rgb = model_RGB(inputs)
            out_of = model_OF(inputs_of)
            # Concatenate features? 
            # Original code: model_fuse(torch.cat((outputs_rgb, outputs_of), 1))
            # Wait, outputs_rgb/of from model_RGB/OF are features or logits?
            # vnn_rgb_of_highQ.VNN returns `x` (features) NOT logits (fc8 commented out in VNN def in original?)
            # Let's check the code I read.
            # In `vnn_rgb_of_highQ.py`, `forward` returns `x` which seems to be the output of `pool4` (96 channels) + `pool5` commented out?
            # It returns `x` which is `pool4` output -> [B, 96, T, H, W] maybe? 
            # Ah, `vnn_rgb_of_highQ.py`: `return x` at end. `x` comes from `pool4`?
            # Actually, looking at `vnn_rgb_of_highQ.py` I just wrote:
            # x = self.pool4(...)
            # return x
            # `nch_out4 = 96`. So it returns 96 channels.
            # So concatenation of 2 branches (RGB + OF) = 96 + 96 = 192 channels.
            # `model_fuse` takes 192 channels. This matches.
            
            # The original `train_VNN_fusion_highQ.py` says:
            # outputs_rgb = model_RGB(inputs)
            # outputs_of = model_OF(inputs_of)
            # outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))
            
            # So yes, we concatenate features.
            
            combined = torch.cat((out_rgb, out_of), 1)
            outputs = model_fuse(combined)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/acc', epoch_acc, epoch)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation
        if (epoch + 1) % args.test_interval == 0: # Using test interval for validation/test
             run_validation(model_RGB, model_OF, model_fuse, val_loader, criterion, device, epoch, writer, 'val')
        
        # Save Checkpoint
        if (epoch + 1) % args.snapshot_interval == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict_rgb': model_RGB.state_dict(),
                'state_dict_of': model_OF.state_dict(),
                'state_dict_fuse': model_fuse.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

    writer.close()

def run_validation(model_RGB, model_OF, model_fuse, loader, criterion, device, epoch, writer, phase='val'):
    model_RGB.eval()
    model_OF.eval()
    model_fuse.eval()
    
    running_loss = 0.0
    running_corrects = 0.0
    
    with torch.no_grad():
        for inputs, inputs_of, labels in tqdm(loader, desc=f"Epoch {epoch+1} [{phase}]"):
            inputs = inputs.float().to(device)
            inputs_of = inputs_of.float().to(device)
            labels = labels.to(device)
            
            out_rgb = model_RGB(inputs)
            out_of = model_OF(inputs_of)
            combined = torch.cat((out_rgb, out_of), 1)
            outputs = model_fuse(combined)
            
            loss = criterion(outputs, labels)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    
    writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
    writer.add_scalar(f'{phase}/acc', epoch_acc, epoch)
    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

if __name__ == "__main__":
    main()
