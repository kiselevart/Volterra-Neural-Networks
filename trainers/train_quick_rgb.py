import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloaders.dataset import VideoDataset
from network.fusion import vnn_rgb_of_highQ, vnn_fusion_highQ

# Use CUDA or MPS if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- Configuration ---
nEpochs = 50
resume_epoch = 0
useTest = True
nTestInterval = 10
snapshot = 5
lr = 1e-4
dataset = 'ucf10'

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
elif dataset == 'ucf10':
    num_classes = 10
else:
    raise NotImplementedError('Dataset not implemented.')

save_dir_root = os.path.dirname(os.path.abspath(__file__))
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = (int(runs[-1].split('_')[-1]) + 1) if runs and resume_epoch == 0 else (int(runs[-1].split('_')[-1]) if runs else 0)
save_dir = os.path.join(save_dir_root, 'run', f'run_{run_id}')

modelName = 'VNN_RGB'
saveName = f"{modelName}-{dataset}"

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):

    # 1. Initialize RGB-only model
    model = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
    # The VNN backbone outputs 96 channels, so we initialize the classification head with 96 input channels
    classifier = vnn_fusion_highQ.VNN_F(num_classes=num_classes, num_ch=96, pretrained=False)
    
    # Use custom LR params if the model provides them
    train_params = [
        {'params': vnn_rgb_of_highQ.get_1x_lr_params(model), 'lr': lr},
        {'params': classifier.parameters(), 'lr': lr}
    ]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 2. Resume logic fix
    if resume_epoch != 0:
        load_path = os.path.join(save_dir, 'models', f"{saveName}_epoch-{resume_epoch - 1}.pth.tar")
        print(f"Initializing weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict_rgb'])
        # classifier.load_state_dict(checkpoint['state_dict_cls']) # Add this if saving cls
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    model.to(device)
    classifier.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 3. Data Loaders
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=16, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=16, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=16, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    for epoch in range(resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss, running_corrects = 0.0, 0.0

            if phase == 'train':
                model.train()
                classifier.train()
            else:
                model.eval()
                classifier.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # Remove Optical Flow computation to focus on RGB
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    features = model(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        features = model(inputs)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)

                preds = torch.max(outputs, 1)[1]
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            writer.add_scalar(f'data/{phase}_loss_epoch', epoch_loss, epoch)
            writer.add_scalar(f'data/{phase}_acc_epoch', epoch_acc, epoch)

            print(f"[{phase}] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"Execution time: {timeit.default_timer() - start_time:.2f}s\n")

        # 4. Save model
        if epoch % save_epoch == (save_epoch - 1):
            save_path = os.path.join(save_dir, 'models', f"{saveName}_epoch-{epoch}.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'state_dict_rgb': model.state_dict(),
                'opt_dict': optimizer.step(),
            }, save_path)
            print(f"Save model at {save_path}\n")

        # 5. Periodic Testing
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            classifier.eval()
            test_loss, test_corrects = 0.0, 0.0
            
            for inputs, labels in tqdm(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    features = model(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    preds = torch.max(outputs, 1)[1]
                
                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

            print(f"[test] Epoch: {epoch+1} Loss: {test_loss/len(test_dataloader.dataset):.4f} Acc: {test_corrects.double()/len(test_dataloader.dataset):.4f}")

    writer.close()

if __name__ == "__main__":
    train_model()