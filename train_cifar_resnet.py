import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import argparse

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Train Orthogonal Res-VNN on CIFAR-10')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') # Higher LR for ResNet style
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs') # Reduced from 200 for demo
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    args = parser.parse_args()

    # --- Device Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # --- Data Preparation ---
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # --- Model Setup ---
    print('==> Building ResNet-18 (Baseline)...')
    # Use standard ResNet18 but adapt for CIFAR-10 size
    net = models.resnet18(pretrained=False)
    # CIFAR-10 has 10 classes, not 1000
    net.fc = nn.Linear(net.fc.in_features, 10)
    # CIFAR-10 images are 32x32, so we need to modify the first layer to avoid downsampling too aggressively
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool to keep spatial dimensions larger for small images
    net.maxpool = nn.Identity()
    
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # Cosine Annealing to 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Training Progress Bar
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update pbar description with current metrics
            pbar.set_postfix({'Loss': f"{train_loss/(batch_idx+1):.3f}", 'Acc': f"{100.*correct/total:.2f}%"})

        epoch_time = time.time() - start_time
        # print(f"Epoch {epoch+1}/{args.epochs} Summary | Time: {epoch_time:.1f}s | "
        #       f"Loss: {train_loss/(len(trainloader)):.3f} | Acc: {100.*correct/total:.2f}%")
        
        scheduler.step()

        # Validation every epoch
        test(net, testloader, device, criterion, epoch, args.epochs)

def test(net, testloader, device, criterion, epoch, total_epochs):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Testing Progress Bar
    pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Epoch {epoch+1}/{total_epochs} [Test ]")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'Loss': f"{test_loss/(batch_idx+1):.3f}", 'Acc': f"{100.*correct/total:.2f}%"})

    print(f"--> Test Set Result | Loss: {test_loss/(len(testloader)):.3f} | Acc: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    main()