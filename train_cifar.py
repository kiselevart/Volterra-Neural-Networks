import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from network.cifar.vnn_cifar import VNN_CIFAR
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train VNN on CIFAR-10')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
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

    # Download to a 'data' folder
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- Model Setup ---
    print('==> Building model...')
    net = VNN_CIFAR(num_classes=10)
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # SGD with Momentum usually works better for CIFAR than Adam
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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

            if (batch_idx + 1) % 100 == 0:
                print(f"   [Epoch {epoch+1} Batch {batch_idx+1}] Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} Summary | Time: {epoch_time:.1f}s | "
              f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")
        
        scheduler.step()

        # Validation every epoch
        test(net, testloader, device, criterion)

def test(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"--> Test Set | Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    main()