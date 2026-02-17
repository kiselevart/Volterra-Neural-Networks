#!/usr/bin/env python3
"""Benchmark GPU performance."""

import torch
import torch.nn as nn
import time

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def sync_device():
    """Synchronize the current device for accurate timing."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

# Test 1: Simple forward pass
print("=" * 60)
print(f"GPU Performance Benchmark (device: {device})")
print("=" * 60)

# Warm up
for _ in range(5):
    x = torch.randn(128, 3, 32, 32, device=device)
    net = nn.Linear(10, 10, device=device)

# ResNet-like model
from torchvision import models
net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 10)
net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net = net.to(device)
net.eval()

# Benchmark forward pass
print("\n1. Forward Pass (128 batch, 10 iterations):")
sync_device()
start = time.time()
for _ in range(10):
    x = torch.randn(128, 3, 32, 32, device=device)
    with torch.no_grad():
        _ = net(x)
sync_device()
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s ({10/elapsed:.1f} batches/sec)")

# Benchmark forward + backward
print("\n2. Forward + Backward (128 batch, 10 iterations):")
net.train()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

sync_device()
start = time.time()
for _ in range(10):
    x = torch.randn(128, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (128,), device=device)
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
sync_device()
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s ({10/elapsed:.1f} batches/sec)")

# Benchmark with CPU sync (like in your code)
print("\n3. Forward + Backward + CPU Sync (like your code):")
sync_device()
start = time.time()
for _ in range(10):
    x = torch.randn(128, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (128,), device=device)
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    # These cause GPU stalls like in your code:
    _ = loss.item()
    _, predicted = outputs.max(1)
    _ = predicted.eq(y).sum().item()
    
sync_device()
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s ({10/elapsed:.1f} batches/sec)")

print("\n" + "=" * 60)
print("If (3) is much slower than (2), the problem is CPU sync overhead!")
print("=" * 60)
