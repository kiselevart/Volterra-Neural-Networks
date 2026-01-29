#!/usr/bin/env python3
"""Verify PyTorch installation and GPU availability."""

import torch
import sys

print("=" * 60)
print("PyTorch Installation Verification")
print("=" * 60)

# Basic info
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"2. Python Version: {sys.version}")

# CUDA availability
print(f"\n3. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   - Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   - Current Device: {torch.cuda.current_device()}")
    print(f"   - CUDA Version: {torch.version.cuda}")
    print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("   ⚠ No CUDA devices found")

# MPS (Mac GPU)
if hasattr(torch.backends, 'mps'):
    print(f"\n4. MPS Available (Mac GPU): {torch.backends.mps.is_available()}")
else:
    print("\n4. MPS Available (Mac GPU): Not supported on this platform")

# Test tensor on GPU
if torch.cuda.is_available():
    print("\n5. Testing GPU Tensor Operations:")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("   ✓ Successfully created and computed on GPU")
        print(f"   - Result shape: {z.shape}")
        print(f"   - Result device: {z.device}")
    except Exception as e:
        print(f"   ✗ Error during GPU computation: {e}")
else:
    print("\n5. GPU Test: Skipped (no GPU available)")

print("\n" + "=" * 60)
if torch.cuda.is_available():
    print("✓ PyTorch is properly configured for GPU!")
else:
    print("✗ GPU not available - training will use CPU")
print("=" * 60)
