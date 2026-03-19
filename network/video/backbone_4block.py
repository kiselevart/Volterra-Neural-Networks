"""
3D VNN Backbone (4-block) with 2nd + 3rd order Volterra interactions.

Architecture:
    Block 1: Multi-kernel (5×5×5 + 3×3×3 + 1×1×1) → Quadratic → Pool
    Block 2: Single kernel → Quadratic → Pool
    Block 3: Single kernel → Quadratic + Symmetric Cubic (no pool)
    Block 4: Single kernel → Quadratic + Symmetric Cubic → Pool

Outputs feature maps [B, 96, T/8, H/8, W/8] for downstream classification head.

Previously: vnn_rgb_of_highQ.py
"""

import torch
import torch.nn as nn

from .blocks import MultiKernelBlock3D, VolterraBlock3D


class VNN(nn.Module):
    """4-block 3D VNN backbone with multi-scale first block.

    Args:
        num_ch: Input channels (3 for RGB, 2 for optical flow).
    """

    def __init__(self, num_ch=3):
        super().__init__()

        # Block 1: Multi-kernel, quadratic only
        self.block1 = MultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2,
        )

        # Block 2: Quadratic only
        self.block2 = VolterraBlock3D(24, 32, Q=4, stride=2, use_shortcut=True)

        # Block 3: Quadratic + Symmetric Cubic (no pool)
        self.block3 = VolterraBlock3D(
            32, 64, Q=4, Qc=2,
            use_cubic=True, cubic_mode='symmetric',
            use_shortcut=True,
        )

        # Block 4: Quadratic + Symmetric Cubic
        self.block4 = VolterraBlock3D(
            64, 96, Q=4, Qc=2, stride=2,
            use_cubic=True, cubic_mode='symmetric',
            use_shortcut=True,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VNN(num_ch=3)
    outputs = net(inputs)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
