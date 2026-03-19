"""
3D VNN Backbone (7-block deep) with 2nd + 3rd order Volterra interactions.

Architecture:
    Block 1: Multi-kernel (5×5×5 + 3×3×3 + 1×1×1) → Quadratic → Pool
    Block 2: Quadratic → Pool
    Block 3: Quadratic (no pool)
    Block 4: Quadratic (no pool)
    Block 5: Quadratic + Cubic → Pool
    Block 6: Quadratic + Cubic (no pool)
    Block 7: Quadratic + Cubic → Pool

All blocks have residual shortcuts and gated quadratic branches.
With ``with_head=True``, includes a classifier (for standalone use).

Previously: vnn_rgb_of_complex.py
"""

import torch
import torch.nn as nn

from .blocks import MultiKernelBlock3D, VolterraBlock3D, ClassifierHead


class VNN(nn.Module):
    """7-block deep 3D VNN backbone with residual shortcuts.

    Args:
        num_classes: Number of output classes (used if ``with_head=True``).
        num_ch: Input channels (3 for RGB).
        with_head: If True, includes ClassifierHead and returns logits.
                   If False, returns feature maps.
    """

    def __init__(self, num_classes=400, num_ch=3, with_head=True):
        super().__init__()
        self.with_head = with_head

        # Block 1: Multi-kernel, quadratic + shortcut
        self.block1 = MultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2, use_shortcut=True, gate_quadratic=True,
        )

        # Blocks 2-4: Quadratic only + shortcut
        self.block2 = VolterraBlock3D(
            24, 32, Q=4, stride=2,
            use_shortcut=True, gate_quadratic=True,
        )
        self.block3 = VolterraBlock3D(
            32, 64, Q=4,
            use_shortcut=True, gate_quadratic=True,
        )
        self.block4 = VolterraBlock3D(
            64, 96, Q=4,
            use_shortcut=True, gate_quadratic=True,
        )

        # Blocks 5-7: Quadratic + Cubic + shortcut
        self.block5 = VolterraBlock3D(
            96, 128, Q=4, Qc=2, stride=2,
            use_cubic=True, use_shortcut=True, gate_quadratic=True,
        )
        self.block6 = VolterraBlock3D(
            128, 256, Q=4, Qc=2,
            use_cubic=True, use_shortcut=True, gate_quadratic=True,
        )
        self.block7 = VolterraBlock3D(
            256, 256, Q=2, Qc=2, stride=2,
            use_cubic=True, use_shortcut=True, gate_quadratic=True,
        )

        if with_head:
            self.classifier = ClassifierHead(12544, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        if self.with_head:
            x = self.classifier(x)
        return x


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VNN(num_classes=101)
    outputs = net(inputs)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
