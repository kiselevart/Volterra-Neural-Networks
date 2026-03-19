"""
3D VNN Backbone with optional Cubic toggle.

4-block backbone using the general cubic decomposition (a·b·c)
with a zero-initialized gate for stable warmup.

The ``use_cubic`` flag enables a clean A/B comparison:
    use_cubic=False → Pure 2nd-order (quadratic only)
    use_cubic=True  → 2nd + 3rd order (quadratic + general cubic)

Previously: vnn_cubic_simple_toggle.py
"""

import torch
import torch.nn as nn

from .blocks import VolterraBlock3D


class SimpleVNN(nn.Module):
    """4-block VNN backbone with optional cubic toggle.

    Args:
        use_cubic: If True, enables general cubic (a·b·c) in all blocks.
    """

    def __init__(self, use_cubic=True):
        super().__init__()

        config = [
            (3,  32, 2),   # Block 1
            (32, 64, 2),   # Block 2
            (64, 64, 1),   # Block 3 (no pool)
            (64, 96, 2),   # Block 4
        ]

        self.layers = nn.ModuleList([
            VolterraBlock3D(
                in_c, out_c, Q=4, Qc=2, stride=s,
                use_cubic=use_cubic, cubic_mode='general',
            )
            for in_c, out_c, s in config
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test 1: Without Cubic (Baseline)
    model_baseline = SimpleVNN(use_cubic=False)
    print(f"Baseline Params: {sum(p.numel() for p in model_baseline.parameters()):,}")

    # Test 2: With Cubic (Experiment — General Form)
    model_cubic = SimpleVNN(use_cubic=True)
    print(f"Cubic Params:    {sum(p.numel() for p in model_cubic.parameters()):,}")

    # Input check
    x = torch.randn(1, 3, 16, 112, 112)
    y = model_cubic(x)
    print(f"Output Shape:    {y.shape}")
