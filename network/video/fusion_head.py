"""
3D VNN Fusion Head with 2nd + 3rd order Volterra interactions.

Single-block fusion classifier that takes concatenated backbone features
and produces class logits.  Uses both quadratic and symmetric cubic
Volterra interactions with a residual shortcut and gating.

Previously: vnn_fusion_highQ.py
"""

import torch
import torch.nn as nn

from .blocks import VolterraBlock3D, ClassifierHead


class VNN_F(nn.Module):
    """Fusion classification head with Quadratic + Symmetric Cubic.

    Architecture::

        VolterraBlock3D (Q+C, shortcut, gated) → Pool → Dropout → FC

    Args:
        num_classes: Number of output classes.
        num_ch: Input channels (96 for single-stream, 192 for two-stream fusion).
        pretrained: Unused (API compat).
    """

    def __init__(self, num_classes, num_ch=3, pretrained=False):
        super().__init__()

        self.block1 = VolterraBlock3D(
            num_ch, 256, Q=2, Qc=2, stride=2,
            use_cubic=True, cubic_mode='symmetric',
            use_shortcut=True, gate_quadratic=True,
        )
        self.classifier = ClassifierHead(12544, num_classes)

    def forward(self, x):
        x = self.block1(x)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Differential LR helpers (10× LR for the final FC layer)
    # ------------------------------------------------------------------

    def get_1x_lr_params(self):
        """Returns all parameters except the final classifier FC layer."""
        skip = {id(p) for p in self.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        """Returns the classifier FC layer parameters (for higher LR)."""
        for p in self.classifier.fc.parameters():
            if p.requires_grad:
                yield p


# ------------------------------------------------------------------
# Module-level functions for backward compat with model_factory
# ------------------------------------------------------------------

def get_1x_lr_params(model):
    """Returns all parameters except the classifier FC layer."""
    return model.get_1x_lr_params()


def get_10x_lr_params(model):
    """Returns the classifier FC layer parameters."""
    return model.get_10x_lr_params()


if __name__ == "__main__":
    inputs = torch.rand(1, 96, 2, 14, 14)
    net = VNN_F(num_classes=101, num_ch=96)
    outputs = net(inputs)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
