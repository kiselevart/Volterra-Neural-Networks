"""
video — 3D Volterra Neural Network modules for video.

Provides backbone architectures and fusion heads with 2nd + 3rd order
Volterra interactions for video classification.

Backbones:
    VNN         — 4-block backbone (backbone_4block)
    VNN_Deep    — 7-block deep backbone (backbone_7block)
    SimpleVNN   — 4-block with cubic toggle (backbone_cubic_toggle)

Heads:
    VNN_F       — Fusion classification head (fusion_head)

Building blocks:
    VolterraBlock3D    — Single-kernel Volterra block
    MultiKernelBlock3D — Multi-kernel Volterra block
    ClassifierHead     — Flatten + FC classifier
"""

from .backbone_4block import VNN
from .backbone_7block import VNN as VNN_Deep
from .backbone_cubic_toggle import SimpleVNN
from .fusion_head import VNN_F
from .blocks import VolterraBlock3D, MultiKernelBlock3D, ClassifierHead
from .volterra_ops import (
    volterra_quadratic,
    volterra_cubic_symmetric,
    volterra_cubic_general,
    init_vnn_weights,
)

__all__ = [
    # Networks
    "VNN", "VNN_Deep", "SimpleVNN", "VNN_F",
    # Building blocks
    "VolterraBlock3D", "MultiKernelBlock3D", "ClassifierHead",
    # Primitives
    "volterra_quadratic", "volterra_cubic_symmetric",
    "volterra_cubic_general", "init_vnn_weights",
]
