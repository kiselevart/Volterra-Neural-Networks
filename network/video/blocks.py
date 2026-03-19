"""
Reusable 3D Volterra building blocks for video models.

Building blocks:
    VolterraBlock3D    — Single-kernel block (linear + quadratic + optional cubic)
    MultiKernelBlock3D — Multi-kernel block (parallel convs at different scales)
    ClassifierHead     — Flatten → Dropout → Linear classifier
"""

import torch
import torch.nn as nn

from .volterra_ops import (
    volterra_quadratic,
    volterra_cubic_symmetric,
    volterra_cubic_general,
    init_vnn_weights,
)


class VolterraBlock3D(nn.Module):
    """Single-kernel 3D Volterra block.

    Computes::

        out = [shortcut(x) +] linear(x) + [quad_gate *] quad(x) [+ cubic_gate * cubic(x)]
        out = pool(out)

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        Q: Rank for quadratic decomposition (default 4).
        Qc: Rank for cubic decomposition (default 2).
        stride: If > 1, applies MaxPool3d(2) after summation.
        use_cubic: Enable 3rd-order cubic path.
        cubic_mode: ``'symmetric'`` (a²·b, 2Q channels) or
                    ``'general'`` (a·b·c, 3Q channels).
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Zero-init gate on the quadratic branch.
        kernel_size: Convolution kernel size (int or tuple).
    """

    def __init__(self, in_ch, out_ch, Q=4, Qc=2, stride=1,
                 use_cubic=False, cubic_mode='symmetric',
                 use_shortcut=False, gate_quadratic=True,
                 kernel_size=3):
        super().__init__()
        self.out_ch = out_ch
        self.Q = Q
        self.Qc = Qc
        self.use_cubic = use_cubic
        self.cubic_mode = cubic_mode
        self.use_shortcut = use_shortcut
        self.gate_quadratic = gate_quadratic

        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        pad = tuple(k // 2 for k in ks)

        # --- Linear path ---
        self.conv_lin = nn.Conv3d(in_ch, out_ch, ks, padding=pad)
        self.bn_lin = nn.BatchNorm3d(out_ch)

        # --- Quadratic path ---
        self.conv_quad = nn.Conv3d(in_ch, 2 * Q * out_ch, ks, padding=pad)
        self.bn_quad = nn.BatchNorm3d(out_ch)
        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        # --- Cubic path (optional) ---
        if use_cubic:
            cubic_mult = 3 if cubic_mode == 'general' else 2
            self.conv_cubic = nn.Conv3d(in_ch, cubic_mult * Qc * out_ch, ks, padding=pad)
            self.bn_cubic = nn.BatchNorm3d(out_ch)
            self.cubic_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        # --- Shortcut (optional) ---
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        # --- Pooling ---
        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        # Linear
        out = self.bn_lin(self.conv_lin(x))

        # Quadratic
        q = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
        if self.gate_quadratic:
            q = self.quad_gate.view(1, -1, 1, 1, 1) * q
        out = out + q

        # Cubic
        if self.use_cubic:
            fn = volterra_cubic_general if self.cubic_mode == 'general' else volterra_cubic_symmetric
            c = self.bn_cubic(fn(self.conv_cubic(x), self.Qc, self.out_ch))
            out = out + self.cubic_gate.view(1, -1, 1, 1, 1) * c

        # Shortcut
        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


class MultiKernelBlock3D(nn.Module):
    """Multi-kernel 3D Volterra block with parallel convolutions.

    Each kernel produces ``ch_per_kernel`` channels; outputs are concatenated.
    Total output channels = ``ch_per_kernel × len(kernels)``.

    Computes::

        out = [shortcut(x) +] concat_linear(x) + [quad_gate *] concat_quad(x)
        out = pool(out)

    Args:
        in_ch: Input channels.
        ch_per_kernel: Output channels per kernel branch.
        kernels: List of kernel sizes, e.g. ``[(5,5,5), (3,3,3), (1,1,1)]``.
        Q: Rank for quadratic decomposition.
        stride: If > 1, applies MaxPool3d(2) after summation.
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Zero-init gate on the quadratic branch.
    """

    def __init__(self, in_ch, ch_per_kernel, kernels, Q=4, stride=1,
                 use_shortcut=False, gate_quadratic=True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.kernels = kernels
        self.out_ch = ch_per_kernel * len(kernels)
        self.use_shortcut = use_shortcut
        self.gate_quadratic = gate_quadratic

        # Linear paths (one conv per kernel size)
        self.lin_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.lin_convs.append(nn.Conv3d(in_ch, ch_per_kernel, ks, padding=pad))
        self.bn_lin = nn.BatchNorm3d(self.out_ch)

        # Quadratic paths (one conv per kernel size)
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.quad_convs.append(nn.Conv3d(in_ch, 2 * Q * ch_per_kernel, ks, padding=pad))
        self.bn_quad = nn.BatchNorm3d(self.out_ch)

        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm3d(self.out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        # Linear: parallel convs → concat → BN
        lin = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))

        # Quadratic: parallel interactions → concat → BN
        quads = [volterra_quadratic(c(x), self.Q, self.ch_per_kernel)
                 for c in self.quad_convs]
        quad = self.bn_quad(torch.cat(quads, dim=1))

        if self.gate_quadratic:
            quad = self.quad_gate.view(1, -1, 1, 1, 1) * quad

        out = lin + quad

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


class ClassifierHead(nn.Module):
    """Flatten → Dropout → FC classifier.

    Args:
        fc_features: Number of features after flatten.
        num_classes: Number of output classes.
        dropout: Dropout probability (default 0.5).
    """

    def __init__(self, fc_features, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
