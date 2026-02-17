"""
3D VNN Backbone (Higher-Order) with 2nd + 3rd order Volterra interactions.

Volterra series expansion up to 3rd order:
    y = h0 + Σ h1(i)·x(i) + Σ h2(i,j)·x(i)·x(j) + Σ h3(i,j,k)·x(i)·x(j)·x(k)

2nd-order (quadratic):  CP decomposition  → left · right          (2·Q·C channels)
3rd-order (cubic):      Symmetric CP      → a² · b                (2·Q·C channels)

The symmetric decomposition exploits the inherent symmetry of Volterra kernels:
    h3(i,j,k) = h3(j,i,k) = h3(k,j,i) = ...
so we tie two CP factors: h3 ≈ Σ_q a_q(i)·a_q(j)·b_q(k)

Blocks 1-2: Linear + Quadratic only (early features, small spatial dims preserved)
Blocks 3-4: Linear + Quadratic + Symmetric Cubic (deeper, more abstract features)
    Cubic terms use zero-initialized gating for stable warmup.
"""

import torch
import torch.nn as nn


def _volterra_quadratic(x_conv, Q, nch_out):
    """Vectorized 2nd-order Volterra interaction.

    CP decomposition: h2(i,j) ≈ Σ_q a_q(i)·b_q(j)

    Args:
        x_conv: Tensor [B, 2*Q*C, T, H, W] from quadratic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    mid = Q * nch_out
    left = x_conv[:, :mid]
    right = x_conv[:, mid:]
    product = left * right  # [B, Q*C, T, H, W]
    # Reshape to [B, Q, C, T, H, W] and sum over Q dimension
    shape = product.shape
    spatial = shape[2:]
    return product.view(shape[0], Q, nch_out, *spatial).sum(dim=1)


def _volterra_cubic_symmetric(x_conv, Q, nch_out):
    """Symmetric 3rd-order Volterra interaction (a² · b decomposition).

    Symmetric CP decomposition: h3(i,j,k) ≈ Σ_q a_q(i)·a_q(j)·b_q(k)
    Since two factors are tied, only 2·Q·C channels needed (same as quadratic).

    The a² term acts as an energy/magnitude detector (always ≥ 0),
    while b modulates sign and scale — a learned nonlinear gating mechanism.

    Gradient: ∂/∂a = 2a·b (self-regularizing: small a → small gradient).

    Args:
        x_conv: Tensor [B, 2*Q*C, T, H, W] from cubic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    mid = Q * nch_out
    a = x_conv[:, :mid]   # Feature detector (will be squared)
    b = x_conv[:, mid:]   # Modulator/gate
    product = (a * a) * b  # a² · b  [B, Q*C, T, H, W]
    shape = product.shape
    spatial = shape[2:]
    return product.view(shape[0], Q, nch_out, *spatial).sum(dim=1)


class VNN(nn.Module):
    """3D VNN Backbone with 2nd + 3rd order Volterra interactions.

    Architecture (4 blocks):
        Block 1: Multi-kernel (3×3×3 + 3×3×3 + 1×1×1) → Quadratic → Pool
        Block 2: Single kernel → Quadratic → Pool
        Block 3: Single kernel → Quadratic + Cubic → No Pool
        Block 4: Single kernel → Quadratic + Cubic → Pool

    This module outputs feature maps (no classifier head).
    Use with a fusion/classification head (e.g., VNN_F).
    """

    def __init__(self, num_classes, num_ch=3, pretrained=False):
        super(VNN, self).__init__()

        # ===== Block 1: Multi-kernel, Quadratic only =====
        Q1 = 4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8
        sum_chans = nch_out1_5 + nch_out1_3 + nch_out1_1  # 24

        # Linear path (3 parallel kernels approximating multi-scale receptive fields)
        self.conv11_5 = nn.Conv3d(num_ch, nch_out1_5, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.conv11_3 = nn.Conv3d(num_ch, nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_1 = nn.Conv3d(num_ch, nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn11 = nn.BatchNorm3d(sum_chans)

        # Quadratic path (3 parallel kernels)
        self.conv21_5 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_5, kernel_size=(5, 5, 5), padding=(2 , 2, 2))
        self.conv21_3 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_1 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn21 = nn.BatchNorm3d(sum_chans)

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Block 2: Quadratic only =====
        Q2 = 4
        nch_out2 = 32
        self.conv12 = nn.Conv3d(sum_chans, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = nn.Conv3d(sum_chans, 2 * Q2 * nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn22 = nn.BatchNorm3d(nch_out2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Block 3: Quadratic + Symmetric Cubic =====
        Q3 = 4
        Q3c = 2  # Cubic uses fewer rank components (parameter efficiency)
        nch_out3 = 64
        # Linear
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        # Quadratic (2nd order)
        self.conv23 = nn.Conv3d(nch_out2, 2 * Q3 * nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn23 = nn.BatchNorm3d(nch_out3)
        # Cubic (3rd order, symmetric a²·b)
        self.conv33 = nn.Conv3d(nch_out2, 2 * Q3c * nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn33 = nn.BatchNorm3d(nch_out3)
        self.scale3 = nn.Parameter(torch.zeros(1))  # Zero-init gate: starts as pure quadratic

        # ===== Block 4: Quadratic + Symmetric Cubic =====
        Q4 = 4
        Q4c = 2
        nch_out4 = 96
        # Linear
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        # Quadratic (2nd order)
        self.conv24 = nn.Conv3d(nch_out3, 2 * Q4 * nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn24 = nn.BatchNorm3d(nch_out4)
        # Cubic (3rd order, symmetric a²·b)
        self.conv34 = nn.Conv3d(nch_out3, 2 * Q4c * nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn34 = nn.BatchNorm3d(nch_out4)
        self.scale4 = nn.Parameter(torch.zeros(1))  # Zero-init gate
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x, activation=False):
        # ===== Block 1: Multi-kernel Quadratic =====
        Q1 = 4; nch1 = 8

        # Linear: 3 parallel convolutions → concat → BN
        x11 = self.bn11(torch.cat((self.conv11_5(x), self.conv11_3(x), self.conv11_1(x)), 1))

        # Quadratic: 3 parallel interactions → concat → BN
        x21_5 = _volterra_quadratic(self.conv21_5(x), Q1, nch1)
        x21_3 = _volterra_quadratic(self.conv21_3(x), Q1, nch1)
        x21_1 = _volterra_quadratic(self.conv21_1(x), Q1, nch1)
        x21 = self.bn21(torch.cat((x21_5, x21_3, x21_1), 1))

        x = self.pool1(x11 + x21)

        # ===== Block 2: Quadratic only =====
        x12 = self.bn12(self.conv12(x))
        x22 = self.bn22(_volterra_quadratic(self.conv22(x), Q=4, nch_out=32))
        x = self.pool2(x12 + x22)

        # ===== Block 3: Quadratic + Cubic =====
        x13 = self.bn13(self.conv13(x))
        x23 = self.bn23(_volterra_quadratic(self.conv23(x), Q=4, nch_out=64))
        x33 = self.bn33(_volterra_cubic_symmetric(self.conv33(x), Q=2, nch_out=64))
        x = x13 + x23 + self.scale3 * x33  # No pool on block 3

        # ===== Block 4: Quadratic + Cubic =====
        x14 = self.bn14(self.conv14(x))
        x24 = self.bn24(_volterra_quadratic(self.conv24(x), Q=4, nch_out=96))
        x34 = self.bn34(_volterra_cubic_symmetric(self.conv34(x), Q=2, nch_out=96))
        x = self.pool4(x14 + x24 + self.scale4 * x34)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """Returns all trainable parameters of the model."""
    for param in model.parameters():
        if param.requires_grad:
            yield param


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VNN(num_classes=101)
    outputs = net(inputs)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    # Show cubic gate values (should start at 0)
    print(f"Cubic gates: scale3={net.scale3.item():.4f}, scale4={net.scale4.item():.4f}")