"""
Shared Volterra interaction primitives for 3D video models.

Volterra series expansion up to 3rd order:
    y = h0 + Σ h1(i)·x(i) + Σ h2(i,j)·x(i)·x(j) + Σ h3(i,j,k)·x(i)·x(j)·x(k)

Decomposition strategies:
    2nd-order (quadratic):  CP factorization    → left · right   (2·Q·C channels)
    3rd-order symmetric:    Tied CP factors      → a² · b         (2·Q·C channels)
    3rd-order general:      Independent factors  → a · b · c      (3·Q·C channels)
"""

import torch
import torch.nn as nn


def volterra_quadratic(x_conv, Q, nch_out):
    """Vectorized 2nd-order Volterra interaction (CP decomposition).

    h2(i,j) ≈ Σ_q a_q(i)·b_q(j)
    Splits channels into left/right halves and element-wise multiplies.

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
    
    # Quadratic product: (a * b)
    product = left * right  # [B, Q*C, T, H, W]
    
    # Stability: Clamp before summation to prevent explosion
    product = torch.clamp(product, min=-10.0, max=10.0)
    
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def volterra_cubic_symmetric(x_conv, Q, nch_out):
    """Symmetric 3rd-order Volterra interaction (a²·b decomposition).

    h3(i,j,k) ≈ Σ_q a_q(i)·a_q(j)·b_q(k)
    Two factors are tied → only 2·Q·C channels needed (same as quadratic).

    The a² term acts as an energy/magnitude detector (always ≥ 0),
    while b modulates sign and scale — a learned nonlinear gating mechanism.

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
    
    # Cubic product: (a² * b)
    product = (a * a) * b  # [B, Q*C, T, H, W]
    
    # Stability: Clamp before summation
    product = torch.clamp(product, min=-10.0, max=10.0)
    
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def volterra_cubic_general(x_conv, Q, nch_out):
    """General 3rd-order Volterra interaction (a·b·c decomposition).

    h3(i,j,k) ≈ Σ_q a_q(i)·b_q(j)·c_q(k)
    Three independent factors → 3·Q·C channels needed.
    More expressive than symmetric, but more parameters.

    Args:
        x_conv: Tensor [B, 3*Q*C, T, H, W] from cubic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    a, b, c = torch.chunk(x_conv, 3, dim=1)
    
    # Cubic product: (a * b * c)
    product = a * b * c  # [B, Q*C, T, H, W]
    
    # Stability: Clamp before summation
    product = torch.clamp(product, min=-10.0, max=10.0)
    
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def init_vnn_weights(module):
    """Standard VNN weight initialization.

    Conv3d: Kaiming normal (fan_in, suitable for Volterra nonlinearity).
    BatchNorm3d: weight=1, bias=0.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
