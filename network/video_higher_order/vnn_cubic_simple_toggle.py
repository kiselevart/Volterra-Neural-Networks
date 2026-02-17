import torch
import torch.nn as nn

def _volterra_quadratic(x_conv, Q, nch_out):
    """
    2nd-order Volterra: h2(i,j) ≈ Σ a_q(i)·b_q(j)
    Splits channels into 2 halves (left, right) and multiplies them.
    """
    # x_conv shape: [B, 2*Q*C, T, H, W]
    mid = Q * nch_out
    left = x_conv[:, :mid]
    right = x_conv[:, mid:]

    # Element-wise product (interaction)
    product = left * right  # [B, Q*C, ...]

    # Sum over Rank (Q) dimension to get [B, C, ...]
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)

def _volterra_cubic_general(x_conv, Q, nch_out):
    """
    3rd-order General Volterra: h3(i,j,k) ≈ Σ a_q(i)·b_q(j)·c_q(k)
    Splits channels into 3 parts (a, b, c) and multiplies them.
    This allows for interactions between three distinct features.
    """
    # x_conv shape: [B, 3*Q*C, T, H, W]
    # We expect 3 parts: a, b, c

    # Split into 3 equal chunks along the channel dimension
    a, b, c = torch.chunk(x_conv, 3, dim=1)

    # Three-way Interaction (Element-wise)
    product = a * b * c  # [B, Q*C, T, H, W]

    # Sum over Rank (Q) dimension to get [B, C, ...]
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)

class VolterraBlock(nn.Module):
    """
    A single block that sums Linear, Quadratic, and (optional) Cubic paths.
    """
    def __init__(self, in_ch, out_ch, stride=1, use_cubic=False, Q=4, Qc=2):
        super().__init__()
        self.use_cubic = use_cubic
        self.out_ch = out_ch
        self.Q = Q
        self.Qc = Qc

        # 1. Linear Path (1st Order)
        self.conv_lin = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn_lin = nn.BatchNorm3d(out_ch)

        # 2. Quadratic Path (2nd Order)
        # Needs 2*Q*C output channels for CP decomposition (Left * Right)
        # Kernel size > 1 ensures channel mixing before interaction
        self.conv_quad = nn.Conv3d(in_ch, 2 * Q * out_ch, kernel_size=3, padding=1)
        self.bn_quad = nn.BatchNorm3d(out_ch)

        # 3. Cubic Path (3rd Order) - Optional
        if self.use_cubic:
            # Needs 3*Qc*C output channels for General decomposition (a * b * c)
            # CHANGED: Multiplier is now 3, not 2
            self.conv_cubic = nn.Conv3d(in_ch, 3 * Qc * out_ch, kernel_size=3, padding=1)
            self.bn_cubic = nn.BatchNorm3d(out_ch)

            # Zero-initialized gate to ensure smooth warmup.
            # Starts effectively as a quadratic model.
            self.cubic_gate = nn.Parameter(torch.zeros(1))

        # Pooling (optional, based on stride)
        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

    def forward(self, x):
        # Linear Term
        out = self.bn_lin(self.conv_lin(x))

        # Quadratic Term
        x_q = self.conv_quad(x)
        out += self.bn_quad(_volterra_quadratic(x_q, self.Q, self.out_ch))

        # Cubic Term
        if self.use_cubic:
            x_c = self.conv_cubic(x)
            # CHANGED: Using general cubic function
            cubic_term = _volterra_cubic_general(x_c, self.Qc, self.out_ch)
            out += self.bn_cubic(cubic_term) * self.cubic_gate

        return self.pool(out)

class SimpleVNN(nn.Module):
    def __init__(self, use_cubic=True):
        super().__init__()

        # Backbone Configuration
        # (in_ch, out_ch, stride)
        config = [
            (3,  32, 2),  # Block 1
            (32, 64, 2),  # Block 2
            (64, 64, 1),  # Block 3 (No pool, deeper features)
            (64, 96, 2)   # Block 4
        ]

        self.layers = nn.ModuleList()
        for in_c, out_c, s in config:
            self.layers.append(VolterraBlock(in_c, out_c, stride=s, use_cubic=use_cubic))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Testing the Isolation ---
if __name__ == "__main__":
    # Test 1: Without Cubic (Baseline)
    model_baseline = SimpleVNN(use_cubic=False)
    print(f"Baseline Params: {sum(p.numel() for p in model_baseline.parameters()):,}")

    # Test 2: With Cubic (Experiment - General Form)
    model_cubic = SimpleVNN(use_cubic=True)
    print(f"Cubic Params:    {sum(p.numel() for p in model_cubic.parameters()):,}")

    # Input check
    x = torch.randn(1, 3, 16, 112, 112)
    y = model_cubic(x)
    print(f"Output Shape:    {y.shape}")
