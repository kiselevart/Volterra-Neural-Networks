"""
3D VNN Fusion Head with 2nd + 3rd order Volterra interactions.

Single-block fusion classifier that takes concatenated RGB + Optical Flow
features and produces class logits. Uses both quadratic and symmetric cubic
Volterra interactions for maximum expressiveness at the decision layer.
"""

import torch
import torch.nn as nn


def _volterra_quadratic(x_conv, Q, nch_out):
    """Vectorized 2nd-order Volterra: split → multiply → sum Q groups."""
    mid = Q * nch_out
    left = x_conv[:, :mid]
    right = x_conv[:, mid:]
    product = left * right
    shape = product.shape
    spatial = shape[2:]
    return product.view(shape[0], Q, nch_out, *spatial).sum(dim=1)


def _volterra_cubic_symmetric(x_conv, Q, nch_out):
    """Symmetric 3rd-order Volterra: a² · b decomposition."""
    mid = Q * nch_out
    a = x_conv[:, :mid]
    b = x_conv[:, mid:]
    product = (a * a) * b
    shape = product.shape
    spatial = shape[2:]
    return product.view(shape[0], Q, nch_out, *spatial).sum(dim=1)


class VNN_F(nn.Module):
    """Fusion classification head with Quadratic + Symmetric Cubic Volterra.

    Architecture:
        Single VNN block (Quadratic + Cubic) → Pool → Dropout → FC classifier

    The cubic term uses the symmetric CP decomposition: a² · b
    with zero-initialized gating for stable training warmup.
    """

    def __init__(self, num_classes, num_ch=3, pretrained=False):
        super(VNN_F, self).__init__()

        # ===== Block 1: Quadratic + Symmetric Cubic =====
        Q1 = 2; Q1c = 2
        nch_out1 = 256

        # Linear path
        self.conv11 = nn.Conv3d(num_ch, nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(nch_out1)
        # Quadratic path
        self.conv21 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn21 = nn.BatchNorm3d(nch_out1)
        # Cubic path (symmetric a²·b)
        self.conv31 = nn.Conv3d(num_ch, 2 * Q1c * nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn31 = nn.BatchNorm3d(nch_out1)
        self.scale1 = nn.Parameter(torch.zeros(1))  # Zero-init gate

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Classifier =====
        self.fc8 = nn.Linear(12544, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()

    def forward(self, x, activation=False):
        # ===== Block 1: Quadratic + Cubic =====
        x11 = self.bn11(self.conv11(x))
        x21 = self.bn21(_volterra_quadratic(self.conv21(x), Q=2, nch_out=256))
        x31 = self.bn31(_volterra_cubic_symmetric(self.conv31(x), Q=2, nch_out=256))
        x = self.pool1(x11 + x21 + self.scale1 * x31)

        # ===== Classifier =====
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """Returns all trainable parameters except the classifier."""
    skip = {id(p) for p in model.fc8.parameters()}
    for param in model.parameters():
        if param.requires_grad and id(param) not in skip:
            yield param


def get_10x_lr_params(model):
    """Returns the classifier (fc8) parameters for higher learning rate."""
    for param in model.fc8.parameters():
        if param.requires_grad:
            yield param


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VNN_F(num_classes=101)
    outputs = net(inputs)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
    print(f"Cubic gate (scale1): {net.scale1.item():.4f}")