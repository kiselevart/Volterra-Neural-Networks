"""
3D VNN Backbone (Complex/Deep variant) with 2nd + 3rd order Volterra interactions.

7-block deep architecture for video feature extraction.
Blocks 1-4: Linear + Quadratic (2nd order)
Blocks 6-7-5: Linear + Quadratic + Symmetric Cubic (3rd order)

See vnn_rgb_of_highQ.py for the mathematical details of the Volterra decomposition.
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


class VNN(nn.Module):
    """3D VNN Backbone (Complex) with 2nd + 3rd order Volterra interactions.

    7-block architecture:
        Block 1: Multi-kernel → Quadratic → Pool
        Block 2: Quadratic → Pool
        Block 3: Quadratic (no pool)
        Block 4: Quadratic (no pool)
        Block 6: Quadratic + Cubic → Pool
        Block 7: Quadratic + Cubic (no pool)
        Block 5: Quadratic + Cubic → Pool → FC classifier

    This module includes a classifier head (fc8) and returns logits.
    """

    def __init__(self, num_classes=400, num_ch=3, pretrained=False):
        super(VNN, self).__init__()

        # ===== Block 1: Multi-kernel, Quadratic only =====
        Q1 = 4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8
        sum_chans = nch_out1_5 + nch_out1_3 + nch_out1_1  # 24

        self.conv11_5 = nn.Conv3d(num_ch, nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_3 = nn.Conv3d(num_ch, nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_1 = nn.Conv3d(num_ch, nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn11 = nn.BatchNorm3d(sum_chans)

        self.conv21_5 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_3 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_1 = nn.Conv3d(num_ch, 2 * Q1 * nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn21 = nn.BatchNorm3d(sum_chans)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Block 2: Quadratic only =====
        Q2 = 4; nch_out2 = 32
        self.conv12 = nn.Conv3d(sum_chans, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = nn.Conv3d(sum_chans, 2 * Q2 * nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn22 = nn.BatchNorm3d(nch_out2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Block 3: Quadratic only =====
        Q3 = 4; nch_out3 = 64
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        self.conv23 = nn.Conv3d(nch_out2, 2 * Q3 * nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn23 = nn.BatchNorm3d(nch_out3)

        # ===== Block 4: Quadratic only =====
        Q4 = 4; nch_out4 = 96
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        self.conv24 = nn.Conv3d(nch_out3, 2 * Q4 * nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn24 = nn.BatchNorm3d(nch_out4)

        # ===== Block 6: Quadratic + Symmetric Cubic =====
        Q6 = 4; Q6c = 2; nch_out6 = 128
        self.conv16 = nn.Conv3d(nch_out4, nch_out6, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn16 = nn.BatchNorm3d(nch_out6)
        self.conv26 = nn.Conv3d(nch_out4, 2 * Q6 * nch_out6, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn26 = nn.BatchNorm3d(nch_out6)
        # Cubic (symmetric a²·b)
        self.conv36 = nn.Conv3d(nch_out4, 2 * Q6c * nch_out6, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn36 = nn.BatchNorm3d(nch_out6)
        self.scale6 = nn.Parameter(torch.zeros(1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Block 7: Quadratic + Symmetric Cubic =====
        Q7 = 4; Q7c = 2; nch_out7 = 256
        self.conv17 = nn.Conv3d(nch_out6, nch_out7, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn17 = nn.BatchNorm3d(nch_out7)
        self.conv27 = nn.Conv3d(nch_out6, 2 * Q7 * nch_out7, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn27 = nn.BatchNorm3d(nch_out7)
        # Cubic (symmetric a²·b)
        self.conv37 = nn.Conv3d(nch_out6, 2 * Q7c * nch_out7, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn37 = nn.BatchNorm3d(nch_out7)
        self.scale7 = nn.Parameter(torch.zeros(1))

        # ===== Block 5: Quadratic + Symmetric Cubic =====
        Q5 = 2; Q5c = 2; nch_out5 = 256
        self.conv15 = nn.Conv3d(nch_out7, nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn15 = nn.BatchNorm3d(nch_out5)
        self.conv25 = nn.Conv3d(nch_out7, 2 * Q5 * nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn25 = nn.BatchNorm3d(nch_out5)
        # Cubic (symmetric a²·b)
        self.conv35 = nn.Conv3d(nch_out7, 2 * Q5c * nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn35 = nn.BatchNorm3d(nch_out5)
        self.scale5 = nn.Parameter(torch.zeros(1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ===== Classifier =====
        self.fc8 = nn.Linear(12544, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x, activation=False):
        # ===== Block 1: Multi-kernel Quadratic =====
        Q1 = 4; nch1 = 8
        x11 = self.bn11(torch.cat((self.conv11_5(x), self.conv11_3(x), self.conv11_1(x)), 1))
        x21_5 = _volterra_quadratic(self.conv21_5(x), Q1, nch1)
        x21_3 = _volterra_quadratic(self.conv21_3(x), Q1, nch1)
        x21_1 = _volterra_quadratic(self.conv21_1(x), Q1, nch1)
        x21 = self.bn21(torch.cat((x21_5, x21_3, x21_1), 1))
        x = self.pool1(x11 + x21)

        # ===== Block 2: Quadratic =====
        x12 = self.bn12(self.conv12(x))
        x22 = self.bn22(_volterra_quadratic(self.conv22(x), 4, 32))
        x = self.pool2(x12 + x22)

        # ===== Block 3: Quadratic (no pool) =====
        x13 = self.bn13(self.conv13(x))
        x23 = self.bn23(_volterra_quadratic(self.conv23(x), 4, 64))
        x = x13 + x23

        # ===== Block 4: Quadratic (no pool) =====
        x14 = self.bn14(self.conv14(x))
        x24 = self.bn24(_volterra_quadratic(self.conv24(x), 4, 96))
        x = x14 + x24

        # ===== Block 6: Quadratic + Cubic =====
        x16 = self.bn16(self.conv16(x))
        x26 = self.bn26(_volterra_quadratic(self.conv26(x), 4, 128))
        x36 = self.bn36(_volterra_cubic_symmetric(self.conv36(x), 2, 128))
        x = self.pool6(x16 + x26 + self.scale6 * x36)

        # ===== Block 7: Quadratic + Cubic (no pool) =====
        x17 = self.bn17(self.conv17(x))
        x27 = self.bn27(_volterra_quadratic(self.conv27(x), 4, 256))
        x37 = self.bn37(_volterra_cubic_symmetric(self.conv37(x), 2, 256))
        x = x17 + x27 + self.scale7 * x37

        # ===== Block 5: Quadratic + Cubic =====
        x15 = self.bn15(self.conv15(x))
        x25 = self.bn25(_volterra_quadratic(self.conv25(x), 2, 256))
        x35 = self.bn35(_volterra_cubic_symmetric(self.conv35(x), 2, 256))
        x = self.pool5(x15 + x25 + self.scale5 * x35)

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
    """Returns all trainable parameters of the model."""
    for param in model.parameters():
        if param.requires_grad:
            yield param