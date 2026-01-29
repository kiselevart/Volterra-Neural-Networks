import torch
import torch.nn as nn
import math

class VNN_ResBlock_Ortho_Optimized(nn.Module):
    """Optimized for AMD GPU - fused operations, fewer kernel launches"""
    def __init__(self, in_channels, out_channels, stride=1, Q=2):
        super(VNN_ResBlock_Ortho_Optimized, self).__init__()
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Q = Q
        
        # 1. Linear Stream (Degree 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Quadratic Stream (Degree 2) - OPTIMIZED
        mid_channels = out_channels * Q
        
        # Fused expansion: Single conv instead of depthwise + pointwise
        # This reduces kernel launches significantly
        self.conv2_expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * 2, kernel_size=3, stride=stride, 
                      padding=1, groups=1, bias=False),  # Standard conv, not depthwise
            nn.BatchNorm2d(mid_channels * 2)
        )
        
        # Single projection instead of separate ops
        self.conv2_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection handling
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # --- Linear Term ---
        linear_out = self.bn1(self.conv1(x))
        
        # --- Quadratic Term (Fused) ---
        quad_features = self.conv2_expand(x)
        
        # Split for interaction
        left, right = torch.chunk(quad_features, 2, dim=1)
        
        # Fused: 4*left*right - 2 (single operation)
        interaction = torch.addcmul(torch.full_like(left, -2.0), left, right, value=4.0)
        
        # Single projection
        quad_out = self.conv2_proj(interaction)
        
        # --- Summation & Residual (Fused) ---
        out = linear_out + quad_out + self.shortcut(x)
        out = self.relu(out)
        
        return out


class ResVNN_Ortho_CIFAR_Optimized(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[2, 2, 2, 2], Q=2):
        super(ResVNN_Ortho_CIFAR_Optimized, self).__init__()
        self.in_channels = 64
        
        # Initial Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stages
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, Q=Q)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, Q=Q)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, Q=Q)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, Q=Q)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride=1, Q=2):
        layers = []
        layers.append(VNN_ResBlock_Ortho_Optimized(self.in_channels, out_channels, stride=stride, Q=Q))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(VNN_ResBlock_Ortho_Optimized(out_channels, out_channels, stride=1, Q=Q))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
