import torch
import torch.nn as nn
import math

class VNN_ResBlock_Ortho(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, Q=2):
        super(VNN_ResBlock_Ortho, self).__init__()
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Q = Q
        
        # 1. Linear Stream (Degree 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Quadratic Stream (Degree 2)
        mid_channels = out_channels * Q
        
        # --- FIXED: Use Depthwise Convolution for expansion ---
        # We project in_channels to mid_channels * 2 using Depthwise + Pointwise
        self.conv2_expand = nn.Sequential(
            # Depthwise: Spatial filtering per channel
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            # Pointwise: Linear combination to reach interaction width
            nn.Conv2d(in_channels, mid_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels * 2)
        )
        
        # Projection 1x1 to mix the quadratic terms back to out_channels
        self.conv2_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2_proj = nn.BatchNorm2d(out_channels)
        
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
        
        # --- Quadratic Term ---
        # Expanded via Depthwise Separable block
        quad_features = self.conv2_expand(x)
        
        # Split for interaction
        left, right = torch.chunk(quad_features, 2, dim=1)
        
        # Orthogonal Interaction: 4x^2 - 2
        interaction = 4.0 * (left * right) - 2.0
        
        # Project back
        quad_out = self.bn2_proj(self.conv2_proj(interaction))
        
        # --- Summation & Residual ---
        out = linear_out + quad_out + self.shortcut(x)
        out = self.relu(out)
        
        return out

class ResVNN_Ortho_CIFAR(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[2, 2, 2, 2], Q=2):
        super(ResVNN_Ortho_CIFAR, self).__init__()
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride, Q):
        layers = []
        # First block handles stride and channel change
        layers.append(VNN_ResBlock_Ortho(self.in_channels, out_channels, stride, Q))
        self.in_channels = out_channels
        # Subsequent blocks are identity-sized
        for _ in range(1, blocks):
            layers.append(VNN_ResBlock_Ortho(out_channels, out_channels, stride=1, Q=Q))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
