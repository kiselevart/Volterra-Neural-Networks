import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class VolterraExpansion(nn.Module):
    """
    Encapsulates the Quadratic Stream: 
    Input -> Depthwise (Space) -> Pointwise (Channel Mixing) -> Chebyshev Interaction
    """
    def __init__(self, in_channels, out_channels, stride=1, Q=2):
        super().__init__()
        mid_channels = out_channels * Q
        
        # 1. Spatial Mixing (Depthwise)
        self.dw = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)),
            nn.BatchNorm2d(in_channels),
            nn.Tanh() # Tanh crucial for stability
        )
        
        # 2. Channel Mixing (Pointwise) - Expands to 2x for splitting
        self.pw = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, mid_channels * 2, 1, bias=False)),
            nn.BatchNorm2d(mid_channels * 2),
            nn.Tanh()
        )
        
        # 3. Projection back to output
        self.proj = spectral_norm(nn.Conv2d(mid_channels, out_channels, 1, bias=False))
        
        # 4. The "Correction" BatchNorm (replaces the manual -2 subtraction)
        self.bn_proj = nn.BatchNorm2d(out_channels)
        
        # 5. Gating Parameter
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1)) # Init to 0 for "Identity" start

        self._init_volterra()

    def _init_volterra(self):
        # Orthogonal init for Spectral layers is best
        for m in [self.dw[0], self.pw[0], self.proj]:
            nn.init.orthogonal_(m.weight)
        
        # "Fixup" Initialization:
        # We init the final BN weight/bias to correct the Chebyshev shift
        # The theoretical Chebyshev mean is -2, so we set bias=+2 to cancel it out initially.
        nn.init.constant_(self.bn_proj.weight, 1e-3) # Start small
        nn.init.constant_(self.bn_proj.bias, 2.0)    # Counteract the -2 shift

    def forward(self, x):
        # Streamlined Forward Pass
        x = self.dw(x)
        x = self.pw(x)
        
        # Low-Rank Bilinear Pooling
        left, right = torch.chunk(x, 2, dim=1)
        
        # Chebyshev T2 Interaction: 4xy - 2
        # We calculate 4xy here. The "-2" is handled by the bn_proj.bias initialized to +2
        interaction = 4.0 * (left * right) 
        
        # Project and Gate
        return self.scale * self.bn_proj(self.proj(interaction))


class VNN_ResBlock_Ortho(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, Q=2):
        super().__init__()
        
        # Linear Branch (Standard ResNet)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Quadratic Branch (Volterra)
        self.quad = VolterraExpansion(in_channels, out_channels, stride, Q)
        
        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        return self.relu(
            self.bn1(self.conv1(x)) + self.quad(x) + self.shortcut(x)
        )


class ResVNN_Ortho_CIFAR(nn.Module):
    def __init__(self, num_classes=10, layers=[2, 2, 2, 2], Q=2):
        super().__init__()
        self.in_channels = 64
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stage1 = self._make_layer(64,  layers[0], stride=1, Q=Q)
        self.stage2 = self._make_layer(128, layers[1], stride=2, Q=Q)
        self.stage3 = self._make_layer(256, layers[2], stride=2, Q=Q)
        self.stage4 = self._make_layer(512, layers[3], stride=2, Q=Q)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Standard Init for Linear Layers (Volterra layers init themselves)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'weight_orig'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, channels, blocks, stride, Q):
        layers = []
        layers.append(VNN_ResBlock_Ortho(self.in_channels, channels, stride, Q))
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(VNN_ResBlock_Ortho(channels, channels, 1, Q))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x