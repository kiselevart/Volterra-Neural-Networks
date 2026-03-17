import torch
import torch.nn as nn
import torchvision.models as models

from network.cifar.vnn_cifar import VNN_CIFAR
from network.cifar_ortho.res_vnn_ortho import ResVNN_Ortho_CIFAR

# Imports from existing codebase
from network.video import (
    vnn_fusion_highQ,
    vnn_fusion_highQ_v3,
    vnn_rgb_of_highQ,
    vnn_rgb_of_highQ_v3,
)
from network.video_higher_order import backbone_4block as vnn_rgb_ho
from network.video_higher_order import backbone_7block as vnn_complex_ho
from network.video_higher_order import backbone_cubic_toggle as vnn_cubic_toggle
from network.video_higher_order import fusion_head as vnn_fusion_ho


def get_model(args, device):
    print(f"==> Building model: {args.model}")

    if args.task == "cifar":
        if args.model == "vnn_simple":
            net = VNN_CIFAR(num_classes=args.num_classes)
        elif args.model == "vnn_ortho":
            # Default to [2,2,2,2] blocks for ResNet18 equivalence
            net = ResVNN_Ortho_CIFAR(
                num_classes=args.num_classes, num_blocks=[2, 2, 2, 2], Q=args.Q
            )
        elif args.model == "resnet18":
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, args.num_classes)
            # Adapt for CIFAR-10 size
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        else:
            raise ValueError(f"Unknown CIFAR model: {args.model}")

    elif args.task == "video":
        if args.model == "vnn_rgb":
            # RGB Backbone -> Classifier
            # Note: We need to wrap them into a single module for simplicity in the training loop
            class VideoVNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = vnn_rgb_of_highQ.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.head = vnn_fusion_highQ.VNN_F(
                        num_classes=num_classes, num_ch=96, pretrained=False
                    )

                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)

                def get_1x_lr_params(self):
                    return list(
                        vnn_rgb_of_highQ.get_1x_lr_params(self.backbone)
                    ) + list(self.head.parameters())

            net = VideoVNN(num_classes=args.num_classes)

        elif args.model == "vnn_fusion":
            # RGB + Flow streams
            # This requires a more complex forward pass handling inputs=(rgb, flow)
            # For this factory, we return the container, but the training loop handles the tuple unpacking
            class VideoVNNFusion(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_of_highQ.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.model_of = vnn_rgb_of_highQ.VNN(
                        num_classes=num_classes, num_ch=2, pretrained=False
                    )
                    self.model_fuse = vnn_fusion_highQ.VNN_F(
                        num_classes=num_classes, num_ch=192, pretrained=False
                    )

                def forward(self, x):
                    # Expects x to be (rgb, flow)
                    rgb, flow = x
                    out_rgb = self.model_rgb(rgb)
                    out_of = self.model_of(flow)
                    out_fuse = self.model_fuse(torch.cat((out_rgb, out_of), 1))
                    return out_fuse

                def get_1x_lr_params(self):
                    p = []
                    p += list(vnn_rgb_of_highQ.get_1x_lr_params(self.model_rgb))
                    p += list(vnn_rgb_of_highQ.get_1x_lr_params(self.model_of))
                    p += list(vnn_fusion_highQ.get_1x_lr_params(self.model_fuse))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_highQ.get_10x_lr_params(self.model_fuse))

            net = VideoVNNFusion(num_classes=args.num_classes)

        elif args.model == "vnn_fusion_v3":
            # Clean v3 fusion path mirroring original functional implementation
            class VideoVNNFusionV3(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_of_highQ_v3.VNN(
                        num_classes=num_classes, num_ch=3, pretrained=False
                    )
                    self.model_of = vnn_rgb_of_highQ_v3.VNN(
                        num_classes=num_classes, num_ch=2, pretrained=False
                    )
                    self.model_fuse = vnn_fusion_highQ_v3.VNN_F(
                        num_classes=num_classes, num_ch=192, pretrained=False
                    )

                def forward(self, x):
                    rgb, flow = x
                    out_rgb = self.model_rgb(rgb)
                    out_of = self.model_of(flow)
                    return self.model_fuse(torch.cat((out_rgb, out_of), 1))

                def get_1x_lr_params(self):
                    p = []
                    p += list(vnn_rgb_of_highQ_v3.get_1x_lr_params(self.model_rgb))
                    p += list(vnn_rgb_of_highQ_v3.get_1x_lr_params(self.model_of))
                    p += list(vnn_fusion_highQ_v3.get_1x_lr_params(self.model_fuse))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_highQ_v3.get_10x_lr_params(self.model_fuse))

            net = VideoVNNFusionV3(num_classes=args.num_classes)

        elif args.model == "vnn_rgb_ho":
            # Higher-order (cubic) RGB backbone + cubic fusion head
            class VideoVNN_HO(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = vnn_rgb_ho.VNN(num_classes=num_classes, num_ch=3)
                    self.head = vnn_fusion_ho.VNN_F(num_classes=num_classes, num_ch=96)

                def forward(self, x):
                    return self.head(self.backbone(x))

            net = VideoVNN_HO(num_classes=args.num_classes)

        elif args.model == "vnn_fusion_ho":
            # Higher-order (cubic) two-stream fusion: RGB + Flow
            class VideoVNNFusion_HO(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_ho.VNN(num_classes=num_classes, num_ch=3)
                    self.model_of = vnn_rgb_ho.VNN(num_classes=num_classes, num_ch=2)
                    self.model_fuse = vnn_fusion_ho.VNN_F(
                        num_classes=num_classes, num_ch=192
                    )

                def forward(self, x):
                    rgb, flow = x
                    out_rgb = self.model_rgb(rgb)
                    out_of = self.model_of(flow)
                    return self.model_fuse(torch.cat((out_rgb, out_of), 1))

            net = VideoVNNFusion_HO(num_classes=args.num_classes)

        elif args.model == "vnn_complex_ho":
            # Higher-order (cubic) deep 7-block complex backbone (includes classifier)
            net = vnn_complex_ho.VNN(num_classes=args.num_classes, num_ch=3)

        elif args.model == "vnn_cubic_simple_toggle":
            # Simple 4-block Volterra backbone with optional cubic toggle + cubic fusion classifier
            class VideoVNNCubicToggle(nn.Module):
                def __init__(self, num_classes, use_cubic=True):
                    super().__init__()
                    self.backbone = vnn_cubic_toggle.SimpleVNN(use_cubic=use_cubic)
                    self.head = vnn_fusion_ho.VNN_F(num_classes=num_classes, num_ch=96)

                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)

                def get_1x_lr_params(self):
                    p = []
                    p += list(self.backbone.parameters())
                    p += list(vnn_fusion_ho.get_1x_lr_params(self.head))
                    p += list(vnn_fusion_ho.get_10x_lr_params(self.head))
                    return p

            net = VideoVNNCubicToggle(
                num_classes=args.num_classes, use_cubic=not args.disable_cubic
            )

        else:
            raise ValueError(f"Unknown Video model: {args.model}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    return net.to(device)
