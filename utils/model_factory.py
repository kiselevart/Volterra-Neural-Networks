import torch
import torch.nn as nn
import torchvision.models as models

from network.cifar.vnn_cifar import VNN_CIFAR
from network.cifar_ortho.res_vnn_ortho import ResVNN_Ortho_CIFAR

from network.video import backbone_4block as vnn_rgb_ho
from network.video import backbone_7block as vnn_complex_ho
from network.video import backbone_cubic_toggle as vnn_cubic_toggle
from network.video import fusion_head as vnn_fusion_ho
from network.video.blocks import TemporalSE


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
        if args.model == "vnn_rgb_ho":
            # Higher-order (cubic) RGB backbone + cubic fusion head
            class VideoVNN_HO(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = vnn_rgb_ho.VNN(num_ch=3)
                    self.head = vnn_fusion_ho.VNN_F(num_classes=num_classes, num_ch=96)

                def forward(self, x):
                    return self.head(self.backbone(x))

                def get_1x_lr_params(self):
                    p = list(self.backbone.parameters())
                    p += list(vnn_fusion_ho.get_1x_lr_params(self.head))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_ho.get_10x_lr_params(self.head))

            net = VideoVNN_HO(num_classes=args.num_classes)

        elif args.model == "vnn_fusion_ho":
            # Higher-order (cubic) two-stream fusion: RGB + Flow
            class VideoVNNFusion_HO(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_ho.VNN(num_ch=3, Q=4)
                    self.model_of = vnn_rgb_ho.VNN(num_ch=2, Q=2)   # smaller Q: flow is noisier
                    self.temporal_se_rgb = TemporalSE(num_frames=2)  # clip_len=16 → T/8=2
                    self.temporal_se_of = TemporalSE(num_frames=2)
                    self.model_fuse = vnn_fusion_ho.VNN_F(
                        num_classes=num_classes, num_ch=288
                    )

                def forward(self, x):
                    rgb, flow = x
                    out_rgb = self.temporal_se_rgb(self.model_rgb(rgb))
                    out_of = self.temporal_se_of(self.model_of(flow))
                    cross = out_rgb * out_of
                    return self.model_fuse(torch.cat((out_rgb, out_of, cross), 1))

                def get_1x_lr_params(self):
                    p = list(self.model_rgb.parameters())
                    p += list(self.model_of.parameters())
                    p += list(vnn_fusion_ho.get_1x_lr_params(self.model_fuse))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_ho.get_10x_lr_params(self.model_fuse))

            net = VideoVNNFusion_HO(num_classes=args.num_classes)

        elif args.model == "vnn_complex_ho":
            net = vnn_complex_ho.VNN(num_classes=args.num_classes, num_ch=3)

        elif args.model == "vnn_cubic_simple_toggle":
            class VideoVNNCubicToggle(nn.Module):
                def __init__(self, num_classes, use_cubic=True):
                    super().__init__()
                    self.backbone = vnn_cubic_toggle.SimpleVNN(use_cubic=use_cubic)
                    self.head = vnn_fusion_ho.VNN_F(num_classes=num_classes, num_ch=96)

                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)

                def get_1x_lr_params(self):
                    p = list(self.backbone.parameters())
                    p += list(vnn_fusion_ho.get_1x_lr_params(self.head))
                    return p

                def get_10x_lr_params(self):
                    return list(vnn_fusion_ho.get_10x_lr_params(self.head))

            net = VideoVNNCubicToggle(
                num_classes=args.num_classes, use_cubic=not args.disable_cubic
            )

        else:
            raise ValueError(f"Unknown Video model: {args.model}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    return net.to(device)
