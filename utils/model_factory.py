import torch
import torch.nn as nn
import torchvision.models as models

# Imports from existing codebase
from network.fusion import vnn_rgb_of_highQ, vnn_fusion_highQ
from network.cifar.vnn_cifar import VNN_CIFAR
from network.cifar_ortho.res_vnn_ortho import ResVNN_Ortho_CIFAR

def get_model(args, device):
    print(f"==> Building model: {args.model}")

    if args.task == 'cifar':
        if args.model == 'vnn_simple':
            net = VNN_CIFAR(num_classes=args.num_classes)
        elif args.model == 'vnn_ortho':
            # Default to [2,2,2,2] blocks for ResNet18 equivalence
            net = ResVNN_Ortho_CIFAR(num_classes=args.num_classes, num_blocks=[2, 2, 2, 2], Q=args.Q)
        elif args.model == 'resnet18':
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, args.num_classes)
            # Adapt for CIFAR-10 size
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        else:
            raise ValueError(f"Unknown CIFAR model: {args.model}")

    elif args.task == 'video':
        if args.model == 'vnn_rgb':
            # RGB Backbone -> Classifier
            # Note: We need to wrap them into a single module for simplicity in the training loop
            class VideoVNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.backbone = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
                    self.head = vnn_fusion_highQ.VNN_F(num_classes=num_classes, num_ch=96, pretrained=False)
                
                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)
                
                def get_1x_lr_params(self):
                    return list(vnn_rgb_of_highQ.get_1x_lr_params(self.backbone)) + list(self.head.parameters())

            net = VideoVNN(num_classes=args.num_classes)

        elif args.model == 'vnn_fusion':
            # RGB + Flow streams
            # This requires a more complex forward pass handling inputs=(rgb, flow)
            # For this factory, we return the container, but the training loop handles the tuple unpacking
            class VideoVNNFusion(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.model_rgb = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
                    self.model_of = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=2, pretrained=False)
                    self.model_fuse = vnn_fusion_highQ.VNN_F(num_classes=num_classes, num_ch=192, pretrained=False)

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
                    
            net = VideoVNNFusion(num_classes=args.num_classes)
        else:
             raise ValueError(f"Unknown Video model: {args.model}")
    
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return net.to(device)
