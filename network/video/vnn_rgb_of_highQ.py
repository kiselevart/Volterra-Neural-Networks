import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class VNN(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN, self).__init__()
        
        # Block 1
        Q1 = 4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;
        sum_chans = nch_out1_5 + nch_out1_3 + nch_out1_1

        self.conv11_5 = nn.Conv3d(num_ch, nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_3 = nn.Conv3d(num_ch, nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_1 = nn.Conv3d(num_ch, nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn11 = nn.BatchNorm3d(sum_chans)

        # Spectral Norm on the quadratic expansion
        self.conv21_5 = spectral_norm(nn.Conv3d(num_ch, 2*Q1*nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.conv21_3 = spectral_norm(nn.Conv3d(num_ch, 2*Q1*nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.conv21_1 = spectral_norm(nn.Conv3d(num_ch, 2*Q1*nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        self.bn21 = nn.BatchNorm3d(sum_chans)

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 2
        Q2 = 4
        nch_out2 = 32
        self.conv12 = nn.Conv3d(sum_chans, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = spectral_norm(nn.Conv3d(sum_chans, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.bn22 = nn.BatchNorm3d(nch_out2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 3
        Q3 = 4
        nch_out3 = 64
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        self.conv23 = spectral_norm(nn.Conv3d(nch_out2, 2*Q3*nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.bn23 = nn.BatchNorm3d(nch_out3)

        # Block 4
        Q4 = 4
        nch_out4 = 96
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        self.conv24 = spectral_norm(nn.Conv3d(nch_out3, 2*Q4*nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.bn24 = nn.BatchNorm3d(nch_out4)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.__init_weight()

    def forward(self, x):
        # Block 1
        Q1=4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;
        
        x11 = self.bn11(torch.cat((self.conv11_5(x), self.conv11_3(x), self.conv11_1(x)), 1))
        
        # Volterra Interaction
        v21_5 = self.conv21_5(x)
        v21_3 = self.conv21_3(x)
        v21_1 = self.conv21_1(x)
        
        # Orthogonal-style interaction: 4xy - 2 (from vnn_ortho)
        def interact(v, Q, n):
            left, right = v[:, :Q*n], v[:, Q*n:]
            return (4.0 * (left * right) - 2.0).view(v.size(0), Q, n, *v.shape[2:]).sum(dim=1)

        x21_add = self.bn21(torch.cat((interact(v21_5, Q1, nch_out1_5), 
                                      interact(v21_3, Q1, nch_out1_3), 
                                      interact(v21_1, Q1, nch_out1_1)), 1))
        x = self.pool1(x11 + x21_add)

        # Block 2
        Q2=4; nch_out2=32
        x12 = self.bn12(self.conv12(x))
        x22 = self.bn22(interact(self.conv22(x), Q2, nch_out2))
        x = self.pool2(x12 + x22)

        # Block 3
        Q3=4; nch_out3=64
        x13 = self.bn13(self.conv13(x))
        x23 = self.bn23(interact(self.conv23(x), Q3, nch_out3))
        x = x13 + x23

        # Block 4
        Q4=4; nch_out4=96
        x14 = self.bn14(self.conv14(x))
        x24 = self.bn24(interact(self.conv24(x), Q4, nch_out4))
        x = self.pool4(x14 + x24)

        return x
 
    def __init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                # Check for spectral norm (PyTorch SN uses weight_orig)
                if hasattr(m, 'weight_orig'):
                    nn.init.orthogonal_(m.weight_orig)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    b = [model.conv11_5, model.conv11_3, model.conv11_1, model.bn11, 
         model.conv21_5, model.conv21_3, model.conv21_1, model.bn21, 
         model.conv12, model.bn12, model.conv22, model.bn22, 
         model.conv13, model.bn13, model.conv23, model.bn23, 
         model.conv14, model.bn14, model.conv24, model.bn24]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k
