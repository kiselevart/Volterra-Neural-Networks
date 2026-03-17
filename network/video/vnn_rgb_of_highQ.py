import torch
import torch.nn as nn

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

        self.conv21_5 = nn.Conv3d(num_ch, 2*Q1*nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_3 = nn.Conv3d(num_ch, 2*Q1*nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_1 = nn.Conv3d(num_ch, 2*Q1*nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn21 = nn.BatchNorm3d(sum_chans)
        
        # Learnable gate for quadratic contribution (start small for stability)
        self.gate1 = nn.Parameter(torch.ones(1) * 1e-4)

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 2
        Q2 = 4
        nch_out2 = 32
        self.conv12 = nn.Conv3d(sum_chans, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = nn.Conv3d(sum_chans, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn22 = nn.BatchNorm3d(nch_out2)
        self.gate2 = nn.Parameter(torch.ones(1) * 1e-4)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 3
        Q3 = 4
        nch_out3 = 64
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        self.conv23 = nn.Conv3d(nch_out2, 2*Q3*nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn23 = nn.BatchNorm3d(nch_out3)
        self.gate3 = nn.Parameter(torch.ones(1) * 1e-4)

        # Block 4
        Q4 = 4
        nch_out4 = 96
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        self.conv24 = nn.Conv3d(nch_out3, 2*Q4*nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn24 = nn.BatchNorm3d(nch_out4)
        self.gate4 = nn.Parameter(torch.ones(1) * 1e-4)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.__init_weight()

    def _interact(self, v, Q, n):
        left, right = v[:, :Q*n], v[:, Q*n:]
        # Standard Volterra (x*y)
        return (left * right).view(v.size(0), Q, n, *v.shape[2:]).sum(dim=1)

    def forward(self, x):
        # Block 1
        Q1=4; nch_out1_5=8; nch_out1_3=8; nch_out1_1=8
        x11 = self.bn11(torch.cat((self.conv11_5(x), self.conv11_3(x), self.conv11_1(x)), 1))
        x21 = self.bn21(torch.cat((self._interact(self.conv21_5(x), Q1, nch_out1_5), 
                                  self._interact(self.conv21_3(x), Q1, nch_out1_3), 
                                  self._interact(self.conv21_1(x), Q1, nch_out1_1)), 1))
        x = self.pool1(x11 + self.gate1 * x21)

        # Block 2
        Q2=4; nch_out2=32
        x12 = self.bn12(self.conv12(x))
        x22 = self.bn22(self._interact(self.conv22(x), Q2, nch_out2))
        x = self.pool2(x12 + self.gate2 * x22)

        # Block 3
        Q3=4; nch_out3=64
        x13 = self.bn13(self.conv13(x))
        x23 = self.bn23(self._interact(self.conv23(x), Q3, nch_out3))
        x = x13 + self.gate3 * x23

        # Block 4
        Q4=4; nch_out4=96
        x14 = self.bn14(self.conv14(x))
        x24 = self.bn24(self._interact(self.conv24(x), Q4, nch_out4))
        x = self.pool4(x14 + self.gate4 * x24)

        return x
 
    def __init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                if 'conv2' in name:
                    nn.init.xavier_normal_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p
