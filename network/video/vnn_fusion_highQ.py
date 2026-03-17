import torch
import torch.nn as nn
from mypath import Path

class VNN_F(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN_F, self).__init__()
#         Q0 = 2
#         nch_out0 = 96 
#         self.conv10 = nn.Conv3d(num_ch, nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
#         self.bn10 = nn.BatchNorm3d(nch_out0)
#         self.conv20 = nn.Conv3d(num_ch, 2*Q0*nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
# #         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn20 = nn.BatchNorm3d(nch_out0)

        Q1 = 2
        nch_out1 = 256 
        self.conv11 = nn.Conv3d(num_ch, nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(nch_out1)
        self.conv21 = nn.Conv3d(num_ch, 2*Q1*nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn21 = nn.BatchNorm3d(nch_out1)

#         Q1_red = 2
#         nch_out1_red = 96
#         self.conv11_red = nn.Conv3d(nch_out1, nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
#         self.bn11_red = nn.BatchNorm3d(nch_out1_red)
#         self.conv21_red = nn.Conv3d(nch_out1, 2*Q1_red*nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
# #         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn21_red = nn.BatchNorm3d(nch_out1_red)

#         Q2 = 2
#         nch_out2 = 512 
#         self.conv12 = nn.Conv3d(nch_out1_red, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.bn12 = nn.BatchNorm3d(nch_out2)
#         self.conv22 = nn.Conv3d(nch_out1_red, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.bn22 = nn.BatchNorm3d(nch_out2)
        
        self.fc8 = nn.Linear(12544, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.gate1 = nn.Parameter(torch.ones(1) * 1e-4)

        self.__init_weight()
        
    def forward(self, x, activation = False):
        Q1=2
        nch_out1 = 256

        x11 = self.bn11(self.conv11(x))
        x21 = self.conv21(x)
        
        x21_mul = x21[:, :Q1*nch_out1] * x21[:, Q1*nch_out1:]
        x21_add = self.bn21(x21_mul.view(x21_mul.size(0), Q1, nch_out1, *x21_mul.shape[2:]).sum(dim=1))
        
        # Addition (No ReLU)
        x = x11 + self.gate1 * x21_add
        x = self.pool1(x)

        x = x.view(-1, 12544)
        
        
        x = self.dropout(x)
     
        logits = self.fc8(x)

        return logits
    
    def __init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                # Use Xavier with small gain for multiplicative branch
                if 'conv2' in name:
                    torch.nn.init.xavier_normal_(m.weight, gain=0.01)
                else:
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                if 'bn2' in name:
                    m.weight.data.fill_(0.1) # Scale down the quadratic branch impact
                else:
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
        
def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv11, model.bn11, model.conv21, model.bn21] #, model.conv11_red, model.bn11_red, model.conv21_red, model.bn21_red, model.conv11, model.bn12, model.conv22, model.bn22] # model.conv10, model.bn10, model.conv20, model.bn20, 
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k



if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())