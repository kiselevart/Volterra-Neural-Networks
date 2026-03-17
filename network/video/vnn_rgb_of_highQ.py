import torch
import torch.nn as nn
from mypath import Path

class VNN(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN, self).__init__()
        Q1 = 4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;
        sum_chans = nch_out1_5+nch_out1_3+nch_out1_1

        self.conv11_5 = nn.Conv3d(num_ch, nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_3 = nn.Conv3d(num_ch, nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv11_1 = nn.Conv3d(num_ch, nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.bn11 = nn.BatchNorm3d(sum_chans)

        self.conv21_5 = nn.Conv3d(num_ch, 2*Q1*nch_out1_5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_3 = nn.Conv3d(num_ch, 2*Q1*nch_out1_3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv21_1 = nn.Conv3d(num_ch, 2*Q1*nch_out1_1, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.bn21 = nn.BatchNorm3d(sum_chans)

        
        Q2 = 4
        nch_out2 = 32
        self.conv12 = nn.Conv3d(nch_out1_5+nch_out1_3+nch_out1_1, nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn12 = nn.BatchNorm3d(nch_out2)
        self.conv22 = nn.Conv3d(nch_out1_5+nch_out1_3+nch_out1_1, 2*Q2*nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn22 = nn.BatchNorm3d(nch_out2)

        Q3 = 4
        nch_out3 = 64
        self.conv13 = nn.Conv3d(nch_out2, nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn13 = nn.BatchNorm3d(nch_out3)
        self.conv23 = nn.Conv3d(nch_out2, 2*Q3*nch_out3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn23 = nn.BatchNorm3d(nch_out3)


        Q4 = 4
        nch_out4 = 96
        self.conv14 = nn.Conv3d(nch_out3, nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn14 = nn.BatchNorm3d(nch_out4)
        self.conv24 = nn.Conv3d(nch_out3, 2*Q4*nch_out4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn24 = nn.BatchNorm3d(nch_out4)
        
        # Q5 = 2
        # nch_out5 = 256
        # self.conv15 = nn.Conv3d(nch_out4, nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn15 = nn.BatchNorm3d(nch_out5)
        # self.conv25 = nn.Conv3d(nch_out4, 2*Q5*nch_out5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.bn25 = nn.BatchNorm3d(nch_out5)

        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
#         self.fc8 = nn.Linear(25088, num_classes)

#         self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        # Learnable gates to ease in the quadratic branches
        self.gate1 = nn.Parameter(torch.ones(1) * 1e-4)
        self.gate2 = nn.Parameter(torch.ones(1) * 1e-4)
        self.gate3 = nn.Parameter(torch.ones(1) * 1e-4)
        self.gate4 = nn.Parameter(torch.ones(1) * 1e-4)

        self.__init_weight()

        # if pretrained:
        #     self.__load_pretrained_weights()


    def forward(self, x, activation = False):
        
        Q1=4
        nch_out1_5 = 8; nch_out1_3 = 8; nch_out1_1 = 8;
        sum_chans = nch_out1_5 + nch_out1_3 + nch_out1_1

        x11_5 = self.conv11_5(x) 
        x11_3 = self.conv11_3(x) 
        x11_1 = self.conv11_1(x) 
        x11 = torch.cat((x11_5, x11_3, x11_1), 1)
        x11 = self.bn11(x11)
        
        # Quadratic branch 1
        x21_5 = self.conv21_5(x)
        x21_5mul = x21_5[:, :Q1*nch_out1_5] * x21_5[:, Q1*nch_out1_5:]
        x21_5add = x21_5mul.view(x21_5mul.size(0), Q1, nch_out1_5, *x21_5mul.shape[2:]).sum(dim=1)

        x21_3 = self.conv21_3(x)
        x21_3mul = x21_3[:, :Q1*nch_out1_3] * x21_3[:, Q1*nch_out1_3:]
        x21_3add = x21_3mul.view(x21_3mul.size(0), Q1, nch_out1_3, *x21_3mul.shape[2:]).sum(dim=1)

        x21_1 = self.conv21_1(x)
        x21_1mul = x21_1[:, :Q1*nch_out1_1] * x21_1[:, Q1*nch_out1_1:]
        x21_1add = x21_1mul.view(x21_1mul.size(0), Q1, nch_out1_1, *x21_1mul.shape[2:]).sum(dim=1)

        x21_add = torch.cat((x21_5add, x21_3add, x21_1add), 1)
        x21_add = self.bn21(x21_add)

        # Addition (No ReLU) + Pooling
        x = x11 + self.gate1 * x21_add
        x = self.pool1(x)

        Q2=4
        nch_out2 = 32 
        x12 = self.bn12(self.conv12(x))
        x22 = self.conv22(x)
        x22_mul = x22[:, :Q2*nch_out2] * x22[:, Q2*nch_out2:]
        x22_add = self.bn22(x22_mul.view(x22_mul.size(0), Q2, nch_out2, *x22_mul.shape[2:]).sum(dim=1))
        x = x12 + self.gate2 * x22_add
        x = self.pool2(x)

        Q3=4
        nch_out3 = 64
        x13 = self.bn13(self.conv13(x))
        x23 = self.conv23(x)
        x23_mul = x23[:, :Q3*nch_out3] * x23[:, Q3*nch_out3:]
        x23_add = self.bn23(x23_mul.view(x23_mul.size(0), Q3, nch_out3, *x23_mul.shape[2:]).sum(dim=1))
        x = x13 + self.gate3 * x23_add

        Q4=4
        nch_out4 = 96
        x14 = self.bn14(self.conv14(x) )
        x24 = self.conv24(x)
        x24_mul = x24[:, :Q4*nch_out4] * x24[:, Q4*nch_out4:]
        x24_add = self.bn24(x24_mul.view(x24_mul.size(0), Q4, nch_out4, *x24_mul.shape[2:]).sum(dim=1))
        x = x14 + self.gate4 * x24_add
        x = self.pool4(x)

        return x
 
    def __init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                # Use Xavier for polynomial layers; scale down the quadratic branches (conv2*)
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
    b = [model.conv11_5, model.conv11_3, model.conv11_1, model.bn11, model.conv21_5, model.conv21_3, model.conv21_1, model.bn21, model.conv12, model.bn12, model.conv22, model.bn22, model.conv13, model.bn13, model.conv23, model.bn23, model.conv14, model.bn14, model.conv24, model.bn24] #, model.fc6, model.fc7]
    
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last fc layer of the net.
#     """
#     b = [model.fc8]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k



if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())