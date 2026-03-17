import torch
import torch.nn as nn

class VNN_F(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN_F, self).__init__()

        Q1 = 2
        nch_out1 = 256 
        self.conv11 = nn.Conv3d(num_ch, nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(nch_out1)
        
        self.conv21 = nn.Conv3d(num_ch, 2*Q1*nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn21 = nn.BatchNorm3d(nch_out1)
        
        # Learnable gate (start small)
        self.gate1 = nn.Parameter(torch.ones(1) * 1e-4)
        
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc8 = nn.Linear(12544, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()
        
    def forward(self, x):
        Q1=2; nch_out1=256

        x11 = self.bn11(self.conv11(x))
        x21 = self.conv21(x)
        
        # Volterra Interaction
        left, right = x21[:, :Q1*nch_out1], x21[:, Q1*nch_out1:]
        interaction = (left * right).view(x.size(0), Q1, nch_out1, *x.shape[2:]).sum(dim=1)
        x21_add = self.bn21(interaction)
        
        # Gated Addition
        x = self.pool1(x11 + self.gate1 * x21_add)

        x = x.view(-1, 12544)
        x = self.dropout(x)
        logits = self.fc8(x)

        return logits
    
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
    skip = set(model.fc8.parameters())
    for p in model.parameters():
        if p.requires_grad and p not in skip:
            yield p

def get_10x_lr_params(model):
    for p in model.fc8.parameters():
        if p.requires_grad:
            yield p
