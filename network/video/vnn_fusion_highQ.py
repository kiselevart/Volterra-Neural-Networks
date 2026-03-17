import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class VNN_F(nn.Module):
    def __init__(self, num_classes, num_ch = 3, pretrained=False):
        super(VNN_F, self).__init__()

        Q1 = 2
        nch_out1 = 256 
        self.conv11 = nn.Conv3d(num_ch, nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(nch_out1)
        
        # Spectral Normalization for stability without activations
        self.conv21 = spectral_norm(nn.Conv3d(num_ch, 2*Q1*nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
        self.bn21 = nn.BatchNorm3d(nch_out1)
        
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc8 = nn.Linear(12544, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()
        
    def forward(self, x):
        Q1=2
        nch_out1 = 256

        x11 = self.bn11(self.conv11(x))
        x21 = self.conv21(x)
        
        # Orthogonal Interaction: 4xy - 2
        left, right = x21[:, :Q1*nch_out1], x21[:, Q1*nch_out1:]
        interaction = (4.0 * (left * right) - 2.0).view(x.size(0), Q1, nch_out1, *x.shape[2:]).sum(dim=1)
        x21_add = self.bn21(interaction)
        
        # Activation-free Fusion
        x = self.pool1(x11 + x21_add)

        x = x.view(-1, 12544)
        x = self.dropout(x)
        logits = self.fc8(x)

        return logits
    
    def __init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
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
    b = [model.conv11, model.bn11, model.conv21, model.bn21]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

def get_10x_lr_params(model):
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
