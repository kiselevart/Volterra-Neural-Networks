import torch
import torch.nn as nn

class VNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10, num_ch=3):
        super(VNN_CIFAR, self).__init__()
        
        # --- Block 1 ---
        # Input: (B, 3, 32, 32)
        Q1 = 4
        # Reduced channel counts for CIFAR to prevent overfitting and keep it lightweight
        nch_out1_5 = 16 
        nch_out1_3 = 16
        nch_out1_1 = 16
        sum_chans = nch_out1_5 + nch_out1_3 + nch_out1_1 # 48 channels

        # Linear Path (1st order)
        self.conv11_5 = nn.Conv2d(num_ch, nch_out1_5, kernel_size=3, padding=1)
        self.conv11_3 = nn.Conv2d(num_ch, nch_out1_3, kernel_size=3, padding=1)
        self.conv11_1 = nn.Conv2d(num_ch, nch_out1_1, kernel_size=1, padding=0)
        self.bn11 = nn.BatchNorm2d(sum_chans)

        # Quadratic Path (2nd order interaction)
        self.conv21_5 = nn.Conv2d(num_ch, 2*Q1*nch_out1_5, kernel_size=3, padding=1)
        self.conv21_3 = nn.Conv2d(num_ch, 2*Q1*nch_out1_3, kernel_size=3, padding=1)
        self.conv21_1 = nn.Conv2d(num_ch, 2*Q1*nch_out1_1, kernel_size=1, padding=0)
        
        # Interaction sum needs a BN too or we use the output BN
        self.bn21 = nn.BatchNorm2d(sum_chans)
        
        # 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 ---
        # Input: (B, 48, 16, 16)
        Q2 = 4
        nch_out2 = 64
        self.conv12 = nn.Conv2d(sum_chans, nch_out2, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(nch_out2)
        
        self.conv22 = nn.Conv2d(sum_chans, 2*Q2*nch_out2, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(nch_out2)
        
        # 16x16 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 3 ---
        # Input: (B, 64, 8, 8)
        Q3 = 4
        nch_out3 = 128
        self.conv13 = nn.Conv2d(nch_out2, nch_out3, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(nch_out3)
        
        self.conv23 = nn.Conv2d(nch_out2, 2*Q3*nch_out3, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm2d(nch_out3)
        
        # Keep spatial size 8x8 (No Pool) similar to original architecture or pool?
        # Let's pool to 4x4 to capture global context better for CIFAR
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # --- Block 4 ---
        # Input: (B, 128, 4, 4)
        Q4 = 4
        nch_out4 = 256
        self.conv14 = nn.Conv2d(nch_out3, nch_out4, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(nch_out4)
        
        self.conv24 = nn.Conv2d(nch_out3, 2*Q4*nch_out4, kernel_size=3, padding=1)
        self.bn24 = nn.BatchNorm2d(nch_out4)
        
        # 4x4 -> 2x2
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier ---
        # Flatten: 256 * 2 * 2 = 1024
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(nch_out4 * 2 * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def volterra_block(self, x, conv1, bn1, conv2, bn2, Q, nch_out):
        # 1. Linear Term
        x1 = bn1(conv1(x))
        
        # 2. Quadratic Term
        x2 = conv2(x)
        # Split features into two halves for interaction
        # x2 shape: [B, 2*Q*nch_out, H, W]
        # We want to multiply pairs.
        
        # The logic in original code:
        # x21_5mul = torch.mul(x21_5[:,0:Q1*nch_out1_5,...], x21_5[:,Q1*nch_out1_5:2*Q1*nch_out1_5,...])
        mid = Q * nch_out
        left = x2[:, :mid, :, :]
        right = x2[:, mid:, :, :]
        
        product = left * right # Element-wise mult
        
        # Sum chunks to reduce back to nch_out
        # In original: loop q in range(Q) and add slices
        res = torch.zeros_like(x1)
        for q in range(Q):
            # slice shape: [B, nch_out, H, W]
            res = res + product[:, q*nch_out : (q+1)*nch_out, :, :]
            
        x2_out = bn2(res)
        
        # 3. Sum
        return x1 + x2_out

    def forward(self, x):
        # --- Block 1 ---
        # Special case because of 3 parallel kernels (1x1, 3x3, 5x5 approx)
        Q1 = 2
        nch_1 = 16
        
        # Linear parts
        x11_5 = self.conv11_5(x)
        x11_3 = self.conv11_3(x)
        x11_1 = self.conv11_1(x)
        x11 = torch.cat((x11_5, x11_3, x11_1), 1)
        x11 = self.bn11(x11)
        
        # Interaction parts
        # 5x5 branch
        x21_5 = self.conv21_5(x)
        x21_5_prod = x21_5[:, :Q1*nch_1] * x21_5[:, Q1*nch_1:]
        x21_5_sum = torch.zeros_like(x11_5)
        for q in range(Q1):
            x21_5_sum += x21_5_prod[:, q*nch_1:(q+1)*nch_1]
            
        # 3x3 branch
        x21_3 = self.conv21_3(x)
        x21_3_prod = x21_3[:, :Q1*nch_1] * x21_3[:, Q1*nch_1:]
        x21_3_sum = torch.zeros_like(x11_3)
        for q in range(Q1):
            x21_3_sum += x21_3_prod[:, q*nch_1:(q+1)*nch_1]
            
        # 1x1 branch
        x21_1 = self.conv21_1(x)
        x21_1_prod = x21_1[:, :Q1*nch_1] * x21_1[:, Q1*nch_1:]
        x21_1_sum = torch.zeros_like(x11_1)
        for q in range(Q1):
            x21_1_sum += x21_1_prod[:, q*nch_1:(q+1)*nch_1]
            
        x21_total = torch.cat((x21_5_sum, x21_3_sum, x21_1_sum), 1)
        x21_total = self.bn21(x21_total)
        
        out1 = self.pool1(x11 + x21_total)
        
        # --- Block 2 ---
        out2 = self.volterra_block(out1, self.conv12, self.bn12, self.conv22, self.bn22, Q=2, nch_out=64)
        out2 = self.pool2(out2)
        
        # --- Block 3 ---
        out3 = self.volterra_block(out2, self.conv13, self.bn13, self.conv23, self.bn23, Q=2, nch_out=128)
        out3 = self.pool3(out3)
        
        # --- Block 4 ---
        out4 = self.volterra_block(out3, self.conv14, self.bn14, self.conv24, self.bn24, Q=2, nch_out=256)
        out4 = self.pool4(out4)
        
        # --- Classifier ---
        x = out4.view(out4.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits

if __name__ == "__main__":
    # Quick Test
    inputs = torch.randn(2, 3, 32, 32)
    model = VNN_CIFAR()
    out = model(inputs)
    print(f"Input: {inputs.shape}, Output: {out.shape}")
