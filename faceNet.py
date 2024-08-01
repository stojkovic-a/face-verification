import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu


class FaceNet(nn.Module):
    def __init__(self, vector_length):
        super().__init__()

        self.c1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.short1 = nn.Sequential()

        self.c4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 512x512
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)  # od sad je 256x256

        self.c5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = relu(self.c1(x))
        x1 = relu(self.c2(x1))
        x1 = self.c3(x1)
        x1 += self.short1(x)
        x1 = relu(x1)

        # xe11 = relu(self.e11(x))
        # xe12 = relu(self.e12(xe11))
        # xp1 = self.pool1(xe12)

        # xe21 = relu(self.e21(xp1))
        # xe22 = relu(self.e22(xe21))
        # xp2 = self.pool2(xe22)

        # xe31 = relu(self.e31(xp2))
        # xe32 = relu(self.e32(xe31))
        # xp3 = self.pool3(xe32)

        # xe41 = relu(self.e41(xp3))
        # xe42 = relu(self.e42(xe41))
        # xp4 = self.pool4(xe42)

        # xe51 = relu(self.e51(xp4))
        # xe52 = relu(self.e52(xe51))

        # # Decoder
        # xu1 = self.upconv1(xe52)
        # xu11 = torch.cat([xu1, xe42], dim=1)
        # xd11 = relu(self.d11(xu11))
        # xd12 = relu(self.d12(xd11))

        # xu2 = self.upconv2(xd12)
        # xu22 = torch.cat([xu2, xe32], dim=1)
        # xd21 = relu(self.d21(xu22))
        # xd22 = relu(self.d22(xd21))

        # xu3 = self.upconv3(xd22)
        # xu33 = torch.cat([xu3, xe22], dim=1)
        # xd31 = relu(self.d31(xu33))
        # xd32 = relu(self.d32(xd31))

        # xu4 = self.upconv4(xd32)
        # xu44 = torch.cat([xu4, xe12], dim=1)
        # xd41 = relu(self.d41(xu44))
        # xd42 = relu(self.d42(xd41))

        # # Output layer
        # out = self.outconv(xd42)

        return out
