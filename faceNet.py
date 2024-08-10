import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu, normalize

CUDA_LAUNCH_BLOCKING = 1


class FaceNet(nn.Module):
    def __init__(self, vector_length):
        super().__init__()

        self.c1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(24, 40, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 250x250
        self.b1 = nn.BatchNorm2d(64)
        

        self.short1 = nn.Sequential()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 125x125

        self.c5 = nn.Conv2d(64, 112, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(112, 112, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(112, 144, kernel_size=3, padding=1)
        self.c8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(256)

        self.short2 = nn.Sequential()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)  # sada je 62x62

        self.c9 = nn.Conv2d(256, 448, kernel_size=3, padding=1)
        self.c10 = nn.Conv2d(448, 448, kernel_size=3, padding=1)
        self.c11 = nn.Conv2d(448, 576, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(1024)

        self.short3 = nn.Sequential()
        # self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)  # sada je 31x31

        # self.c13 = nn.Conv2d(1024, 1792, kernel_size=3, padding=1)
        # self.c14 = nn.Conv2d(1792, 1792, kernel_size=3, padding=1)
        # self.c15 = nn.Conv2d(1792, 2304, kernel_size=3, padding=1)
        # self.c16 = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)

        self.short4 = nn.Sequential()
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        # self.lin = nn.Linear(4096, vector_length)
        self.lin = nn.Linear(1024, vector_length)

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
        x2 = self.c2(x1)
        x2 += self.short1(x1)
        x2 = relu(x2)
        x2 = relu(self.c3(x2))
        x2 = torch.cat([x2, x1], dim=1)
        x2 = relu(self.b1(self.c4(x2)))
        x2 = self.mp1(x2)

        x2 = relu(self.c5(x2))
        x3 = self.c6(x2)
        x3 += self.short2(x2)
        x3 = relu(x3)
        x3 = relu(self.c7(x3))
        x3 = torch.cat([x3, x2], dim=1)
        x3 = relu(self.b2(self.c8(x3)))
        x3 = self.mp2(x3)

        x3 = relu(self.c9(x3))
        x4 = self.c10(x3)
        x4 += self.short3(x3)
        x4 = relu(x4)
        x4 = relu(self.c11(x4))
        x4 = torch.cat([x4, x3], dim=1)
        x4 = relu(self.b3(self.c12(x4)))
        # x4 = self.mp3(x4)

        # x4 = relu(self.c13(x4))
        # x5 = self.c14(x4)
        # x5 += self.short4(x4)
        # x5 = relu(x5)
        # x5 = relu(self.c15(x5))
        # x5 = torch.cat([x5, x4], dim=1)
        # x5 = relu(self.c16(x5))

        # x5 = self.aap(x5)
        # out = x5.reshape((x5.size(0), -1))
        # out = self.lin(out)
        # out = normalize(out, p=2, dim=1)
        # return out

        x4 = self.aap(x4)
        out = x4.reshape((x4.size(0), -1))
        out = self.lin(out)
        out = normalize(out, p=2, dim=1)
        return out
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
