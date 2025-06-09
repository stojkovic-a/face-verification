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

