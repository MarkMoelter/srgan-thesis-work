import math

import torch
import torch.nn as nn

from .residual_block import ResidualBlock
from .upsample_block import UpsampleBlock


class Generator(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(9, 9), padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64)
        )

        # create a number of upsample blocks in a list
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        # unpack upsample blocks and apply final conv layer
        self.block8 = nn.Sequential(*block8, nn.Conv2d(64, 3, kernel_size=(9, 9), padding=4))

    def forward(self, features):
        b1 = self.block1(features)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        b6 = self.block6(b5)
        b7 = self.block7(b6)
        b8 = self.block8(b1 + b7)
        return (torch.tanh(b8) + 1) / 2
