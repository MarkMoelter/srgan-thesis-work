import torch.nn as nn

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, up_scale: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * up_scale ** 2,
            kernel_size=(3, 3),
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, features):
        x = self.conv(features)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
