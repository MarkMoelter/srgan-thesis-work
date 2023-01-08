import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.b_norm1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.b_norm2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.b_norm3 = nn.BatchNorm2d(channels)

    def forward(self, features):
        x = self.conv1(features)
        x = self.b_norm1(x)
        x = self.prelu1(x)

        # added another conv, bn, PRelu step
        x = self.conv2(x)
        x = self.b_norm2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.b_norm3(x)
        return features + x
