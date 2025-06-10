import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []
        for feature in features:
            layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
